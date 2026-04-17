# Qwen3-TTS 0.6B Voice Clone

> **Status: Implemented** — all planned work is complete. This doc describes the
> design and what was built.

## Goal

A second variant of the Qwen3-TTS backend that performs **voice cloning** from a
reference audio clip (3–10 s) plus its transcript, using the
[`Qwen/Qwen3-TTS-12Hz-0.6B-Base`][hf-base] checkpoint.

The existing 1.7B VoiceDesign backend is untouched. The two coexist as sibling
structs: `Qwen3Tts` (1.7B VoiceDesign) and `Qwen3TtsClone` (0.6B Base).

The user provides:

- `text` — what to say
- `ref_samples` — short waveform (24 kHz mono float32 PCM)
- `ref_text` — transcript of the reference audio (required for ICL mode)
- `language` — target language (optional, defaults to `"en"`)

The model returns audio in the cloned voice. This first pass covers the **0.6B**
size; the 1.7B Base variant uses the same architecture and can be added later by
swapping model files.

## Why a separate variant

Voice clone is **not** just "VoiceDesign with a different prompt". The Base
checkpoint differs from VoiceDesign in three ways that matter for inference:

| | VoiceDesign (1.7B, current) | Base (0.6B, this work) |
|---|---|---|
| Talker hidden | 2048 | 1024 |
| Talker layers | 28 | 28 |
| Talker KV heads | 8 | 8 |
| Speaker encoder | none | ECAPA-TDNN, mel→1024-dim embed |
| Speech tokenizer encoder | unused at inference | **used** to encode ref_audio |
| Prefill prefix | text-only instruction tokens | speaker_embed + ref_text + ref_codes interleaving (ICL) |

So we need (a) one or more new ONNX assets, (b) new prefill construction logic
in Rust, and (c) a new request entry point (`synthesize_clone`) that takes
reference audio.

## Architecture: how voice clone works

Source of truth: [`generate_voice_clone`][gen_clone] and
[`generate_icl_prompt`][gen_icl] in `qwen_tts/inference/qwen3_tts_model.py`
and `qwen_tts/core/models/modeling_qwen3_tts.py`.

### Reference-audio preprocessing (host code, runs once per ref clip)

```
ref_audio (any sr)
    │
    ├──► resample to 24 kHz ──► mel-spectrogram (n_fft=1024, hop=256, win=1024,
    │                                            n_mels=128, fmin=0, fmax=12000)
    │                              │
    │                              ▼
    │                      [Speaker Encoder]      ECAPA-TDNN
    │                              │
    │                              ▼
    │                       ref_spk_embedding (1024,)
    │
    └──► [Speech Tokenizer Encoder]  Mimi-based (12 Hz, 16 quantizer groups)
                  │
                  ▼
           ref_code  (T_ref, 16)        i64 codebook indices
```

ICL mode uses **both** outputs. `x_vector_only_mode` uses only the speaker
embedding (lower quality, no transcript needed). We implement ICL first.

### Prefill embedding construction (Base + ICL)

```
talker_input_embed = concat over seq dim:

  [ im_start, assistant, \n ]                              text_proj only         (3)
  [ tts_pad + codec_embed[think]       ]
  [ tts_pad + codec_embed[think_bos]   ]
  [ tts_pad + codec_embed[lang_id]     ]                  codec think prefix      (4)
  [ tts_pad + codec_embed[think_eos]   ]
  [ speaker_embed + codec_embed[codec_pad]  ]              speaker slot           (1)
  [ tts_bos      + codec_embed[codec_pad]   ]              transition             (1)

  ICL block (non_streaming, len = max(text_lens, codec_lens)):
    text_part = [ text_proj(ref_id ++ text_id), tts_eos ]                  (T1)
    codec_part = [ codec_embed[codec_bos], Σ_g codec_embed_g[ref_code[:,g]] ] (T2 = 1 + T_ref)

    text_part  +=  codec_embed[codec_pad]   (broadcast over T1)
    codec_part +=  tts_pad                  (broadcast over T2)

    icl_input = concat([text_part, codec_part], dim=1)
```

Two crucial differences from VoiceDesign:

1. The codec prefix carries an **extra speaker slot** (`speaker_embed +
   codec_pad`) inserted after `think_eos` and before the transition. This is
   how the model receives the cloned-voice condition.
2. The text path (`text_proj(ref_id ++ text_id) + tts_eos`) is **summed
   element-wise** with the codec path (`codec_bos + Σ ref_code_embeds`) along
   the sequence dim, using `codec_pad` / `tts_pad` to pad whichever is shorter.
   This is the in-context-learning mechanism — the ref text and ref codes are
   delivered together as a single positional stream the model has been trained
   to "continue".

### Decode loop

After prefill, the autoregressive decode loop is **identical** to VoiceDesign:
sample group-0 from talker logits, run code predictor for groups 1–15, sum 16
codec embeddings + `trailing_text_hidden` (= `tts_pad` in non-streaming) for the
next step, stop on `codec_eos`.

### Vocoder / output trim

The vocoder is run on `concat([ref_code, generated_codes])` so the model's
generated tail blends smoothly with the reference timbre. Then we **cut off**
the leading portion proportional to `ref_len / total_len` so the returned
waveform contains only the new content. (See lines 612–631 in
`qwen3_tts_model.py`.)

## ONNX assets

The 0.6B Base checkpoint provides a different talker plus extra modules (speaker
encoder, tokenizer encoder). All are exported via `tools/qwen3-tts-onnx/`.

### Reused (re-exported with smaller dims)

| File | Notes |
|---|---|
| `talker_prefill.onnx` | hidden=1024, vocab=3072, 28 layers, 8 KV heads |
| `talker_decode.onnx`  | same |
| `code_predictor.onnx` | 1024 hidden, 5 layers |
| `vocoder.onnx`        | Mimi v2 decoder |
| `embeddings/text_embedding.npy` etc. | re-extracted from Base weights |
| `embeddings/cp_codec_embedding_0..14.npy` | re-extracted |

### New for voice clone

| File | Source | Shape | Purpose |
|---|---|---|---|
| `speaker_encoder.onnx` | `model.speaker_encoder` (ECAPA-TDNN) | in: `(1, T_mel, 128)` mel; out: `(1, 1024)` | Encode ref audio → speaker embed |
| `tokenizer_encoder.onnx` | `model.speech_tokenizer.encode()` | in: `(1, 1, S_audio)` waveform; out: `(1, T_codes, 16)` i64 | Encode ref audio → ref codes |

Mel computation is done in host code (pure-Rust STFT + mel filterbank in
`mel.rs`) rather than a separate ONNX. Params match the reference exactly:
`n_fft=1024, hop=256, win=1024, n_mels=128, fmin=0, fmax=12000`, `center=False`,
log on top.

### Export tooling

`tools/qwen3-tts-onnx/` contains:

- `export_speaker_encoder.py` — ECAPA-TDNN, dynamic axis on mel time dim,
  opset 17. Validates against PyTorch (atol=1e-4).
- `export_tokenizer_encoder.py` — Mimi-based tokenizer encoder. Uses JIT trace
  with fixed size (240k samples = 10 s @ 24 kHz). Applies `mask_patch.py` to
  handle Mimi's causal mask incompatibility with tracing. Validates exact i64
  code match.
- `mask_patch.py` — patches Mimi causal mask for JIT tracing compatibility.
- `generate_clone_onnx.py` — end-to-end Python ONNX voice clone reference
  (547 lines). Loads all 6 ONNX sessions, builds ICL prefill, runs decode loop,
  vocoders, and trims the reference portion.
- Existing `export_talker.py`, `export_code_predictor.py`, `export_vocoder.py`,
  `export_embeddings.py` are reused with `MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base`.
- `quantize_int4.py` quantizes talker/CP/vocoder; speaker and tokenizer encoders
  stay FP32 (small, conditioning path — no upside to quantizing).

### Makefile targets

```
make clone-all              # full export + quantize + HF packaging
make clone-export           # orchestrate 6 component exports
make clone-base-preset      # INT4 quantization (encoders stay FP32)
make clone-hf               # package for HF Hub
make clone-fixture          # regenerate ref WAV fixture
make clone-generate         # test FP32 output
```

### Published HF repo

[`wavekat/Qwen3-TTS-0.6B-Base-ONNX`](https://huggingface.co/wavekat/Qwen3-TTS-0.6B-Base-ONNX)
— separate from the existing 1.7B repo. Layout:

```
speaker_encoder.onnx          FP32 only
tokenizer_encoder.onnx        FP32 only
fp32/                         talker_prefill, talker_decode, code_predictor, vocoder
int4/                         same (INT4 quantized)
embeddings/                   text_embedding, text_projection, codec embeddings
tokenizer/                    vocab.json, merges.txt
config.json
```

## Rust backend

### Module structure

```
crates/wavekat-tts/src/backends/qwen3_tts/
├── mod.rs              — Qwen3Tts (1.7B) + Qwen3TtsClone (0.6B) + CloneRequest
├── download.rs         — resolve_model_dir + resolve_clone_model_dir
├── model.rs            — existing talker/CP/vocoder pipeline (untouched)
├── clone_model.rs      — CloneModel: 6 ONNX sessions + ICL prefill builder
├── mel.rs              — pure-Rust STFT + mel filterbank (realfft)
├── tokenizer.rs        — shared text tokenization (unchanged)
└── sampler.rs          — shared sampling logic (unchanged)
```

### Public API

```rust
use wavekat_tts::backends::qwen3_tts::{Qwen3TtsClone, CloneRequest, ModelConfig};

let tts = Qwen3TtsClone::new()?;                          // 0.6B Base, INT4, CPU
let req = CloneRequest::new("Text to say", &pcm_24k, 24000, "transcript of ref")
    .with_language("en");
let frame: AudioFrame = tts.synthesize_clone(&req)?;
```

`Qwen3TtsClone` exposes `fn synthesize_clone(&self, req: &CloneRequest) ->
Result<AudioFrame<'static>, TtsError>`. The existing `TtsBackend` contract
doesn't fit (no place for a reference clip), so `Qwen3TtsClone` has its own
method rather than implementing `TtsBackend`.

`CloneRequest`:
```rust
pub struct CloneRequest<'a> {
    pub text: &'a str,
    pub ref_samples: &'a [f32],     // 24 kHz mono float32 PCM
    pub ref_sample_rate: u32,       // must be 24000
    pub ref_text: &'a str,          // required for ICL mode
    pub language: Option<&'a str>,  // defaults to "en"
}
```

`ModelConfig` is shared between both backends (precision, EP, model dir).

### Implementation

`clone_model.rs` (`CloneModel`) handles the full pipeline:

1. **`encode_speaker()`** — computes mel spectrogram via `mel.rs`, runs
   `speaker_encoder.onnx` → `(1, 1024)` speaker embedding.
2. **`encode_ref_codes()`** — runs `tokenizer_encoder.onnx` on raw PCM →
   `(T_ref, 16)` i64 codebook indices.
3. **`build_icl_prefill()`** — constructs the prefill embedding tensor:
   role prefix → codec prefix → speaker slot → transition → ICL block
   (text + codec interleaving). Produces `(1, T, 1024)`.
4. **`run_talker_prefill()` / `run_talker_decode()`** — talker pipeline
   (same structure as 1.7B, adapted to 1024-dim hidden).
5. **`run_code_predictor()`** — predicts codec groups 1–15 from group 0.
6. **`run_vocoder_clone()`** — prepends ref codes before vocoding, then trims
   the leading portion proportional to `ref_len / total_len`.

### Mel-spectrogram in Rust

`mel.rs` — pure-Rust implementation using `realfft`:
- `MelSpectrogram` struct with precomputed Hann window + Slaney mel filterbank
- `.compute(audio) → (T, 128)` log-mel frames
- Params match the reference: `n_fft=1024, hop=256, win=1024, n_mels=128,
  fmin=0, fmax=12000, center=False`
- Unit tests for scale roundtrip, filterbank shape, output shape

### Audio I/O and resampling

`CloneRequest` takes raw `&[f32]` PCM samples. Reading WAV files is the
caller's responsibility (example uses `hound`). Sample rate must be 24 kHz —
validated at runtime with a clear error. Resampling is not done internally;
callers resample before calling.

## Example

`examples/synthesize_clone.rs` — full CLI example (165 lines) with `--ref-audio`,
`--ref-text`, `--text`, `--language`, `--precision`, `--provider` flags. Reads a
24 kHz mono WAV, runs `Qwen3TtsClone::synthesize_clone`, writes output, and
prints RTF.

## CI

`.github/workflows/export-onnx.yml` — unified workflow with a variant selector
dropdown (`voicedesign` | `clone`). The `clone` variant runs `clone-export`,
`clone-base-preset` (INT4 quantization, encoders FP32), and `clone-hf`
(HF Hub packaging). Conditional validation and cleanup between steps for
runner disk space.

## Verification

1. **Per-component parity** (in export scripts):
   - speaker_encoder ONNX vs PyTorch on a fixed mel: max abs err < 1e-4
   - tokenizer_encoder ONNX vs PyTorch on a fixed wav: identical i64 codes
2. **End-to-end ONNX (Python)** via `generate_clone_onnx.py`:
   - Loads all 6 ONNX sessions, builds ICL prefill, runs decode + vocoder
   - Supports FP32 and INT4 variants, all 10 languages
3. **Rust vs Python parity**:
   - Same ICL prefill layout, same vocoder trim logic
4. **Reference fixture**: `tools/qwen3-tts-onnx/fixtures/ref_clone.wav` for
   integration tests and examples.

## Decisions made

1. **API shape**: sibling struct `Qwen3TtsClone` + `CloneRequest`. Existing
   `Qwen3Tts` and `TtsBackend` unchanged.
2. **Speaker / tokenizer encoder precision**: FP32 only. They're small and
   sit on the conditioning path — no upside to quantizing.
3. **HF repo**: separate `wavekat/Qwen3-TTS-0.6B-Base-ONNX` (does not reuse
   the 1.7B repo).
4. **Mel in host code**: pure-Rust (`realfft`) rather than a separate ONNX
   model. Deterministic, small, and avoids an extra session.
5. **No internal resampling**: callers must provide 24 kHz audio. Keeps the
   backend simple and avoids pulling in `rubato`.
6. **Tokenizer encoder uses fixed-size JIT trace** (240k samples) rather than
   dynamic axes, due to Mimi causal mask incompatibility with tracing.

## Future work

- 1.7B Base voice clone (same architecture; swap model dir)
- `x_vector_only_mode` (no `ref_text`) — speaker-embedding-only conditioning
- Streaming voice clone
- True batch inference

[hf-base]: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
[gen_clone]: https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py
[gen_icl]: https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py
