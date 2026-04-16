# Qwen3-TTS 0.6B Voice Clone (Plan)

> **Status: Planning** — no implementation yet. This doc captures the approach
> before any code is written so we can review it.

## Goal

Add a second variant of the Qwen3-TTS backend that performs **voice cloning**
from a reference audio clip (3–10 s) plus its transcript, using the
[`Qwen/Qwen3-TTS-12Hz-0.6B-Base`][hf-base] checkpoint.

We keep the existing 1.7B VoiceDesign backend untouched. The two coexist behind
the same `Qwen3Tts` Rust struct (selectable via `ModelConfig`), or we expose a
sibling `Qwen3TtsClone` struct — see "Rust API surface" below.

The user provides:

- `text` — what to say
- `language` — target language
- `ref_audio` — short waveform (path, URL, or PCM samples)
- `ref_text` — transcript of `ref_audio` (required for ICL mode)

The model returns audio in the cloned voice. We focus on the **0.6B** size for
this first pass; the 1.7B Base variant uses the same architecture and can be
added later by swapping model files.

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

## What ONNX assets we need

The 0.6B Base repo gives us a different talker checkpoint plus extra modules
(speaker encoder, tokenizer encoder). We export them like the existing 1.7B
pipeline.

### Reused (export with the same scripts, smaller dims)

| File | Notes |
|---|---|
| `talker_prefill.onnx` | hidden=1024, vocab=3072, 28 layers, 8 KV heads — re-run `export_talker.py` against the Base repo |
| `talker_decode.onnx`  | same |
| `code_predictor.onnx` | unchanged across variants (1024 hidden, 5 layers) |
| `vocoder.onnx`        | same Mimi v2 decoder |
| `embeddings/text_embedding.npy` etc. | re-extract from Base weights |
| `embeddings/cp_codec_embedding_0..14.npy` | re-extract |

### New for voice clone

| File | Source | Shape | Purpose |
|---|---|---|---|
| `speaker_encoder.onnx` | `model.speaker_encoder` (ECAPA-TDNN) | in: `(1, T_mel, 128)` mel; out: `(1, 1024)` | Encode ref audio → speaker embed |
| `tokenizer_encoder.onnx` | `model.speech_tokenizer.encode()` | in: `(1, 1, S_audio)` waveform; out: `(1, T_codes, 16)` i64 | Encode ref audio → ref codes |

Mel computation stays in host code (a Rust port — STFT + mel filterbank). It is
deterministic and small; not worth a separate ONNX. We use the same params as
the reference: `n_fft=1024, hop=256, win=1024, n_mels=128, fmin=0, fmax=12000`,
`center=False`, log on top.

### Optional: keep host code lighter with a combined `clone_preprocessor.onnx`

Decision deferred. Splitting keeps each ONNX small and lets us swap mel impls.

## ONNX export plan

Add to `tools/qwen3-tts-onnx/`:

1. `export_speaker_encoder.py` — wrap `model.speaker_encoder` (forward expects
   mel `(B, T_mel, 128)`), trace with dynamic axis on `T_mel`, opset 17.
2. `export_tokenizer_encoder.py` — wrap `model.speech_tokenizer.encode()`. The
   encoder is `MimiModel`-based (transformers); it traces cleanly with dynamo.
   Dynamic axis on the audio sample dim. Output the per-frame code matrix only
   (drop the `Qwen3TTSTokenizerV2EncoderOutput` wrapper).
3. Re-use `export_talker.py`, `export_code_predictor.py`, `export_vocoder.py`,
   `export_embeddings.py` against `MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base`.
   They already read dimensions from the loaded config, so should "just work";
   if not, the dim-handling is a small patch.
4. `quantize_int4.py` — extend to also quantize the new
   `speaker_encoder.onnx` / `tokenizer_encoder.onnx` (or skip; they're tiny).
5. `validate.py` — add a stage that compares speaker-encoder + tokenizer-encoder
   outputs against PyTorch with `atol=1e-4`, then runs an end-to-end voice
   clone against the reference Python pipeline.
6. `Makefile` — add a `MODEL_PRESET = clone-0.6b` switch (or just document
   `make all MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base OUTPUT_DIR=...`).

The published HF repo will be `wavekat/Qwen3-TTS-0.6B-Base-ONNX` (new), with
the same `int4/`, `fp32/`, `embeddings/`, `tokenizer/` layout plus a top-level
`speaker_encoder.onnx` and `tokenizer_encoder.onnx`. (Both files apply equally
to either precision; no need to duplicate.)

## Rust backend changes

### New module structure

```
src/backends/qwen3_tts/
├── mod.rs              — Qwen3Tts (existing); add Qwen3TtsClone
├── download.rs         — extend MODEL_FILES per variant
├── model.rs            — existing talker/CP/vocoder pipeline (untouched)
├── clone_model.rs      — NEW: voice-clone-specific ONNX + prefill builder
├── speaker_encoder.rs  — NEW: ONNX session for speaker encoder + mel computation
├── tokenizer.rs        — unchanged
└── sampler.rs          — unchanged
```

### Public API

Two options. We pick **(B)** to keep the trait surface clean and avoid leaking
clone-only state into the existing struct.

(A) Single struct, runtime mode flag:
```rust
let tts = Qwen3Tts::from_config(ModelConfig::default()
    .with_variant(ModelVariant::Clone06B));
let req = SynthesizeRequest::new("Hello").with_reference(ref_audio, "ref text");
tts.synthesize(&req)?;
```

(B) Sibling struct, dedicated request type — chosen:
```rust
use wavekat_tts::backends::qwen3_tts::{Qwen3TtsClone, CloneConfig, CloneRequest};

let tts = Qwen3TtsClone::new()?;                          // 0.6B Base, INT4, CPU
let pcm = wavekat_tts::audio::read_wav("ref.wav")?;
let req = CloneRequest::new("New text to say", &pcm)
    .with_ref_text("transcript of ref.wav")
    .with_language("en");
let frame: AudioFrame = tts.synthesize(&req)?;
```

`Qwen3TtsClone` implements an additional trait or just exposes
`fn synthesize(&self, req: &CloneRequest) -> Result<AudioFrame, TtsError>`. The
existing `TtsBackend` contract doesn't fit cleanly because it has no place for
a reference clip; we don't try to shoehorn it in.

`CloneRequest`:
```rust
pub struct CloneRequest<'a> {
    pub text: &'a str,
    pub ref_audio: &'a AudioFrame<'a>,   // any sr; we resample to 24k internally
    pub ref_text: Option<&'a str>,       // None ⇒ x_vector_only_mode
    pub language: Option<&'a str>,
}
```

`CloneConfig` mirrors `ModelConfig` (precision, EP, model dir).

### New code paths

1. `speaker_encoder.rs` — load `speaker_encoder.onnx`; expose
   `fn embed(&self, pcm_24k: &[f32]) -> Result<Array1<f32>, TtsError>` that
   computes the mel and runs the session.
2. `clone_model.rs::encode_ref_codes(pcm_24k) -> Array2<i64>` — runs
   `tokenizer_encoder.onnx`.
3. `clone_model.rs::build_prefill_embeds_clone(...)` — replaces
   `model::Model::build_prefill_embeds`. Takes the speaker embed, ref codes,
   ref-text tokens, and target-text tokens; produces the `(1, T, 1024)` tensor
   following the layout in "Prefill embedding construction" above.
4. The decode loop in `model.rs` is reusable — we factor it into a helper that
   both pipelines call (`run_decode_loop(&self, prefill_embeds, ...)`).
5. `run_vocoder` is reused, but we prepend `ref_codes` before vocoding and trim
   the leading portion from the resulting waveform (proportional cut, like the
   reference Python does).

### Mel-spectrogram in Rust

We need an STFT + mel filterbank with the exact params above. Two options:

- **Pure Rust**: `rustfft` + `realfft`, hand-rolled mel filterbank constants.
  ~150 lines. We control numerics.
- **Dependency**: `librosa-rs` / `aubio-rs`. Heavier; mismatched defaults.

Going with **pure Rust**. We add a tiny `mel.rs` module with constants matching
the reference (`HiFi-GAN`-style mel from `mel_spectrogram(...)` in
`modeling_qwen3_tts.py`). Validation: bit-identity with the Python output for a
fixed test waveform, atol=1e-4.

### Audio I/O

`CloneRequest` takes an `AudioFrame`. To read a WAV from disk, callers use
`hound` or `wavekat-core` helpers. We do **not** add file-loading to the
backend itself; that stays at the example/binding layer.

### Resampling

Reference clips will not be 24 kHz in general. We resample to 24 kHz inside
`Qwen3TtsClone::synthesize` using `rubato` (already a transitive dep via
wavekat-core, IIRC; if not, add it). High-quality sinc resampling is fine for
an offline preprocessing step on a short clip.

## Examples

Add `examples/synthesize_clone.rs`:

```rust
use wavekat_core::{AudioFrame, wav::read_wav};
use wavekat_tts::backends::qwen3_tts::{Qwen3TtsClone, CloneRequest};

fn main() -> anyhow::Result<()> {
    let tts = Qwen3TtsClone::new()?;
    let ref_audio = read_wav("ref.wav")?;
    let req = CloneRequest::new(
        "I am solving the equation x = (-b ± √(b²-4ac)) / 2a.",
        &ref_audio,
    )
    .with_ref_text("Okay. Yeah. I resent you. I love you. I respect you.")
    .with_language("en");

    let out: AudioFrame = tts.synthesize(&req)?;
    wavekat_core::wav::write_wav("clone_out.wav", &out)?;
    Ok(())
}
```

## Verification plan

1. **Per-component parity** (in `tools/qwen3-tts-onnx/validate.py`):
   - speaker_encoder ONNX vs PyTorch on a fixed mel: max abs err < 1e-4
   - tokenizer_encoder ONNX vs PyTorch on a fixed wav: identical i64 codes
2. **End-to-end ONNX (Python) vs PyTorch reference**:
   - Greedy decode (`do_sample=False`) over same text/ref → identical talker
     code sequences for first ~50 frames (sampling drift past that is fine).
   - Audio SNR > 35 dB.
3. **End-to-end Rust (`Qwen3TtsClone`) vs Python ONNX**:
   - Same prefill embed tensor (atol 1e-4) on the first decode step.
   - Manual listening on 5+ samples covering en/zh/ja.
4. **Smoke test**: example runs to completion on the published `clone.wav` ref
   clip; saved WAV plays the cloned voice.

## Out of scope (for this PR)

- 1.7B Base voice clone (mechanically the same; future patch flips a model dir)
- `x_vector_only_mode` (no `ref_text`) — we'll add after ICL works
- Streaming voice clone — the Python reference simulates streaming; we stay
  non-streaming to match the existing VoiceDesign path
- True batch inference

## Open questions

1. **Single struct vs sibling struct?** Drafting both APIs; will pick one in
   review. Currently leaning sibling (`Qwen3TtsClone`).
2. **Ship speaker/tokenizer encoders as INT4?** They are small (≪ 100 MB FP32).
   Probably keep FP32 to avoid quality loss; revisit if size matters.
3. **Re-use the existing 1.7B HF repo or publish a fresh `0.6B-Base-ONNX`?**
   New repo is cleaner — different model, different weights. Confirmed: new.

[hf-base]: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base
[gen_clone]: https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py
[gen_icl]: https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/models/modeling_qwen3_tts.py
