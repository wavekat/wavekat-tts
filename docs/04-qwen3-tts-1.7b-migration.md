# Qwen3-TTS: 1.7B VoiceDesign Migration

Documents the changes made in `feat/qwen3-tts-1.7b-int4` to migrate the
Qwen3-TTS backend from the third-party 0.6B ONNX repo to the WaveKat-owned
1.7B VoiceDesign ONNX repo with INT4 quantization by default.

## What changed and why

| | Before | After |
|---|---|---|
| **Model** | Qwen3-TTS-12Hz-0.6B-Base | Qwen3-TTS-12Hz-1.7B-VoiceDesign |
| **HF repo** | `elbruno/Qwen3-TTS-12Hz-0.6B-Base-ONNX` | `wavekat/Qwen3-TTS-1.7B-VoiceDesign-ONNX` |
| **ONNX variant** | FP32 | INT4 weight-only (RTN, block=128) |
| **Download** | Custom HTTP + manual cache | `hf-hub` crate (standard HF cache) |
| **Prefill mode** | Streaming (text fed per decode step) | Non-streaming (all text in prefill) |
| **Sampling** | top-p | top-k |

The 1.7B VoiceDesign model produces significantly higher quality audio, particularly
for longer utterances and non-English languages. The INT4 models are ~4× smaller than
FP32 (talker: 5.3 GB → 1.4 GB) with negligible quality loss.

## HF repo structure

`wavekat/Qwen3-TTS-1.7B-VoiceDesign-ONNX` mirrors the export output from
`tools/qwen3-tts-onnx/`. The Rust backend downloads and uses only the INT4 variant:

```
{hf_snapshot_root}/
├── config.json               # model dimensions, token IDs, sampling config
├── int4/                     # INT4 weight-only quantized ONNX models
│   ├── talker_prefill.onnx   + .onnx.data   (~1.4 GB)
│   ├── talker_decode.onnx    + .onnx.data   (~1.4 GB)
│   ├── code_predictor.onnx   + .onnx.data   (~322 MB)
│   └── vocoder.onnx          + .onnx.data   (~558 MB)
├── embeddings/               # pre-extracted weights as .npy
│   ├── text_embedding.npy
│   ├── text_projection_fc1_{weight,bias}.npy
│   ├── text_projection_fc2_{weight,bias}.npy
│   ├── talker_codec_embedding.npy
│   └── cp_codec_embedding_{0..14}.npy
└── tokenizer/
    ├── vocab.json
    └── merges.txt
```

`embeddings/small_to_mtp_projection_{weight,bias}.npy` also exist in the repo
but are not downloaded by the Rust backend — the `small_to_mtp` projection is
baked into `code_predictor.onnx`.

## Download: hf-hub

The `ureq` dependency and custom HTTP download logic were replaced with
`hf-hub = "0.5"` (the official Rust HF Hub client).

`download::ensure_model_dir()` calls `repo.get(filename)` for each file in
the hardcoded `MODEL_FILES` list.  HF Hub handles caching, LFS resolution, and
redirect following transparently.  hf-hub 0.5 fixes the relative-`Location`
redirect handling that broke 0.3 with HuggingFace's `/api/resolve-cache/` backend.

**Cache location**: `$HF_HOME/hub/` (default `~/.cache/huggingface/hub/`).
The function returns the snapshot root directory, which has the repo's subdirectory
layout intact.

**Environment variables**:
| Variable | Purpose |
|---|---|
| `WAVEKAT_MODEL_DIR` | Skip HF Hub; load from this local path directly |
| `HF_TOKEN` | Authentication for private/gated repos |
| `HF_HOME` | Override cache root (also sets the token file location) |
| `HF_ENDPOINT` | Override the HuggingFace endpoint URL |

> **Note on `HF_TOKEN`**: hf-hub 0.5 does not natively read `HF_TOKEN`
> from the environment (it reads `$HF_HOME/token`, written by
> `huggingface-cli login`).  `ensure_model_dir()` bridges this by passing
> `HF_TOKEN` to `ApiBuilder::with_token` when the env var is set.

## Model dimensions (1.7B vs 0.6B)

| Parameter | 0.6B | 1.7B |
|---|---|---|
| `HIDDEN_DIM` | 1024 | **2048** |
| `NUM_LAYERS` | 28 | 28 |
| `NUM_KV_HEADS` | 8 | 8 |
| `HEAD_DIM` | 128 | 128 |
| `CP_NUM_LAYERS` | 5 | 5 |
| `CP_NUM_KV_HEADS` | 8 | 8 |
| `MAX_NEW_TOKENS` | 2048 | **8192** |

Only `HIDDEN_DIM` changes (1024 → 2048). All KV cache shapes, the code predictor
architecture, and the vocoder are identical between the two sizes.

## Sampling (top-k replacing top-p)

`SamplerConfig` now has `top_k: usize` instead of `top_p: f32`.
Values from `config.json`:

| | Talker (group 0) | Code Predictor (groups 1-15) |
|---|---|---|
| `temperature` | 0.9 | 0.9 |
| `top_k` | 50 | 50 |
| `repetition_penalty` | 1.05 | 1.0 |

The previous 0.6B values (temp=0.7, top_p=0.8 for talker; temp=0.2, top_p=0.5
for CP) were derived from an older reference and did not match the official
`config.json`.

Additionally, `min_new_tokens=2` is now enforced: CODEC_EOS is masked for the
first two decode steps, matching `generate_onnx.py`.

## Prefill: non-streaming mode

The 0.6B implementation used a **streaming** prefill: only the first text token
was included in the prefill; the rest were fed one-per-decode-step as a "trailing"
vector added to the codec embedding sum.

The 1.7B implementation uses **non-streaming** prefill to match `generate_onnx.py`.
All text tokens are embedded in the prefill sequence:

```
[im_start, assistant, \n]                  — role (text_proj only)
[think, think_bos, lang_id, think_eos]     — codec prefix (tts_pad + codec_embed)
[tts_bos + codec_pad]                      — transition
[text_proj(tok) + codec_pad] × N           — all text tokens
[text_proj(TTS_EOS) + codec_pad]           — TTS_EOS
[tts_pad + codec_bos]                      — final
```

The decode loop trailing is always `tts_pad_embed` (a constant) rather than
per-step text projections.

Non-streaming gives the Talker LM full visibility over the complete text from the
first token, which improves prosody and accuracy on longer inputs.

**Note**: the 0.6B codec prefix had 5 tokens (included a speaker slot = CODEC_PAD).
VoiceDesign has no predefined speakers, so the codec prefix is 4 tokens.

## Bug fix: prefill logits extraction

`run_talker_prefill` previously returned `logits_data.to_vec()` which includes
logits for **all** T prefill positions — a flat vector of T × 3072 elements.
The sampler then treated this as a 3072-token vocab, producing wrong samples.

Fixed by slicing only the last position:

```rust
let logits: Vec<f32> = logits_data[logits_data.len() - TALKER_VOCAB_SIZE..].to_vec();
```

The decode step was unaffected (it always returns shape `(1, 1, 3072)` = 3072 elements).

## code_predictor.onnx and small_to_mtp

For the 1.7B model, the Talker hidden size is 2048 but the Code Predictor
transformer runs at 1024. The `small_to_mtp_projection` (Linear 2048 → 1024)
is **baked into `code_predictor.onnx`** rather than applied in host code.

As a result:
- Host code passes `(1, 2, 2048)` to the code predictor on the first call
  (concat of talker hidden and group-0 codec embedding, both 2048-dim)
- Subsequent calls pass `(1, 1, 2048)` from `cp_codec_embeddings[g]` (shape 2048×2048)
- No host-side projection step is needed

`small_to_mtp_projection_{weight,bias}.npy` are present in the HF repo for
reference but are not downloaded or used by the Rust backend.

## Usage

```bash
# Auto-download via HF Hub and synthesize
cargo run --example synthesize --features qwen3-tts,hound -- "Hello, world!"

# Interactive mode
cargo run --example synthesize --features qwen3-tts,hound -- -i

# Load from manually downloaded snapshot
WAVEKAT_MODEL_DIR=/path/to/snapshot \
  cargo run --example synthesize --features qwen3-tts,hound -- "Hello"

# With a VoiceDesign instruction
cargo run --example synthesize --features qwen3-tts,hound -- \
  --instruction "Speak in a calm, professional tone." "Hello, world!"
```

The first run downloads ~3.7 GB of INT4 ONNX files and ~0.6 GB of embeddings.
Subsequent runs load directly from the HF Hub cache.
