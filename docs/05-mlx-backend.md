# MLX Backend for Qwen3-TTS (Apple Silicon)

## Goal

Run the full Qwen3-TTS pipeline natively on Apple Silicon GPU/ANE using Apple's
[MLX](https://github.com/ml-explore/mlx) framework. This replaces ONNX Runtime
for the LM decode loop — the primary synthesis bottleneck — where ORT's CoreML EP
cannot help (crashes on large transformer graphs).

## Why not ONNX + CoreML

ONNX Runtime's CoreML execution provider is designed for CNNs and simple models.
On ORT 2.x it crashes (SIGSEGV) when compiling 1.7B-parameter transformer graphs.
Even if compilation succeeded, CoreML requires static shapes, which conflicts with
the growing KV cache in the autoregressive decode loop.

MLX is purpose-built for Apple Silicon LLM inference:
- Native Metal GPU + ANE dispatch
- Lazy evaluation with graph fusion
- Dynamic shapes with zero overhead
- Transformer attention and KV cache primitives built-in

## Architecture

```
Current (ONNX):
  Rust tokenizer → ORT(talker_prefill) → ORT(talker_decode)×N →
    ORT(code_predictor)×N×15 → ORT(vocoder) → AudioFrame

MLX backend:
  Rust tokenizer → MLX bridge(talker_prefill) → MLX bridge(talker_decode)×N →
    MLX bridge(code_predictor)×N×15 → MLX bridge(vocoder) → AudioFrame
```

Everything moves to MLX. The tokenizer stays in Rust (BPE, no change).
The `TtsBackend` trait surface is identical — callers see no difference.

## Weight format

MLX uses `safetensors` + `config.json` (same convention as HuggingFace
Transformers). The ONNX `.onnx` / `.onnx.data` files are **not used**.

Two options for weight distribution:

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A: Same HF repo, new subdir** | Add `mlx/` alongside `int4/` and `fp32/` | One repo, one download path | Larger repo |
| **B: Separate HF repo** | `wavekat/Qwen3-TTS-1.7B-VoiceDesign-MLX` | Clean separation, independent versioning | Two repos to maintain |

Recommendation: **Option B** — MLX weights have a different update cadence and
the repo size difference is significant (MLX FP16 ≈ 3.4 GB vs ONNX INT4 ≈ 4.5 GB).

### Weight conversion

Apple's `mlx-lm` Python package converts HuggingFace checkpoints to MLX format:

```bash
pip install mlx-lm
mlx_lm.convert --hf-path Qwen/Qwen3-TTS-1.7B-VoiceDesign \
                --mlx-path mlx-weights/ \
                --quantize --q-bits 4
```

The vocoder weights need separate handling — it is a custom convolutional
architecture, not a standard LLM. A bespoke export script in
`tools/qwen3-tts-mlx/` will be needed (similar to the existing ONNX export
tooling in `tools/qwen3-tts-onnx/`).

## Rust ↔ MLX bridge

MLX is a C++/Swift library. Rust cannot call it directly. Three viable
integration strategies:

### Strategy 1: `mlx-rs` crate (pure Rust bindings)

The [`mlx-rs`](https://github.com/oxideai/mlx-rs) community crate provides
Rust bindings to MLX's C++ API via `mlx-sys`.

```toml
mlx = { version = "...", optional = true }
```

```rust
// Conceptual — actual API depends on mlx-rs maturity
use mlx::Array;
let logits: Array = model.talker_decode(&embeds, &past_kv)?;
```

**Pros**: stays fully in Rust, no subprocess or FFI shim
**Cons**: `mlx-rs` is community-maintained, may lag behind MLX releases, API
stability is not guaranteed

### Strategy 2: Swift dylib with C FFI

Build a `libwavekat_mlx.dylib` in Swift that wraps MLX and exposes a C API.
Rust calls in via `extern "C"` declarations and `cc`/`build.rs` linking.

```
crates/wavekat-tts/
  build.rs              — compile Swift, link dylib
  swift/WavekatMLX/
    Sources/
      model.swift       — MLX inference logic
      bridge.swift      — @_cdecl C exports
```

```swift
// bridge.swift
@_cdecl("wavekat_mlx_synthesize")
public func synthesize(/* C-compatible params */) -> UnsafeMutablePointer<Float> { ... }
```

```rust
// model.rs
extern "C" {
    fn wavekat_mlx_synthesize(...) -> *mut f32;
}
```

**Pros**: uses Apple's first-party MLX Swift API (most stable), stays on-device
**Cons**: build system complexity, Swift toolchain required, FFI marshalling

### Strategy 3: Swift sidecar process

Build a standalone Swift CLI `wavekat-tts-mlx` that loads the model once and
serves synthesis requests over a local Unix socket. The Rust backend spawns and
manages the process.

```
Rust TtsBackend::synthesize()
  └─ write SynthesizeRequest to socket (msgpack)
  └─ read AudioFrame samples from socket
```

**Pros**: cleanest separation, Swift process can be prewarmed
**Cons**: IPC overhead (negligible for TTS), distribution requires bundling the CLI

### Recommendation

Start with **Strategy 1** (`mlx-rs`) for the fastest path to working code.
If `mlx-rs` proves too unstable, fall back to **Strategy 2** (Swift dylib).
Strategy 3 is reserved for cases where in-process linking is not viable.

## Cargo feature

```toml
# Cargo.toml
mlx-qwen3-tts = ["dep:mlx", "dep:hf-hub", "dep:tokenizers", "dep:npyz", "dep:rand"]
```

Usage:
```toml
wavekat-tts = { version = "...", features = ["mlx-qwen3-tts"] }
```

`mlx-qwen3-tts` and `qwen3-tts` are independent features — callers can enable
either or both. The `TtsBackend` trait is the common interface.

## Source layout

```
src/backends/
  qwen3_tts/         — existing ONNX backend (unchanged)
  mlx_qwen3/
    mod.rs           — MlxQwen3Tts struct, TtsBackend impl, ModelConfig
    model.rs         — MLX inference: prefill, decode loop, code predictor, vocoder
    download.rs      — HF Hub download for MLX weights
    tokenizer.rs     — re-use or thin wrapper around qwen3_tts::tokenizer
    sampler.rs       — re-use or thin wrapper around qwen3_tts::sampler
```

`tokenizer.rs` and `sampler.rs` are identical to the ONNX backend. Consider
lifting them to `src/backends/qwen3_shared/` to avoid duplication (out of scope
for this branch).

## Model file layout (HF repo)

```
wavekat/Qwen3-TTS-1.7B-VoiceDesign-MLX
  config.json
  model.safetensors          (or sharded: model-00001-of-00003.safetensors, ...)
  model.safetensors.index.json
  tokenizer/
    vocab.json
    merges.txt
  vocoder/
    config.json
    model.safetensors
```

## MLX inference design

The talker LM is a standard Qwen3 transformer — MLX's `nn.TransformerBlock`
with GQA and M-RoPE. The only custom piece is the dual-stream embedding
(text_proj + codec_embed) built in Rust before the first MLX call.

```
prefill:
  embeddings (Rust) → mlx.eval(talker_prefill_graph) → logits, kv_cache

decode loop (×N):
  codec_embed (Rust) → mlx.eval(talker_decode_graph) → logits, kv_cache

code predictor (×N×15):
  mlx.eval(cp_graph) → group tokens

vocoder (×1):
  mlx.eval(vocoder_graph) → waveform Float32
```

KV cache is an `mlx::Array` allocated once and grown in-place — no Rust-side
tensor management needed for the cache itself.

## Implementation order

1. Validate `mlx-rs` API coverage: attention, KV cache, matmul, conv1d.
2. Set up `tools/qwen3-tts-mlx/` export script for vocoder weights.
3. Add `mlx-qwen3-tts` feature to `Cargo.toml`.
4. Implement `src/backends/mlx_qwen3/download.rs` (HF Hub, new repo).
5. Implement `src/backends/mlx_qwen3/model.rs` — talker + code predictor.
6. Implement vocoder in MLX.
7. Wire `TtsBackend` impl in `mod.rs`.
8. Add `test-mlx` Makefile target.
9. Benchmark vs ONNX CPU: measure RTF on M4 Pro.

## Open questions

- **mlx-rs maturity**: does it expose KV cache primitives and M-RoPE? If not,
  we need to implement them as raw MLX ops.
- **Vocoder export**: the vocoder is not a standard architecture — need to verify
  safetensors export produces loadable weights without graph-level ops baked in.
- **Quantisation**: MLX 4-bit quantisation (Q4) uses a different scheme than
  ORT INT4. Validate audio quality parity before publishing MLX weights.
- **macOS version floor**: MLX requires macOS 13.5+. Document this requirement.
