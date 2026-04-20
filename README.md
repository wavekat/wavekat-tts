<p align="center">
  <a href="https://github.com/wavekat/wavekat-tts">
    <img src="https://github.com/wavekat/wavekat-brand/raw/main/assets/banners/wavekat-tts-narrow.svg" alt="WaveKat TTS">
  </a>
</p>

[![Crates.io](https://img.shields.io/crates/v/wavekat-tts.svg)](https://crates.io/crates/wavekat-tts)
[![docs.rs](https://docs.rs/wavekat-tts/badge.svg)](https://docs.rs/wavekat-tts)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-wavekat%2Fwavekat--tts-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/wavekat/wavekat-tts)

Unified text-to-speech for voice pipelines, wrapping multiple TTS engines
behind common Rust traits. 
Same pattern as
[wavekat-vad](https://github.com/wavekat/wavekat-vad) and
[wavekat-turn](https://github.com/wavekat/wavekat-turn).

> [!WARNING]
> Early development. API may change between minor versions.

## Backends

| Backend | Feature flag | Status | License |
|---------|-------------|--------|---------|
| [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) (VoiceDesign 1.7B) | `qwen3-tts` | ✅ Available | Apache 2.0 |
| [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) (Voice Clone 0.6B) | `qwen3-tts` | ✅ Available | Apache 2.0 |
| [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | `cosyvoice` | 🚧 Planned | Apache 2.0 |

## Model weights

ONNX-converted weights are published under the [`wavekat`](https://huggingface.co/wavekat) organization on Hugging Face.

| Backend | Repository | Precision |
|---------|------------|-----------|
| Qwen3-TTS VoiceDesign | [wavekat/Qwen3-TTS-1.7B-VoiceDesign-ONNX](https://huggingface.co/wavekat/Qwen3-TTS-1.7B-VoiceDesign-ONNX) | FP32, INT4 |
| Qwen3-TTS Voice Clone | [wavekat/Qwen3-TTS-0.6B-Base-ONNX](https://huggingface.co/wavekat/Qwen3-TTS-0.6B-Base-ONNX) | FP32, INT4 |

The INT4 models use weight-only RTN quantization via ONNX Runtime's `MatMulNBitsQuantizer`.
See [ONNX Export & Quantization Guide](docs/QUANTIZATION.md) for how to
reproduce or customize the process.

## Quick start

```sh
cargo add wavekat-tts --features qwen3-tts
```

### VoiceDesign (prompt-based styling)

```rust
use wavekat_tts::{TtsBackend, SynthesizeRequest};
use wavekat_tts::backends::qwen3_tts::Qwen3Tts;
// use wavekat_tts::backends::qwen3_tts::{ModelConfig, ModelPrecision, ExecutionProvider};

fn main() {
    let tts = Qwen3Tts::new().unwrap(); // auto-downloads INT4 model on first run

    // For FP32 or GPU inference, use from_config:
    // let config = ModelConfig::default()
    //     .with_precision(ModelPrecision::Fp32)       // FP32 instead of INT4
    //     .with_execution_provider(ExecutionProvider::Cuda); // GPU via CUDA
    // let tts = Qwen3Tts::from_config(config).unwrap();

    let request = SynthesizeRequest::new("Hello, world")
        .with_instruction("Speak naturally and clearly.");
    let audio = tts.synthesize(&request).unwrap();
    audio.write_wav("output.wav").unwrap();
    println!("Wrote output.wav ({:.2}s)", audio.duration_secs());
}
```

### Voice Clone (reference-audio cloning)

> **Requires a reference WAV file** (`ref.wav`) — a short mono clip of the voice
> you want to clone, plus a transcript of what is spoken in the clip.

```rust
use wavekat_tts::AudioFrame;
use wavekat_tts::backends::qwen3_tts::{Qwen3TtsClone, CloneRequest};
// use wavekat_tts::backends::qwen3_tts::{ModelConfig, ModelPrecision};

fn main() {
    let ref_audio = AudioFrame::from_wav("ref.wav").unwrap();
    let ref_audio = ref_audio.resample(24000).unwrap(); // resample to 24 kHz (no-op if already 24 kHz)
    let tts = Qwen3TtsClone::new().unwrap(); // auto-downloads 0.6B Base INT4 model

    // For FP32 precision:
    // let config = ModelConfig::default().with_precision(ModelPrecision::Fp32);
    // let tts = Qwen3TtsClone::from_config(config).unwrap();

    let req = CloneRequest::new(
        "Text to say in the cloned voice",
        ref_audio.samples(),
        24000,
        "Transcript of the reference clip.",
    ).with_language("en");
    let audio = tts.synthesize_clone(&req).unwrap();
    audio.write_wav("clone_output.wav").unwrap();
    println!("Wrote clone_output.wav ({:.2}s)", audio.duration_secs());
}
```

Model files are cached by the HF Hub client at `$HF_HOME/hub/` (default `~/.cache/huggingface/hub/`).
Set `WAVEKAT_MODEL_DIR` to load from a local directory and skip all downloads.

All backends produce `AudioFrame<'static>` from [`wavekat-core`](https://github.com/wavekat/wavekat-core) — the same
type consumed by `wavekat-vad` and `wavekat-turn`.

## Architecture

```
wavekat-vad   →  "is someone speaking?"
wavekat-turn  →  "are they done speaking?"
wavekat-tts   →  "synthesize the response"
     │                   │                     │
     └───────────────────┴─────────────────────┘
                         │
                   AudioFrame (wavekat-core)
```

Two trait families:

- **`TtsBackend`** — batch synthesis: text → `AudioFrame<'static>`
- **`StreamingTtsBackend`** — streaming: text → iterator of `AudioFrame<'static>` chunks

## Examples

Generate a WAV file from text (model files are auto-downloaded on first run):

```sh
# VoiceDesign (1.7B)
cargo run --example synthesize --features qwen3-tts -- "Hello, world\!"
cargo run --example synthesize --features qwen3-tts -- --instruction "Speak in a warm, friendly tone." "Give every small business the voice of a big one."
# cargo run --example synthesize --features qwen3-tts -- --precision fp32 "Hello, world\!"

# Voice Clone (0.6B)
cargo run --example synthesize_clone --features qwen3-tts -- \
  --ref-audio ref.wav --ref-text "Transcript of the reference clip." \
  "Text to synthesize in the cloned voice."
# cargo run --example synthesize_clone --features qwen3-tts -- --precision fp32 \
#   --ref-audio ref.wav --ref-text "Transcript." "Text to synthesize."
```

## Performance

<!-- bench:start -->
| Backend | Precision | Provider | Hardware | RTF short | RTF medium | RTF long |
|---------|-----------|----------|----------|:-----------:|:-----------:|:-----------:|
| qwen3-tts | int4 | CPU | Standard_NC4as_T4_v3 | 1.98 | 2.04 | 2.34 |
| qwen3-tts | int4 | CUDA | Standard_NC4as_T4_v3 | **0.78** | **0.85** | 1.07 |

_RTF < 1.0 = faster-than-real-time. Lower is better._  
_To update: run `make bench-csv-cuda` on target hardware, then commit `bench/results/`._
<!-- bench:end -->

## Feature flags

### Backends

| Flag | Default | Description |
|------|---------|-------------|
| `qwen3-tts` | off | Qwen3-TTS local ONNX inference |
| `cosyvoice` | off | CosyVoice local ONNX inference (planned) |

### Execution providers

Composable with any backend flag. Selects the inference hardware at build time.

| Flag | Description | Status |
|------|-------------|--------|
| `cuda` | NVIDIA CUDA GPU | ✅ Working |
| `tensorrt` | NVIDIA TensorRT | 🚧 Not configured |
| `coreml` | Apple CoreML (macOS) | 🚧 Not configured |

## License

Licensed under [Apache 2.0](LICENSE).

Copyright 2026 WaveKat.
