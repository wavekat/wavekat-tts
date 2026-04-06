<p align="center">
  <a href="https://github.com/wavekat/wavekat-tts">
    <img src="https://github.com/wavekat/wavekat-brand/raw/main/assets/banners/wavekat-tts-narrow.svg" alt="WaveKat TTS">
  </a>
</p>

[![Crates.io](https://img.shields.io/crates/v/wavekat-tts.svg)](https://crates.io/crates/wavekat-tts)
[![docs.rs](https://docs.rs/wavekat-tts/badge.svg)](https://docs.rs/wavekat-tts)

Unified text-to-speech for voice pipelines, wrapping multiple TTS engines
behind common Rust traits. 
Same pattern as
[wavekat-vad](https://github.com/wavekat/wavekat-vad) and
[wavekat-turn](https://github.com/wavekat/wavekat-turn).

> [!WARNING]
> Early development. API may change between minor versions.

## Backends

| Backend | Feature flag | License |
|---------|-------------|---------|
| [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) | `qwen3-tts` | Apache 2.0 |
| [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | `cosyvoice` | Apache 2.0 |

## Quick start

```sh
cargo add wavekat-tts --features qwen3-tts
```

```rust
use wavekat_tts::{TtsBackend, SynthesizeRequest};
use wavekat_tts::backends::qwen3_tts::Qwen3Tts;

// Auto-downloads model files (~3.8 GB) on first run:
let tts = Qwen3Tts::new()?;

// Or load from an explicit directory:
// let tts = Qwen3Tts::from_dir("models/qwen3-tts-0.6b")?;

let request = SynthesizeRequest::new("Hello, world");
let audio = tts.synthesize(&request)?;

println!("{}s at {} Hz", audio.duration_secs(), audio.sample_rate());
```

Model files are cached at `$WAVEKAT_MODEL_DIR` or `~/.cache/wavekat/qwen3-tts-0.6b/`.

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
cargo run --example synthesize --features qwen3-tts,hound -- "Hello, world\!"
cargo run --example synthesize --features qwen3-tts,hound -- --instruction "Speak in a warm, friendly tone." "Give every small business the voice of a big one."
cargo run --example synthesize --features qwen3-tts,hound -- --model-dir /path/to/model --output hello.wav "Hello"
```

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `qwen3-tts` | off | Qwen3-TTS local ONNX inference |
| `cosyvoice` | off | CosyVoice local ONNX inference |

## License

Licensed under [Apache 2.0](LICENSE).

Copyright 2026 WaveKat.
