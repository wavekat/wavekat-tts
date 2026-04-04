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

> **Warning** Early development. API may change between minor versions.

## Backends

| Backend | Feature flag | Chinese quality | Requires | License |
|---------|-------------|-----------------|----------|---------|
| [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh) | `kokoro` | Good (4 voices) | ONNX model download | Apache 2.0 |
| [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) | `qwen3-tts` | Excellent | ONNX model download | Apache 2.0 |
| [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) | `cosyvoice` | Excellent | ONNX model download | Apache 2.0 |

## Quick start

```toml
[dependencies]
wavekat-tts = { version = "0.0.1", features = ["kokoro"] }
```

```rust
use wavekat_tts::{TtsBackend, SynthesizeRequest};
use wavekat_tts::backends::kokoro::KokoroTts;

let tts = KokoroTts::new("path/to/model.onnx")?;
let request = SynthesizeRequest::new("你好，世界")
    .with_voice("zf_xiaobei");
let audio = tts.synthesize(&request)?;

println!("{}s at {} Hz", audio.duration_secs(), audio.sample_rate());
```

All backends produce `AudioFrame<'static>` from `wavekat-core` — the same
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

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `kokoro` | off | Kokoro 82M local ONNX inference |
| `qwen3-tts` | off | Qwen3-TTS local ONNX inference |
| `cosyvoice` | off | CosyVoice local ONNX inference |

## License

Licensed under [Apache 2.0](LICENSE).

Copyright 2026 WaveKat.
