# wavekat-tts

[![Crates.io](https://img.shields.io/crates/v/wavekat-tts.svg)](https://crates.io/crates/wavekat-tts)
[![docs.rs](https://docs.rs/wavekat-tts/badge.svg)](https://docs.rs/wavekat-tts)

Unified text-to-speech for voice pipelines, wrapping multiple TTS engines
behind common Rust traits. Same pattern as
[wavekat-vad](https://github.com/wavekat/wavekat-vad) and
[wavekat-turn](https://github.com/wavekat/wavekat-turn).

> **Warning** Early development. API may change between minor versions.

## Backends

| Backend | Feature flag | Chinese quality | Requires | License |
|---------|-------------|-----------------|----------|---------|
| [Edge-TTS](https://github.com/rany2/edge-tts) | `edge-tts` | Excellent (9+ voices) | `pip install edge-tts` + `ffmpeg` | Free |
| [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh) | `kokoro` | Good (4 voices) | ONNX model download | Apache 2.0 |
| [ElevenLabs](https://elevenlabs.io) | `elevenlabs` | OK | API key | Commercial |
| [OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech) | `openai` | Limited | API key | Commercial |

## Quick start

```bash
# Prerequisites (macOS)
pip install edge-tts
brew install ffmpeg
```

```toml
[dependencies]
wavekat-tts = { version = "0.0.1", features = ["edge-tts"] }
```

```rust
use wavekat_tts::{TtsBackend, SynthesizeRequest};
use wavekat_tts::backends::edge_tts::EdgeTtsCli;

let tts = EdgeTtsCli::new();
let request = SynthesizeRequest::new("你好，世界")
    .with_voice("zh-CN-XiaoxiaoNeural");
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

## Turn detection dataset generation

Generate Chinese audio samples for training turn detectors:

```bash
cargo run --example batch_generate --features edge-tts
```

Produces labeled WAV files + `metadata.json`:
- `finished_*.wav` — complete utterances (speaker is done)
- `unfinished_*.wav` — incomplete utterances (speaker will continue)

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `edge-tts` | off | Microsoft Edge TTS via CLI bridge |
| `kokoro` | off | Kokoro 82M local ONNX inference |
| `elevenlabs` | off | ElevenLabs commercial API |
| `openai` | off | OpenAI TTS API |

## License

Licensed under [Apache 2.0](LICENSE).

Copyright 2026 WaveKat.
