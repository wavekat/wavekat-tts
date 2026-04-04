# CLAUDE.md

## Project overview

wavekat-tts is a unified TTS library for the WaveKat voice pipeline ecosystem.
It wraps multiple TTS backends behind common Rust traits, producing `AudioFrame`
from `wavekat-core`.

## Build & test

```bash
make check          # clippy + fmt + test (no features)
make test-kokoro    # test kokoro backend
make test-all       # test all backends
```

## Architecture

- `src/traits.rs` — `TtsBackend` and `StreamingTtsBackend` traits
- `src/types.rs` — `SynthesizeRequest`, `VoiceInfo`, `Gender`
- `src/error.rs` — `TtsError`
- `src/backends/` — one module per backend, gated by feature flags
- All backends return `AudioFrame<'static>` from `wavekat-core`

## Key design decisions

1. **AudioFrame is the only audio type** — no custom AudioOutput/AudioChunk types.
   TTS output is `AudioFrame<'static>`, same type consumed by wavekat-vad/turn.
2. **Sample rate is not hardcoded** — each backend outputs at its native rate.
   Callers check `frame.sample_rate()` and resample if needed.
3. **Sync traits** — matches wavekat-vad and wavekat-turn. Async backends
   use internal tokio runtimes.
4. **Feature flags per backend** — same pattern as wavekat-vad/turn.

## Pending wavekat-core change

`AudioFrame::from_owned(Vec<f32>, u32) -> AudioFrame<'static>` — avoids the
borrow-then-clone path when creating frames from producer-owned data (TTS output).
Currently uses `AudioFrame::new(slice, rate).into_owned()` as a workaround.
