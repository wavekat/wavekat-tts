//! Unified text-to-speech for voice pipelines.
//!
//! Provides a clean abstraction over TTS engines — both local models and
//! cloud APIs — behind common Rust traits. Same pattern as
//! [`wavekat-vad`](https://github.com/wavekat/wavekat-vad) and
//! [`wavekat-turn`](https://github.com/wavekat/wavekat-turn).
//!
//! All backends produce [`AudioFrame<'static>`](wavekat_core::AudioFrame)
//! from `wavekat-core`, keeping audio abstract across the WaveKat ecosystem.
//!
//! # Architecture
//!
//! ```text
//! wavekat-vad   →  "is someone speaking?"
//! wavekat-turn  →  "are they done speaking?"
//! wavekat-tts   →  "synthesize the response"
//!      │                   │                     │
//!      └───────────────────┴─────────────────────┘
//!                          │
//!                AudioFrame (wavekat-core)
//! ```
//!
//! # Feature flags
//!
//! | Feature | Backend | Chinese | Requires |
//! |---------|---------|---------|----------|
//! | `edge-tts` | Microsoft Edge TTS | Excellent | `pip install edge-tts` + `ffmpeg` |
//! | `kokoro` | Kokoro 82M (ONNX) | Good | ONNX model download |
//! | `elevenlabs` | ElevenLabs API | OK | API key |
//! | `openai` | OpenAI TTS | Limited | API key |
//!
//! # Quick start
//!
//! ```toml
//! [dependencies]
//! wavekat-tts = { version = "0.0.1", features = ["edge-tts"] }
//! ```
//!
//! ```ignore
//! use wavekat_tts::{TtsBackend, SynthesizeRequest};
//! use wavekat_tts::backends::edge_tts::EdgeTtsCli;
//!
//! let tts = EdgeTtsCli::new();
//! let request = SynthesizeRequest::new("我觉得这个方案")
//!     .with_voice("zh-CN-YunxiNeural");
//! let audio = tts.synthesize(&request)?;
//! // audio: AudioFrame<'static> at 24kHz
//! ```

mod error;
mod traits;
mod types;

pub mod backends;

pub use error::TtsError;
pub use traits::{StreamingTtsBackend, TtsBackend};
pub use types::{Gender, SynthesizeRequest, VoiceInfo};

// Re-export AudioFrame so users don't need to depend on wavekat-core directly.
pub use wavekat_core::AudioFrame;
