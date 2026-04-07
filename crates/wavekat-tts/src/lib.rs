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
//! ## Backends
//!
//! | Feature | Backend | Multilingual | Requires |
//! |---------|---------|-------------|----------|
//! | `qwen3-tts` | Qwen3-TTS (ONNX) | 10 languages | ONNX model download |
//! | `cosyvoice` | CosyVoice (ONNX) | Yes | ONNX model download |
//!
//! ## Execution providers
//!
//! Composable with any backend feature. Selects the inference hardware at build time.
//!
//! | Feature | Provider | Platform |
//! |---------|----------|----------|
//! | `cuda` | NVIDIA CUDA | Linux / Windows |
//! | `tensorrt` | NVIDIA TensorRT | Linux / Windows |
//! | `coreml` | Apple CoreML | macOS / iOS |
//!
//! # Quick start
//!
//! ```toml
//! [dependencies]
//! wavekat-tts = { version = "0.0.1", features = ["qwen3-tts"] }
//! ```
//!
//! ```ignore
//! use wavekat_tts::{TtsBackend, SynthesizeRequest};
//! use wavekat_tts::backends::qwen3_tts::Qwen3Tts;
//!
//! let tts = Qwen3Tts::new("path/to/model.onnx")?;
//! let request = SynthesizeRequest::new("我觉得这个方案");
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
