/// Backend implementations for various TTS engines.
///
/// Each backend is gated behind a feature flag. Enable the corresponding
/// feature in your `Cargo.toml` to use it:
///
/// ```toml
/// wavekat-tts = { version = "0.0.1", features = ["qwen3-tts"] }
/// ```

#[cfg(feature = "qwen3-tts")]
pub mod qwen3_tts;

#[cfg(feature = "cosyvoice")]
pub mod cosyvoice;
