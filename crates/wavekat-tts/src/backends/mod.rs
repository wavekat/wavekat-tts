/// Backend implementations for various TTS engines.
///
/// Each backend is gated behind a feature flag. Enable the corresponding
/// feature in your `Cargo.toml` to use it:
///
/// ```toml
/// wavekat-tts = { version = "0.0.1", features = ["kokoro"] }
/// ```

#[cfg(feature = "kokoro")]
pub mod kokoro;

#[cfg(feature = "elevenlabs")]
pub mod elevenlabs;

#[cfg(feature = "openai")]
pub mod openai;
