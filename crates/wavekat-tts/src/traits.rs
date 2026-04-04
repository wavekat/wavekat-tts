use wavekat_core::AudioFrame;

use crate::error::TtsError;
use crate::types::{SynthesizeRequest, VoiceInfo};

/// Batch TTS backend: text in, `AudioFrame<'static>` out.
///
/// Every backend must implement this trait. The output `AudioFrame` carries
/// its native sample rate (e.g. 24000 for Kokoro, 24000 for Edge-TTS).
/// Callers should check `frame.sample_rate()` and resample if needed.
///
/// # Example
///
/// ```ignore
/// use wavekat_tts::{TtsBackend, SynthesizeRequest};
///
/// let tts = SomeBackend::new()?;
/// let request = SynthesizeRequest::new("你好，世界");
/// let audio = tts.synthesize(&request)?;
///
/// println!("Generated {} samples at {} Hz ({:.2}s)",
///     audio.len(), audio.sample_rate(), audio.duration_secs());
/// ```
pub trait TtsBackend: Send + Sync {
    /// Synthesize text into audio.
    ///
    /// Returns an owned `AudioFrame` at the backend's native sample rate.
    fn synthesize(&self, request: &SynthesizeRequest) -> Result<AudioFrame<'static>, TtsError>;

    /// List available voices for this backend.
    fn voices(&self) -> Result<Vec<VoiceInfo>, TtsError>;
}

/// Streaming TTS backend: text in, `AudioFrame<'static>` chunks out.
///
/// Extends [`TtsBackend`] with streaming support. Each chunk is a
/// self-contained `AudioFrame` that can be played or forwarded immediately.
pub trait StreamingTtsBackend: TtsBackend {
    /// Start streaming synthesis.
    ///
    /// Returns an iterator of audio chunks. Each chunk is an owned
    /// `AudioFrame` at the backend's native sample rate.
    /// The last chunk can be detected by checking the iterator exhaustion.
    fn stream(
        &self,
        request: &SynthesizeRequest,
    ) -> Result<Box<dyn Iterator<Item = Result<AudioFrame<'static>, TtsError>> + Send>, TtsError>;
}
