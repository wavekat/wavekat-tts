/// Errors produced by TTS backends.
#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    /// The requested voice was not found.
    #[error("voice not found: {0}")]
    VoiceNotFound(String),

    /// The requested language is not supported by this backend.
    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),

    /// Model loading or initialization failed.
    #[error("model error: {0}")]
    Model(String),

    /// Inference / synthesis failed.
    #[error("synthesis error: {0}")]
    Synthesis(String),

    /// Network or API error (for remote backends).
    #[error("api error: {0}")]
    Api(String),

    /// I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<wavekat_core::CoreError> for TtsError {
    fn from(err: wavekat_core::CoreError) -> Self {
        match err {
            wavekat_core::CoreError::Io(io) => Self::Io(io),
            wavekat_core::CoreError::Audio(msg) => Self::Synthesis(msg),
        }
    }
}
