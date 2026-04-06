/// A TTS synthesis request.
///
/// Backend-agnostic parameters that describe what to synthesize.
/// Each backend interprets `voice` and `language` according to its own catalog.
#[derive(Debug, Clone)]
pub struct SynthesizeRequest<'a> {
    /// Text to synthesize.
    pub text: &'a str,

    /// Voice identifier (backend-specific).
    ///
    /// For Edge-TTS: `"zh-CN-XiaoxiaoNeural"`, `"zh-CN-YunxiNeural"`, etc.
    /// For Kokoro: `"af_heart"`, `"zf_xiaobei"`, etc.
    /// `None` uses the backend's default voice.
    pub voice: Option<&'a str>,

    /// Language / locale code.
    ///
    /// E.g. `"zh-CN"`, `"en-US"`, `"ja-JP"`.
    /// `None` uses the backend's default or auto-detects.
    pub language: Option<&'a str>,

    /// Speed multiplier. `1.0` is normal speed.
    ///
    /// Values below 1.0 slow down, above 1.0 speed up.
    /// Not all backends support this; unsupported values are ignored.
    pub speed: Option<f32>,
}

impl<'a> SynthesizeRequest<'a> {
    /// Create a minimal request with just text.
    pub fn new(text: &'a str) -> Self {
        Self {
            text,
            voice: None,
            language: None,
            speed: None,
        }
    }

    /// Set the voice.
    pub fn with_voice(mut self, voice: &'a str) -> Self {
        self.voice = Some(voice);
        self
    }

    /// Set the language.
    pub fn with_language(mut self, language: &'a str) -> Self {
        self.language = Some(language);
        self
    }

    /// Set the speed multiplier.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = Some(speed);
        self
    }
}

/// Metadata about a voice available in a backend.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VoiceInfo {
    /// Backend-specific voice identifier.
    pub id: String,

    /// Human-readable display name.
    pub name: String,

    /// Supported language / locale codes.
    pub languages: Vec<String>,

    /// Gender hint, if available.
    pub gender: Option<Gender>,
}

/// Voice gender hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Gender {
    Male,
    Female,
    Neutral,
}
