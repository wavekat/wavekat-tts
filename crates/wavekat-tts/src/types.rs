/// A TTS synthesis request.
///
/// Backend-agnostic parameters that describe what to synthesize.
/// Each backend interprets `voice`, `instruction`, and `language` according to
/// its own capabilities; unsupported fields are silently ignored.
#[derive(Debug, Clone)]
pub struct SynthesizeRequest<'a> {
    /// Text to synthesize.
    pub text: &'a str,

    /// Voice identifier (backend-specific).
    ///
    /// Used by backends with a fixed speaker catalog:
    /// - Edge-TTS: `"zh-CN-XiaoxiaoNeural"`, `"zh-CN-YunxiNeural"`, …
    /// - Kokoro: `"af_heart"`, `"zf_xiaobei"`, …
    ///
    /// `None` uses the backend's default voice.
    pub voice: Option<&'a str>,

    /// Free-form voice instruction / style prompt.
    ///
    /// Used by instruction-following backends (e.g. Qwen3-TTS VoiceDesign).
    /// The text describes how the model should speak:
    ///
    /// ```text
    /// "Speak in a calm, professional tone."
    /// "Narrate with warmth and a gentle pace."
    /// "Respond with high energy and enthusiasm!"
    /// ```
    ///
    /// `None` lets the backend use its default voice character.
    pub instruction: Option<&'a str>,

    /// Language / locale code.
    ///
    /// E.g. `"zh"`, `"en"`, `"ja"`.
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
            instruction: None,
            language: None,
            speed: None,
        }
    }

    /// Set the voice identifier.
    pub fn with_voice(mut self, voice: &'a str) -> Self {
        self.voice = Some(voice);
        self
    }

    /// Set the voice instruction / style prompt.
    pub fn with_instruction(mut self, instruction: &'a str) -> Self {
        self.instruction = Some(instruction);
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
