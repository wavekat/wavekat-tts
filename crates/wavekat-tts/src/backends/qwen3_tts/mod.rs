//! Qwen3-TTS backend (ONNX INT4, 1.7B VoiceDesign).
//!
//! Runs the Qwen3-TTS-12Hz-1.7B-VoiceDesign model via ONNX Runtime using the
//! INT4 weight-only quantized models from `wavekat/Qwen3-TTS-1.7B-VoiceDesign-ONNX`.
//!
//! ```ignore
//! use wavekat_tts::{TtsBackend, SynthesizeRequest};
//! use wavekat_tts::backends::qwen3_tts::Qwen3Tts;
//!
//! // Auto-download model files via HF Hub (cached at ~/.cache/huggingface/hub/):
//! let tts = Qwen3Tts::new()?;
//!
//! // Or load from an explicit directory (must mirror the HF repo layout):
//! let tts = Qwen3Tts::from_dir("models/qwen3-tts-1.7b")?;
//!
//! let request = SynthesizeRequest::new("Hello, world");
//! let audio = tts.synthesize(&request)?;
//! ```

use std::path::Path;

use wavekat_core::AudioFrame;

use crate::error::TtsError;
use crate::traits::TtsBackend;
use crate::types::{SynthesizeRequest, VoiceInfo};

mod download;
mod model;
mod sampler;
mod tokenizer;

/// Qwen3-TTS backend using ONNX Runtime.
pub struct Qwen3Tts {
    model: model::Model,
    tokenizer: tokenizer::Tokenizer,
}

impl Qwen3Tts {
    /// Create a new backend, downloading model files from HF Hub if needed.
    ///
    /// Files are cached by the HF Hub client (default `~/.cache/huggingface/hub/`).
    /// Set `HF_HOME` to change the cache root, or `HF_TOKEN` for authentication.
    /// Set `WAVEKAT_MODEL_DIR` to load from a local directory and skip all downloads.
    ///
    /// Use [`from_dir`](Self::from_dir) to load from an explicit path.
    pub fn new() -> Result<Self, TtsError> {
        let model_dir = download::ensure_model_dir()?;
        Self::from_dir(model_dir)
    }

    /// Load the model from a directory that mirrors the HF repo layout.
    ///
    /// Expected subdirectories:
    /// - `int4/` — ONNX models (`talker_prefill.onnx`, `talker_decode.onnx`, etc.)
    /// - `embeddings/` — `.npy` embedding tables
    /// - `tokenizer/` — `vocab.json`, `merges.txt`
    pub fn from_dir(model_dir: impl AsRef<Path>) -> Result<Self, TtsError> {
        let model_dir = model_dir.as_ref();
        let model = model::Model::load(model_dir)?;
        let tokenizer = tokenizer::Tokenizer::new(model_dir)?;
        Ok(Self { model, tokenizer })
    }
}

impl TtsBackend for Qwen3Tts {
    fn synthesize(&self, request: &SynthesizeRequest) -> Result<AudioFrame<'static>, TtsError> {
        let tokens = self.tokenizer.encode(request.text)?;
        let language = request.language.unwrap_or("en");
        self.model.synthesize(&tokens, language)
    }

    fn voices(&self) -> Result<Vec<VoiceInfo>, TtsError> {
        Ok(vec![VoiceInfo {
            id: "default".into(),
            name: "Qwen3-TTS Default".into(),
            languages: vec![
                "en".into(),
                "zh".into(),
                "ja".into(),
                "ko".into(),
                "de".into(),
                "es".into(),
                "fr".into(),
                "ru".into(),
                "it".into(),
                "pt".into(),
            ],
            gender: None,
        }])
    }
}
