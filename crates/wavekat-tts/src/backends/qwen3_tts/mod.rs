//! Qwen3-TTS backend (ONNX, 12Hz-0.6B).
//!
//! Runs the Qwen3-TTS-12Hz-0.6B model via ONNX Runtime.
//!
//! ```ignore
//! use wavekat_tts::{TtsBackend, SynthesizeRequest};
//! use wavekat_tts::backends::qwen3_tts::Qwen3Tts;
//!
//! // Auto-download model files (~3.8 GB, cached for reuse):
//! let tts = Qwen3Tts::new()?;
//!
//! // Or load from an explicit directory:
//! let tts = Qwen3Tts::from_dir("models/qwen3-tts-0.6b")?;
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
    /// Create a new backend, auto-downloading model files if needed.
    ///
    /// Model files (~3.8 GB) are cached at:
    /// - `$WAVEKAT_MODEL_DIR` if set, otherwise
    /// - `$XDG_CACHE_HOME/wavekat/qwen3-tts-0.6b/`, otherwise
    /// - `$HOME/.cache/wavekat/qwen3-tts-0.6b/`
    ///
    /// Use [`from_dir`](Self::from_dir) to skip auto-download and load from
    /// a specific directory.
    pub fn new() -> Result<Self, TtsError> {
        let model_dir = download::ensure_model_dir()?;
        Self::from_dir(model_dir)
    }

    /// Load the model from a directory containing ONNX files and embeddings.
    ///
    /// Expected files:
    /// - `talker_prefill.onnx`, `talker_decode.onnx`, `code_predictor.onnx`, `vocoder.onnx`
    /// - `text_embedding.npy`, `text_projection_fc1_weight.npy`, etc.
    /// - `talker_codec_embedding.npy`, `cp_codec_embedding_{0..14}.npy`
    /// - `vocab.json`, `merges.txt`
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
