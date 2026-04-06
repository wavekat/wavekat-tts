//! Qwen3-TTS backend (ONNX, 1.7B VoiceDesign).
//!
//! Runs the Qwen3-TTS-12Hz-1.7B-VoiceDesign model via ONNX Runtime.
//! Supports INT4 (weight-only quantized, default) and FP32 precision.
//!
//! ```ignore
//! use wavekat_tts::{TtsBackend, SynthesizeRequest};
//! use wavekat_tts::backends::qwen3_tts::{Qwen3Tts, ModelPrecision};
//!
//! // Auto-download INT4 model files via HF Hub (default):
//! let tts = Qwen3Tts::new()?;
//!
//! // Auto-download FP32 model files:
//! let tts = Qwen3Tts::new_with_precision(ModelPrecision::Fp32)?;
//!
//! // Or load from an explicit directory (must mirror the HF repo layout):
//! let tts = Qwen3Tts::from_dir("models/qwen3-tts-1.7b", ModelPrecision::Int4)?;
//!
//! let request = SynthesizeRequest::new("Hello, world");
//! let audio = tts.synthesize(&request)?;
//! ```

use std::path::Path;

use wavekat_core::AudioFrame;

use crate::error::TtsError;
use crate::traits::TtsBackend;
use crate::types::{SynthesizeRequest, VoiceInfo};

use tokenizer::{IM_END, IM_START, NEWLINE};

mod download;
mod model;
mod sampler;
mod tokenizer;

/// ONNX model precision variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelPrecision {
    /// Weight-only INT4 quantized — smaller download, faster load. Default.
    #[default]
    Int4,
    /// Full FP32 — larger download, no quantization error.
    Fp32,
}

impl ModelPrecision {
    pub(crate) fn subdir(self) -> &'static str {
        match self {
            Self::Int4 => "int4",
            Self::Fp32 => "fp32",
        }
    }
}

/// Qwen3-TTS backend using ONNX Runtime.
pub struct Qwen3Tts {
    model: model::Model,
    tokenizer: tokenizer::Tokenizer,
}

impl Qwen3Tts {
    /// Create a new INT4 backend, downloading model files from HF Hub if needed.
    ///
    /// Files are cached by the HF Hub client (default `~/.cache/huggingface/hub/`).
    /// Set `HF_HOME` to change the cache root, or `HF_TOKEN` for authentication.
    /// Set `WAVEKAT_MODEL_DIR` to load from a local directory and skip all downloads.
    ///
    /// Use [`new_with_precision`](Self::new_with_precision) to select FP32.
    pub fn new() -> Result<Self, TtsError> {
        Self::new_with_precision(ModelPrecision::Int4)
    }

    /// Create a new backend with the given precision, downloading files if needed.
    pub fn new_with_precision(precision: ModelPrecision) -> Result<Self, TtsError> {
        let model_dir = download::ensure_model_dir(precision)?;
        Self::from_dir(model_dir, precision)
    }

    /// Load the model from a directory that mirrors the HF repo layout.
    ///
    /// Expected subdirectories:
    /// - `int4/` or `fp32/` — ONNX models (`talker_prefill.onnx`, etc.)
    /// - `embeddings/` — `.npy` embedding tables
    /// - `tokenizer/` — `vocab.json`, `merges.txt`
    pub fn from_dir(
        model_dir: impl AsRef<Path>,
        precision: ModelPrecision,
    ) -> Result<Self, TtsError> {
        let model_dir = model_dir.as_ref();
        let model = model::Model::load(model_dir, precision)?;
        let tokenizer = tokenizer::Tokenizer::new(model_dir)?;
        Ok(Self { model, tokenizer })
    }
}

impl TtsBackend for Qwen3Tts {
    fn synthesize(&self, request: &SynthesizeRequest) -> Result<AudioFrame<'static>, TtsError> {
        let tokens = self.tokenizer.encode(request.text)?;
        let language = request.language.unwrap_or("en");

        if request.instruction.is_none() {
            eprintln!(
                "wavekat-tts warning: Qwen3-TTS is a VoiceDesign model — \
                 synthesize quality may be inconsistent without a style instruction. \
                 Set `SynthesizeRequest::with_instruction` to control voice style."
            );
        }

        let instruction_tokens = if let Some(instr) = request.instruction {
            let mut toks = vec![IM_START];
            toks.extend(self.tokenizer.encode("user")?);
            toks.push(NEWLINE);
            toks.extend(self.tokenizer.encode("<instruct>")?);
            toks.extend(self.tokenizer.encode(instr)?);
            toks.extend(self.tokenizer.encode("</instruct>")?);
            toks.push(IM_END);
            toks.push(NEWLINE);
            Some(toks)
        } else {
            None
        };

        self.model
            .synthesize(&tokens, language, instruction_tokens.as_deref())
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
