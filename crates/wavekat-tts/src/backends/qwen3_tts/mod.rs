//! Qwen3-TTS backend (ONNX, 1.7B VoiceDesign).
//!
//! Runs the Qwen3-TTS-12Hz-1.7B-VoiceDesign model via ONNX Runtime.
//! Supports INT4 (weight-only quantized, default) and FP32 precision.
//!
//! ```ignore
//! use wavekat_tts::{TtsBackend, SynthesizeRequest};
//! use wavekat_tts::backends::qwen3_tts::{Qwen3Tts, ModelConfig, ModelPrecision, ExecutionProvider};
//!
//! // Auto-download INT4 model files via HF Hub, run on CPU (default):
//! let tts = Qwen3Tts::new()?;
//!
//! // Auto-download FP32, run on CPU:
//! let tts = Qwen3Tts::from_config(ModelConfig::default().with_precision(ModelPrecision::Fp32))?;
//!
//! // INT4 from a local directory, run on CUDA:
//! let tts = Qwen3Tts::from_config(
//!     ModelConfig::default()
//!         .with_dir("models/qwen3-tts-1.7b")
//!         .with_execution_provider(ExecutionProvider::Cuda),
//! )?;
//!
//! let request = SynthesizeRequest::new("Hello, world");
//! let audio = tts.synthesize(&request)?;
//! ```

use std::path::PathBuf;

use wavekat_core::AudioFrame;

use crate::error::TtsError;
use crate::traits::TtsBackend;
use crate::types::{SynthesizeRequest, VoiceInfo};

use std::sync::Once;

use tokenizer::{IM_END, IM_START, NEWLINE};

static WARNED_NO_INSTRUCTION: Once = Once::new();

mod download;
mod model;
mod sampler;
mod tokenizer;

/// ONNX model precision variant.
///
/// Selects which quantized model files to download and load.
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

/// ONNX execution provider (inference hardware backend).
///
/// Selecting a provider that is unavailable at runtime causes an error at load
/// time rather than silently falling back. Use [`ExecutionProvider::Cpu`] (the
/// default) if you need guaranteed availability.
///
/// Enable the corresponding Cargo feature to bundle the native libraries:
/// - `cuda` for [`Cuda`](ExecutionProvider::Cuda)
/// - `tensorrt` for [`TensorRt`](ExecutionProvider::TensorRt)
/// - `coreml` for [`CoreMl`](ExecutionProvider::CoreMl)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    /// CPU inference via ONNX Runtime. Always available. Default.
    #[default]
    Cpu,
    /// NVIDIA CUDA GPU inference. Requires `cuda` feature.
    Cuda,
    /// NVIDIA TensorRT. Requires `tensorrt` feature.
    TensorRt,
    /// Apple CoreML (macOS / iOS). Requires `coreml` feature.
    CoreMl,
}

/// Model loading configuration for [`Qwen3Tts`].
///
/// All fields default to sensible values: INT4 quantization, CPU inference,
/// and auto-download from HF Hub.
///
/// # Examples
///
/// ```rust,no_run
/// # use wavekat_tts::backends::qwen3_tts::{ModelConfig, ModelPrecision, ExecutionProvider};
/// // INT4, CPU, auto-download (equivalent to ModelConfig::default())
/// let config = ModelConfig::default();
///
/// // FP32 from a local directory
/// let config = ModelConfig::default()
///     .with_precision(ModelPrecision::Fp32)
///     .with_dir("models/qwen3-tts-1.7b");
///
/// // INT4, CUDA, auto-download
/// let config = ModelConfig::default()
///     .with_execution_provider(ExecutionProvider::Cuda);
/// ```
#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    /// Weight quantization variant (determines which ONNX files to load).
    pub precision: ModelPrecision,
    /// Inference hardware backend.
    pub execution_provider: ExecutionProvider,
    /// Local model directory. `None` = resolve via `WAVEKAT_MODEL_DIR` env var,
    /// then auto-download from HF Hub.
    pub model_dir: Option<PathBuf>,
}

impl ModelConfig {
    /// Set a local model directory, bypassing HF Hub download.
    ///
    /// The directory must mirror the HF repo layout:
    /// `int4/` or `fp32/`, `embeddings/`, `tokenizer/`.
    pub fn with_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.model_dir = Some(dir.into());
        self
    }

    /// Set the model precision (quantization variant).
    pub fn with_precision(mut self, precision: ModelPrecision) -> Self {
        self.precision = precision;
        self
    }

    /// Set the ONNX execution provider.
    pub fn with_execution_provider(mut self, ep: ExecutionProvider) -> Self {
        self.execution_provider = ep;
        self
    }
}

/// Qwen3-TTS backend using ONNX Runtime.
pub struct Qwen3Tts {
    model: model::Model,
    tokenizer: tokenizer::Tokenizer,
}

impl Qwen3Tts {
    /// Create a new backend with default config (INT4, CPU, auto-download).
    ///
    /// Files are cached by the HF Hub client (default `~/.cache/huggingface/hub/`).
    /// Set `HF_HOME` to change the cache root, `HF_TOKEN` for authentication, or
    /// `WAVEKAT_MODEL_DIR` to load from a local directory and skip all downloads.
    pub fn new() -> Result<Self, TtsError> {
        Self::from_config(ModelConfig::default())
    }

    /// Create a new backend with the given [`ModelConfig`].
    ///
    /// Model files are resolved in priority order:
    /// 1. `config.model_dir` (if set)
    /// 2. `WAVEKAT_MODEL_DIR` environment variable
    /// 3. Auto-download from HF Hub
    pub fn from_config(config: ModelConfig) -> Result<Self, TtsError> {
        let model_dir = download::resolve_model_dir(&config)?;
        let model = model::Model::load(model_dir.as_ref(), &config)?;
        let tokenizer = tokenizer::Tokenizer::new(&model_dir)?;
        Ok(Self { model, tokenizer })
    }
}

impl TtsBackend for Qwen3Tts {
    fn synthesize(&self, request: &SynthesizeRequest) -> Result<AudioFrame<'static>, TtsError> {
        let tokens = self.tokenizer.encode(request.text)?;
        let language = request.language.unwrap_or("en");

        if request.instruction.is_none() {
            WARNED_NO_INSTRUCTION.call_once(|| {
                eprintln!(
                    "wavekat-tts warning: Qwen3-TTS is a VoiceDesign model — \
                     synthesize quality may be inconsistent without a style instruction. \
                     Set `SynthesizeRequest::with_instruction` to control voice style."
                );
            });
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
