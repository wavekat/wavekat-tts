//! Qwen3-TTS backends (ONNX).
//!
//! Two sibling structs:
//!
//! - [`Qwen3Tts`] — 1.7B VoiceDesign (prompt-based voice styling)
//! - [`Qwen3TtsClone`] — 0.6B Base (reference-audio voice cloning)
//!
//! # VoiceDesign (1.7B)
//!
//! ```ignore
//! use wavekat_tts::{TtsBackend, SynthesizeRequest};
//! use wavekat_tts::backends::qwen3_tts::{Qwen3Tts, ModelConfig, ModelPrecision, ExecutionProvider};
//!
//! let tts = Qwen3Tts::new()?;
//! let request = SynthesizeRequest::new("Hello, world");
//! let audio = tts.synthesize(&request)?;
//! ```
//!
//! # Voice Clone (0.6B)
//!
//! ```ignore
//! use wavekat_tts::backends::qwen3_tts::{Qwen3TtsClone, CloneRequest, ModelConfig};
//!
//! let tts = Qwen3TtsClone::new()?;
//! let ref_audio: Vec<f32> = todo!("24 kHz mono PCM");
//! let req = CloneRequest::new("Text to say", &ref_audio, 24000, "Transcript of ref audio");
//! let audio = tts.synthesize_clone(&req)?;
//! ```

use std::path::PathBuf;

use wavekat_core::AudioFrame;

use crate::error::TtsError;
use crate::traits::TtsBackend;
use crate::types::{SynthesizeRequest, VoiceInfo};

use std::sync::Once;

use tokenizer::{IM_END, IM_START, NEWLINE};

static WARNED_NO_INSTRUCTION: Once = Once::new();

mod clone_model;
mod download;
mod mel;
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

/// Model loading configuration for [`Qwen3Tts`] and [`Qwen3TtsClone`].
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

/// Qwen3-TTS 1.7B VoiceDesign backend using ONNX Runtime.
///
/// Generates speech from text using a style instruction to control voice
/// characteristics (tone, pace, emotion). Implements [`TtsBackend`].
///
/// # Examples
///
/// ```rust,no_run
/// use wavekat_tts::{TtsBackend, SynthesizeRequest};
/// use wavekat_tts::backends::qwen3_tts::Qwen3Tts;
///
/// let tts = Qwen3Tts::new()?;
/// let audio = tts.synthesize(
///     &SynthesizeRequest::new("Hello, world")
///         .with_instruction("Speak naturally and clearly."),
/// )?;
/// audio.write_wav("output.wav")?;
/// # Ok::<(), wavekat_tts::TtsError>(())
/// ```
///
/// Use [`Qwen3Tts::from_config`] with [`ModelConfig`] to select FP32
/// precision or a GPU execution provider.
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

// ---------------------------------------------------------------------------
// Voice Clone (0.6B Base)
// ---------------------------------------------------------------------------

/// A voice-clone synthesis request.
///
/// Requires a reference audio clip (3–10 s, mono) and its transcript. The
/// model produces speech in the cloned voice speaking `text`.
///
/// Reference audio **must be 24 kHz mono float32 PCM**. If your audio is at
/// a different sample rate, resample before passing it in.
///
/// # Examples
///
/// ```rust,no_run
/// use wavekat_tts::backends::qwen3_tts::CloneRequest;
///
/// let ref_samples: Vec<f32> = vec![]; // 24 kHz mono float32
/// let req = CloneRequest::new("Hello", &ref_samples, 24000, "ref transcript")
///     .with_language("en");
/// ```
#[derive(Debug, Clone)]
pub struct CloneRequest<'a> {
    /// Text to synthesize in the cloned voice.
    pub text: &'a str,
    /// Reference audio samples (24 kHz mono float32).
    pub ref_samples: &'a [f32],
    /// Sample rate of `ref_samples` (must be 24000).
    pub ref_sample_rate: u32,
    /// Transcript of the reference audio (required for ICL mode).
    pub ref_text: &'a str,
    /// Language code (e.g. `"en"`, `"zh"`). `None` defaults to `"en"`.
    pub language: Option<&'a str>,
}

impl<'a> CloneRequest<'a> {
    /// Create a clone request with all required fields.
    pub fn new(
        text: &'a str,
        ref_samples: &'a [f32],
        ref_sample_rate: u32,
        ref_text: &'a str,
    ) -> Self {
        Self {
            text,
            ref_samples,
            ref_sample_rate,
            ref_text,
            language: None,
        }
    }

    /// Set the language code.
    pub fn with_language(mut self, language: &'a str) -> Self {
        self.language = Some(language);
        self
    }
}

/// Qwen3-TTS 0.6B Base voice-clone backend using ONNX Runtime.
///
/// Clones a speaker's voice from a short reference clip (3–10 s) and its
/// transcript, then synthesizes new text in that voice.
///
/// # Examples
///
/// ```rust,no_run
/// use wavekat_tts::AudioFrame;
/// use wavekat_tts::backends::qwen3_tts::{Qwen3TtsClone, CloneRequest};
///
/// let ref_audio = AudioFrame::from_wav("ref.wav")?;
/// let tts = Qwen3TtsClone::new()?;
/// let req = CloneRequest::new(
///     "Text to say in the cloned voice",
///     ref_audio.samples(),
///     24000,
///     "Transcript of the reference clip.",
/// );
/// let audio = tts.synthesize_clone(&req)?;
/// audio.write_wav("clone_output.wav")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// Use [`Qwen3TtsClone::from_config`] with [`ModelConfig`] to select FP32
/// precision or a GPU execution provider.
pub struct Qwen3TtsClone {
    model: clone_model::CloneModel,
    tokenizer: tokenizer::Tokenizer,
}

impl Qwen3TtsClone {
    /// Create a new clone backend with default config (INT4, CPU, auto-download).
    pub fn new() -> Result<Self, TtsError> {
        Self::from_config(ModelConfig::default())
    }

    /// Create a new clone backend with the given [`ModelConfig`].
    pub fn from_config(config: ModelConfig) -> Result<Self, TtsError> {
        let model_dir = download::resolve_clone_model_dir(&config)?;
        let model = clone_model::CloneModel::load(model_dir.as_ref(), &config)?;
        let tokenizer = tokenizer::Tokenizer::new(&model_dir)?;
        Ok(Self { model, tokenizer })
    }

    /// Synthesize text in a cloned voice.
    pub fn synthesize_clone(
        &self,
        request: &CloneRequest,
    ) -> Result<AudioFrame<'static>, TtsError> {
        if request.ref_sample_rate != 24000 {
            return Err(TtsError::Synthesis(format!(
                "reference audio must be 24 kHz, got {} Hz",
                request.ref_sample_rate,
            )));
        }

        let language = request.language.unwrap_or("en");
        let ref_tokens = self.tokenizer.encode(request.ref_text)?;
        let text_tokens = self.tokenizer.encode(request.text)?;

        self.model
            .synthesize(request.ref_samples, &ref_tokens, &text_tokens, language)
    }
}

// ---------------------------------------------------------------------------
// TtsBackend for Qwen3Tts (1.7B VoiceDesign)
// ---------------------------------------------------------------------------

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
