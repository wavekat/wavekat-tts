//! Kokoro-82M backend (ONNX), English.
//!
//! A small, fast model: on a laptop CPU it synthesizes many times faster
//! than real time, which suits short prompts such as phone messages.
//!
//! ```ignore
//! use wavekat_tts::{TtsBackend, SynthesizeRequest};
//! use wavekat_tts::backends::kokoro::Kokoro;
//!
//! let tts = Kokoro::new()?; // downloads ~165 MB on first use
//! let audio = tts.synthesize(&SynthesizeRequest::new("Hello there.").with_voice("af_heart"))?;
//! audio.write_wav("hello.wav")?;
//! ```
//!
//! Weights: [`onnx-community/Kokoro-82M-v1.0-ONNX`](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)
//! (Apache-2.0). Text is converted to phonemes with
//! [misaki-rs](https://crates.io/crates/misaki-rs), built without espeak-ng.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::value::TensorRef;
use wavekat_core::AudioFrame;

use crate::error::TtsError;
use crate::traits::TtsBackend;
use crate::types::{Gender, SynthesizeRequest, VoiceInfo};

mod download;
mod phonemize;

use download::ModelSource;
use phonemize::{Phonemizer, Vocab};

/// Output sample rate of every Kokoro voice.
pub const SAMPLE_RATE: u32 = 24_000;

/// Voice used when the request names none.
pub const DEFAULT_VOICE: &str = "af_heart";

/// Width of one voice style vector.
const STYLE_DIM: usize = 256;

/// Which ONNX export of the model to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelVariant {
    /// 8-bit weights, fp16 activations (~86 MB). Smallest download, but
    /// about 2.5x slower than fp16 on CPU.
    Q8f16,
    /// 8-bit dynamic quantization (~92 MB). Also slower than fp16 on CPU.
    Quantized,
    /// fp16 (~163 MB). Default: as fast as fp32 on CPU at half the size.
    #[default]
    Fp16,
    /// Full precision (~326 MB).
    Fp32,
}

impl ModelVariant {
    /// File name inside the model repo.
    pub fn file(self) -> &'static str {
        match self {
            Self::Q8f16 => "onnx/model_q8f16.onnx",
            Self::Quantized => "onnx/model_quantized.onnx",
            Self::Fp16 => "onnx/model_fp16.onnx",
            Self::Fp32 => "onnx/model.onnx",
        }
    }

    /// Parse `q8f16`, `quantized`, `fp16` or `fp32`.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "q8f16" => Some(Self::Q8f16),
            "quantized" | "q8" | "int8" => Some(Self::Quantized),
            "fp16" => Some(Self::Fp16),
            "fp32" => Some(Self::Fp32),
            _ => None,
        }
    }
}

/// ONNX execution provider for Kokoro.
///
/// A provider that isn't available at runtime is an error at load time, not
/// a silent fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    /// CPU. Always available. Default.
    #[default]
    Cpu,
    /// NVIDIA CUDA. Requires the `cuda` feature.
    Cuda,
    /// Apple CoreML. Requires the `coreml` feature.
    CoreMl,
}

/// Loading options for [`Kokoro`].
#[derive(Debug, Clone, Default)]
pub struct KokoroConfig {
    /// Which ONNX export to load.
    pub variant: ModelVariant,
    /// Inference hardware.
    pub execution_provider: ExecutionProvider,
    /// Local directory mirroring the HF repo (`tokenizer.json`, `onnx/`,
    /// `voices/`). `None` = `WAVEKAT_KOKORO_MODEL_DIR`, then HF Hub.
    pub model_dir: Option<PathBuf>,
    /// Intra-op threads for ONNX Runtime. `None` = ONNX Runtime's default.
    pub threads: Option<usize>,
}

impl KokoroConfig {
    /// Set the model variant.
    pub fn with_variant(mut self, variant: ModelVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Set the execution provider.
    pub fn with_execution_provider(mut self, ep: ExecutionProvider) -> Self {
        self.execution_provider = ep;
        self
    }

    /// Load from a local directory and never download.
    pub fn with_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.model_dir = Some(dir.into());
        self
    }

    /// Set the number of intra-op threads.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = Some(threads);
        self
    }
}

/// Kokoro-82M text-to-speech. Implements [`TtsBackend`].
///
/// Voices are downloaded the first time they are used and then cached.
pub struct Kokoro {
    session: Mutex<Session>,
    vocab: Vocab,
    phonemizer: Phonemizer,
    source: ModelSource,
    voices: Mutex<HashMap<String, Arc<Vec<f32>>>>,
}

impl Kokoro {
    /// Load with the default config (fp16, CPU, auto-download).
    pub fn new() -> Result<Self, TtsError> {
        Self::from_config(KokoroConfig::default())
    }

    /// Load with `config`.
    pub fn from_config(config: KokoroConfig) -> Result<Self, TtsError> {
        let source = ModelSource::resolve(&config)?;
        let vocab = Vocab::load(&source.file("tokenizer.json")?)?;
        let model_path = source.file(config.variant.file())?;

        let mut builder = Session::builder()
            .map_err(|e| TtsError::Model(format!("session builder error: {e}")))?;
        if let Some(n) = config.threads {
            builder = builder
                .with_intra_threads(n)
                .map_err(|e| TtsError::Model(format!("thread config error: {e}")))?;
        }
        let session = apply_execution_provider(builder, config.execution_provider)?
            .commit_from_file(&model_path)
            .map_err(|e| {
                TtsError::Model(format!("failed to load {}: {e}", model_path.display()))
            })?;

        Ok(Self {
            session: Mutex::new(session),
            vocab,
            phonemizer: Phonemizer::new(),
            source,
            voices: Mutex::new(HashMap::new()),
        })
    }

    /// Download (if needed) and load a voice's style table.
    ///
    /// Call this ahead of time to keep the first synthesis fast.
    pub fn load_voice(&self, voice: &str) -> Result<(), TtsError> {
        self.voice(voice).map(|_| ())
    }

    fn voice(&self, voice: &str) -> Result<Arc<Vec<f32>>, TtsError> {
        if !VOICES.iter().any(|v| v.0 == voice) {
            return Err(TtsError::VoiceNotFound(voice.to_string()));
        }
        if let Some(v) = self.voices.lock().unwrap().get(voice) {
            return Ok(v.clone());
        }
        let path = self.source.file(&format!("voices/{voice}.bin"))?;
        let bytes = std::fs::read(&path)?;
        if bytes.len() % (4 * STYLE_DIM) != 0 || bytes.is_empty() {
            return Err(TtsError::Model(format!(
                "bad voice file {}",
                path.display()
            )));
        }
        let table: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let table = Arc::new(table);
        self.voices
            .lock()
            .unwrap()
            .insert(voice.to_string(), table.clone());
        Ok(table)
    }

    fn run(&self, ids: &[i64], style: &[f32], speed: f32) -> Result<Vec<f32>, TtsError> {
        let mut padded = Vec::with_capacity(ids.len() + 2);
        padded.push(0);
        padded.extend_from_slice(ids);
        padded.push(0);
        let input_ids = Array2::from_shape_vec((1, padded.len()), padded)
            .map_err(|e| TtsError::Synthesis(format!("input_ids shape: {e}")))?;
        let style = Array2::from_shape_vec((1, STYLE_DIM), style.to_vec())
            .map_err(|e| TtsError::Synthesis(format!("style shape: {e}")))?;
        let speed = Array1::from_vec(vec![speed]);

        let t_ids = TensorRef::from_array_view(&input_ids)
            .map_err(|e| TtsError::Synthesis(format!("tensor input_ids: {e}")))?;
        let t_style = TensorRef::from_array_view(&style)
            .map_err(|e| TtsError::Synthesis(format!("tensor style: {e}")))?;
        let t_speed = TensorRef::from_array_view(&speed)
            .map_err(|e| TtsError::Synthesis(format!("tensor speed: {e}")))?;

        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![
                "input_ids" => t_ids,
                "style" => t_style,
                "speed" => t_speed,
            ])
            .map_err(|e| TtsError::Synthesis(format!("kokoro inference failed: {e}")))?;
        let (_, wave) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::Synthesis(format!("extract waveform: {e}")))?;
        Ok(wave.to_vec())
    }
}

impl TtsBackend for Kokoro {
    fn synthesize(&self, request: &SynthesizeRequest) -> Result<AudioFrame<'static>, TtsError> {
        let voice_id = request.voice.unwrap_or(DEFAULT_VOICE);
        let british = match request.language {
            Some(l) => {
                let l = l.to_ascii_lowercase();
                if !l.starts_with("en") {
                    return Err(TtsError::UnsupportedLanguage(l));
                }
                l == "en-gb" || (l == "en" && voice_id.starts_with('b'))
            }
            None => voice_id.starts_with('b'),
        };
        let table = self.voice(voice_id)?;
        let rows = table.len() / STYLE_DIM;
        let speed = request.speed.unwrap_or(1.0).clamp(0.5, 2.0);

        let mut samples = Vec::new();
        for ids in self.phonemizer.chunks(request.text, british, &self.vocab)? {
            // The style vector depends on the utterance length.
            let row = ids.len().min(rows - 1);
            let style = &table[row * STYLE_DIM..(row + 1) * STYLE_DIM];
            samples.extend(self.run(&ids, style, speed)?);
        }
        Ok(AudioFrame::from_vec(samples, SAMPLE_RATE))
    }

    fn voices(&self) -> Result<Vec<VoiceInfo>, TtsError> {
        Ok(voices())
    }
}

/// English voices: id, display name, gender.
const VOICES: &[(&str, &str, Gender)] = &[
    ("af_heart", "Heart", Gender::Female),
    ("af_alloy", "Alloy", Gender::Female),
    ("af_aoede", "Aoede", Gender::Female),
    ("af_bella", "Bella", Gender::Female),
    ("af_jessica", "Jessica", Gender::Female),
    ("af_kore", "Kore", Gender::Female),
    ("af_nicole", "Nicole", Gender::Female),
    ("af_nova", "Nova", Gender::Female),
    ("af_river", "River", Gender::Female),
    ("af_sarah", "Sarah", Gender::Female),
    ("af_sky", "Sky", Gender::Female),
    ("am_adam", "Adam", Gender::Male),
    ("am_echo", "Echo", Gender::Male),
    ("am_eric", "Eric", Gender::Male),
    ("am_fenrir", "Fenrir", Gender::Male),
    ("am_liam", "Liam", Gender::Male),
    ("am_michael", "Michael", Gender::Male),
    ("am_onyx", "Onyx", Gender::Male),
    ("am_puck", "Puck", Gender::Male),
    ("bf_alice", "Alice", Gender::Female),
    ("bf_emma", "Emma", Gender::Female),
    ("bf_isabella", "Isabella", Gender::Female),
    ("bf_lily", "Lily", Gender::Female),
    ("bm_daniel", "Daniel", Gender::Male),
    ("bm_fable", "Fable", Gender::Male),
    ("bm_george", "George", Gender::Male),
    ("bm_lewis", "Lewis", Gender::Male),
];

/// All English voices, without loading the model.
pub fn voices() -> Vec<VoiceInfo> {
    VOICES
        .iter()
        .map(|(id, name, gender)| VoiceInfo {
            id: id.to_string(),
            name: name.to_string(),
            languages: vec![if id.starts_with('b') {
                "en-GB"
            } else {
                "en-US"
            }
            .to_string()],
            gender: Some(*gender),
        })
        .collect()
}

fn apply_execution_provider(
    builder: ort::session::builder::SessionBuilder,
    ep: ExecutionProvider,
) -> Result<ort::session::builder::SessionBuilder, TtsError> {
    use ort::execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider};
    match ep {
        ExecutionProvider::Cpu => Ok(builder),
        ExecutionProvider::Cuda => builder
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
            .map_err(|e| TtsError::Model(format!("CUDA execution provider error: {e}"))),
        ExecutionProvider::CoreMl => builder
            .with_execution_providers([CoreMLExecutionProvider::default()
                .build()
                .error_on_failure()])
            .map_err(|e| TtsError::Model(format!("CoreML execution provider error: {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_list_is_english_and_has_default() {
        let v = voices();
        assert!(v.iter().any(|v| v.id == DEFAULT_VOICE));
        assert!(v.iter().all(|v| v.languages[0].starts_with("en-")));
    }

    #[test]
    fn variant_parse_round_trips() {
        for v in [
            ModelVariant::Q8f16,
            ModelVariant::Quantized,
            ModelVariant::Fp16,
            ModelVariant::Fp32,
        ] {
            let name = v
                .file()
                .trim_start_matches("onnx/model")
                .trim_end_matches(".onnx");
            let name = name.trim_start_matches('_');
            let parsed = ModelVariant::parse(if name.is_empty() { "fp32" } else { name });
            assert_eq!(parsed, Some(v));
        }
    }

    /// Downloads the model; run with `cargo test --features kokoro -- --ignored`.
    #[test]
    #[ignore]
    fn synthesizes_audio() {
        let tts = Kokoro::new().unwrap();
        let audio = tts
            .synthesize(&SynthesizeRequest::new(
                "Hello from Kokoro. Version 2 is ready.",
            ))
            .unwrap();
        assert_eq!(audio.sample_rate(), SAMPLE_RATE);
        assert!(audio.duration_secs() > 1.0);
    }
}
