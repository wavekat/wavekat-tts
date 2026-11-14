//! Kokoro-82M backend (ONNX), English and Mandarin Chinese.
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
//! Mandarin uses the `zf_*` / `zm_*` voices. Text in a script the model
//! can't read (Japanese kana, Korean, Cyrillic, …), or Chinese sent to an
//! English voice, is an [`TtsError::UnsupportedLanguage`] error rather than
//! garbled audio. Use [`detect_language`] to pick a voice up front.
//!
//! Weights: [`onnx-community/Kokoro-82M-v1.0-ONNX`](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX)
//! (Apache-2.0). English is converted to phonemes with
//! [misaki-rs](https://crates.io/crates/misaki-rs), built without espeak-ng;
//! Mandarin with [jieba-rs](https://crates.io/crates/jieba-rs) and bundled
//! pinyin tables (see the `zh` module).

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
mod zh;

use download::ModelSource;
use phonemize::{Accent, Phonemizer, Vocab};

/// Output sample rate of every Kokoro voice.
pub const SAMPLE_RATE: u32 = 24_000;

/// Voice used when the request names none.
pub const DEFAULT_VOICE: &str = "af_heart";

/// A good default for Chinese text.
pub const DEFAULT_ZH_VOICE: &str = "zf_xiaoxiao";

/// The language of a piece of text, judged by its script.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextLanguage {
    /// Latin script (and text with no letters at all).
    English,
    /// Contains Chinese characters. Latin words mixed in are read as English.
    Chinese,
}

/// What language `text` is in, as far as Kokoro is concerned.
///
/// Any Chinese character makes it [`TextLanguage::Chinese`]. Letters from a
/// script Kokoro can't read are an [`TtsError::UnsupportedLanguage`] error
/// naming the script; digits, punctuation and emoji are ignored.
pub fn detect_language(text: &str) -> Result<TextLanguage, TtsError> {
    let mut chinese = false;
    for c in text.chars() {
        if is_cjk(c) {
            chinese = true;
        } else if c.is_alphabetic() && !is_latin(c) {
            return Err(TtsError::UnsupportedLanguage(format!(
                "{} text",
                script_name(c)
            )));
        }
    }
    Ok(if chinese {
        TextLanguage::Chinese
    } else {
        TextLanguage::English
    })
}

fn is_cjk(c: char) -> bool {
    zh::is_han(c)
        || ('\u{3400}'..='\u{4DBF}').contains(&c) // Extension A
        || ('\u{F900}'..='\u{FAFF}').contains(&c) // Compatibility ideographs
}

fn is_latin(c: char) -> bool {
    c.is_ascii_alphabetic()
        || ('\u{00C0}'..='\u{024F}').contains(&c) // Latin-1 and Extended-A/B
        || ('\u{1E00}'..='\u{1EFF}').contains(&c) // Latin Extended Additional
        || ('\u{FF21}'..='\u{FF5A}').contains(&c) // Fullwidth Latin
        || c == 'ª'
        || c == 'º'
}

fn script_name(c: char) -> &'static str {
    match c as u32 {
        0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => "Japanese",
        0x1100..=0x11FF | 0x3130..=0x318F | 0xAC00..=0xD7AF => "Korean",
        0x0400..=0x052F => "Cyrillic",
        0x0370..=0x03FF => "Greek",
        0x0590..=0x05FF => "Hebrew",
        0x0600..=0x06FF | 0x0750..=0x077F => "Arabic",
        0x0900..=0x097F => "Devanagari",
        0x0E00..=0x0E7F => "Thai",
        _ => "non-Latin",
    }
}

/// Is `voice` one of the Mandarin voices?
pub fn is_mandarin_voice(voice: &str) -> bool {
    voice.starts_with('z')
}

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
        self.voice(voice)?;
        if is_mandarin_voice(voice) {
            self.phonemizer.warm_mandarin();
        }
        Ok(())
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
        let accent = accent_for(voice_id, request.language, request.text)?;
        let table = self.voice(voice_id)?;
        let rows = table.len() / STYLE_DIM;
        let speed = request.speed.unwrap_or(1.0).clamp(0.5, 2.0);

        let mut samples = Vec::new();
        for ids in self.phonemizer.chunks(request.text, accent, &self.vocab)? {
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

/// Pick the G2P for a request, refusing text the voice can't read.
fn accent_for(voice: &str, language: Option<&str>, text: &str) -> Result<Accent, TtsError> {
    let detected = detect_language(text)?;
    let mandarin = is_mandarin_voice(voice);
    if let Some(l) = language {
        let l = l.to_ascii_lowercase();
        let wants_zh = l == "zh" || l.starts_with("zh-") || l == "cmn";
        if !(l.starts_with("en") || wants_zh) {
            return Err(TtsError::UnsupportedLanguage(l));
        }
        if wants_zh != mandarin {
            return Err(TtsError::UnsupportedLanguage(format!(
                "{l} with voice `{voice}`"
            )));
        }
    }
    if mandarin {
        return Ok(Accent::Mandarin);
    }
    if detected == TextLanguage::Chinese {
        return Err(TtsError::UnsupportedLanguage(format!(
            "Chinese text with English voice `{voice}` (use a Mandarin voice such as {DEFAULT_ZH_VOICE})"
        )));
    }
    let british = match language.map(str::to_ascii_lowercase).as_deref() {
        Some("en-gb") => true,
        Some("en") | None => voice.starts_with('b'),
        Some(_) => false,
    };
    Ok(if british { Accent::Gb } else { Accent::Us })
}

/// Voices: id, display name, gender. The language comes from the id's
/// first letter: `a` US English, `b` UK English, `z` Mandarin.
///
/// The v1.0 model has two more Mandarin voices, `zf_xiaoni` and
/// `zm_yunxia`; they are left out because speech recognition misheard
/// roughly half of what they said with the fp16 export.
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
    ("zf_xiaoxiao", "Xiaoxiao", Gender::Female),
    ("zf_xiaobei", "Xiaobei", Gender::Female),
    ("zf_xiaoyi", "Xiaoyi", Gender::Female),
    ("zm_yunxi", "Yunxi", Gender::Male),
    ("zm_yunyang", "Yunyang", Gender::Male),
    ("zm_yunjian", "Yunjian", Gender::Male),
];

/// All voices, without loading the model.
pub fn voices() -> Vec<VoiceInfo> {
    VOICES
        .iter()
        .map(|(id, name, gender)| VoiceInfo {
            id: id.to_string(),
            name: name.to_string(),
            languages: vec![match id.as_bytes()[0] {
                b'b' => "en-GB",
                b'z' => "zh-CN",
                _ => "en-US",
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
    fn voice_list_has_defaults_and_languages() {
        let v = voices();
        assert!(v.iter().any(|v| v.id == DEFAULT_VOICE));
        assert!(v.iter().any(|v| v.id == DEFAULT_ZH_VOICE));
        for voice in &v {
            let want = if is_mandarin_voice(&voice.id) {
                "zh-"
            } else {
                "en-"
            };
            assert!(voice.languages[0].starts_with(want), "{}", voice.id);
        }
    }

    #[test]
    fn detects_language_by_script() {
        use TextLanguage::*;
        assert_eq!(detect_language("Hello, world 42!").unwrap(), English);
        assert_eq!(detect_language("Café déjà vu").unwrap(), English);
        assert_eq!(detect_language("").unwrap(), English);
        assert_eq!(detect_language("您好，订单 OK 了 🎉").unwrap(), Chinese);
        for (text, script) in [
            ("こんにちは", "Japanese"),
            ("漢字とかな", "Japanese"),
            ("안녕하세요", "Korean"),
            ("Привет", "Cyrillic"),
            ("مرحبا", "Arabic"),
        ] {
            let err = detect_language(text).unwrap_err().to_string();
            assert!(err.contains(script), "{text}: {err}");
        }
    }

    #[test]
    fn accent_follows_voice_and_text() {
        assert_eq!(accent_for("af_heart", None, "Hi").unwrap(), Accent::Us);
        assert_eq!(accent_for("bf_emma", None, "Hi").unwrap(), Accent::Gb);
        assert_eq!(
            accent_for("af_heart", Some("en-gb"), "Hi").unwrap(),
            Accent::Gb
        );
        assert_eq!(
            accent_for("zf_xiaoxiao", None, "你好").unwrap(),
            Accent::Mandarin
        );
        // A Mandarin voice also reads plain English (with an accent).
        assert_eq!(
            accent_for("zm_yunxi", None, "Hi").unwrap(),
            Accent::Mandarin
        );
        assert_eq!(
            accent_for("zm_yunxi", Some("zh-CN"), "你好").unwrap(),
            Accent::Mandarin
        );
        // Chinese to an English voice is refused, with a hint.
        let err = accent_for("af_heart", None, "你好")
            .unwrap_err()
            .to_string();
        assert!(err.contains(DEFAULT_ZH_VOICE), "{err}");
        // A language that doesn't match the voice is refused.
        assert!(accent_for("af_heart", Some("zh"), "Hi").is_err());
        assert!(accent_for("zf_xiaoxiao", Some("en"), "Hi").is_err());
        assert!(accent_for("af_heart", Some("fr"), "Salut").is_err());
        // Unreadable scripts are refused for every voice.
        assert!(accent_for("zf_xiaoxiao", None, "こんにちは").is_err());
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

    /// Downloads the model; run with `cargo test --features kokoro -- --ignored`.
    #[test]
    #[ignore]
    fn synthesizes_mandarin() {
        let tts = Kokoro::new().unwrap();
        let audio = tts
            .synthesize(
                &SynthesizeRequest::new("您好，您的订单已经确认。").with_voice(DEFAULT_ZH_VOICE),
            )
            .unwrap();
        assert!(audio.duration_secs() > 1.0);
    }
}
