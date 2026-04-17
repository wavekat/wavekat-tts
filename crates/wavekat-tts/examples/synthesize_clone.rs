//! Synthesize text in a cloned voice using Qwen3-TTS 0.6B Base.
//!
//! Usage:
//!   cargo run --example synthesize_clone --features qwen3-tts -- [OPTIONS]
//!
//! Options:
//!   --ref-audio <PATH>      Reference audio WAV (24 kHz mono)
//!   --ref-text <TEXT>        Transcript of the reference audio
//!   --text <TEXT>            Text to synthesize in the cloned voice
//!   --language <LANG>        Language code (default: en)
//!   --model-dir <PATH>       Model directory (default: auto-download)
//!   --precision <PREC>       Model precision: int4 (default) or fp32
//!   --provider <EP>          Execution provider: cpu (default), cuda, tensorrt, coreml
//!   --output <PATH>          Output WAV path (default: clone_output.wav)
//!
//! Example:
//!   cargo run --example synthesize_clone --features qwen3-tts -- \
//!     --ref-audio ref.wav \
//!     --ref-text "Give every small business the voice of a big one." \
//!     --text "Your customers deserve a voice they can trust."

use std::path::PathBuf;

use wavekat_tts::backends::qwen3_tts::{
    CloneRequest, ExecutionProvider, ModelConfig, ModelPrecision, Qwen3TtsClone,
};
use wavekat_tts::AudioFrame;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut model_dir: Option<PathBuf> = None;
    let mut precision = ModelPrecision::Int4;
    let mut provider = ExecutionProvider::Cpu;
    let mut language = "en".to_string();
    let mut output = PathBuf::from("clone_output.wav");
    let mut ref_audio_path: Option<PathBuf> = None;
    let mut ref_text: Option<String> = None;
    let mut text: Option<String> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                model_dir = Some(PathBuf::from(&args[i]));
            }
            "--precision" => {
                i += 1;
                precision = match args[i].as_str() {
                    "int4" => ModelPrecision::Int4,
                    "fp32" => ModelPrecision::Fp32,
                    other => {
                        eprintln!("error: unknown precision \"{other}\"");
                        std::process::exit(1);
                    }
                };
            }
            "--provider" => {
                i += 1;
                provider = match args[i].as_str() {
                    "cpu" => ExecutionProvider::Cpu,
                    "cuda" => ExecutionProvider::Cuda,
                    "tensorrt" => ExecutionProvider::TensorRt,
                    "coreml" => ExecutionProvider::CoreMl,
                    other => {
                        eprintln!("error: unknown provider \"{other}\"");
                        std::process::exit(1);
                    }
                };
            }
            "--language" => {
                i += 1;
                language = args[i].clone();
            }
            "--output" => {
                i += 1;
                output = PathBuf::from(&args[i]);
            }
            "--ref-audio" => {
                i += 1;
                ref_audio_path = Some(PathBuf::from(&args[i]));
            }
            "--ref-text" => {
                i += 1;
                ref_text = Some(args[i].clone());
            }
            "--text" => {
                i += 1;
                text = Some(args[i].clone());
            }
            other => {
                eprintln!("error: unknown argument \"{other}\"");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let ref_audio_path = ref_audio_path.unwrap_or_else(|| {
        eprintln!("error: --ref-audio is required");
        std::process::exit(1);
    });
    let ref_text = ref_text.unwrap_or_else(|| {
        eprintln!("error: --ref-text is required");
        std::process::exit(1);
    });
    let text = text.unwrap_or_else(|| {
        eprintln!("error: --text is required");
        std::process::exit(1);
    });

    // Read reference audio WAV
    eprintln!("Reading reference audio: {} ...", ref_audio_path.display());
    let ref_audio = AudioFrame::from_wav(&ref_audio_path).expect("failed to read reference WAV");
    if ref_audio.sample_rate() != 24000 {
        eprintln!(
            "error: reference audio must be 24 kHz, got {} Hz",
            ref_audio.sample_rate()
        );
        eprintln!("hint: resample with: ffmpeg -i input.wav -ar 24000 -ac 1 ref_24k.wav");
        std::process::exit(1);
    }
    eprintln!(
        "  {:.1}s, {} Hz, {} samples",
        ref_audio.duration_secs(),
        ref_audio.sample_rate(),
        ref_audio.len(),
    );

    // Load model
    eprintln!("Loading clone model ...");
    let mut config = ModelConfig::default()
        .with_precision(precision)
        .with_execution_provider(provider);
    if let Some(dir) = model_dir {
        config = config.with_dir(dir);
    }
    let tts = Qwen3TtsClone::from_config(config).expect("failed to load clone model");

    // Synthesize
    let request =
        CloneRequest::new(&text, ref_audio.samples(), 24000, &ref_text).with_language(&language);

    eprintln!("Synthesizing: \"{text}\" (language={language})");
    let start = std::time::Instant::now();
    let audio = tts.synthesize_clone(&request).expect("synthesis failed");
    let elapsed = start.elapsed();

    let duration = audio.duration_secs();
    let rtf = elapsed.as_secs_f64() / duration;

    eprintln!(
        "Generated {} samples at {} Hz ({:.2}s) in {:.2}s (RTF: {:.2})",
        audio.len(),
        audio.sample_rate(),
        duration,
        elapsed.as_secs_f64(),
        rtf,
    );

    audio.write_wav(&output).expect("failed to write WAV");
    eprintln!("Wrote {}", output.display());
}
