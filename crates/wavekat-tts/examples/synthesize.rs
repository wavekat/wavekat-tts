//! Synthesize text to a WAV file using Qwen3-TTS.
//!
//! Usage:
//!   cargo run --example synthesize --features qwen3-tts,hound -- [OPTIONS] <TEXT>
//!
//! Options:
//!   --model-dir <PATH>   Model directory (default: models/qwen3-tts-0.6b)
//!   --language <LANG>    Language code (default: en)
//!   --output <PATH>      Output WAV path (default: output.wav)
//!
//! Example:
//!   cargo run --example synthesize --features qwen3-tts,hound -- "Hello, world!"
//!   cargo run --example synthesize --features qwen3-tts,hound -- --language zh "你好世界"

use std::path::PathBuf;

use wavekat_tts::backends::qwen3_tts::Qwen3Tts;
use wavekat_tts::{SynthesizeRequest, TtsBackend};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut model_dir = PathBuf::from("models/qwen3-tts-0.6b");
    let mut language = "en".to_string();
    let mut output = PathBuf::from("output.wav");
    let mut text_parts: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                model_dir = PathBuf::from(&args[i]);
            }
            "--language" => {
                i += 1;
                language = args[i].clone();
            }
            "--output" => {
                i += 1;
                output = PathBuf::from(&args[i]);
            }
            other => text_parts.push(other.to_string()),
        }
        i += 1;
    }

    let text = text_parts.join(" ");
    if text.is_empty() {
        eprintln!("Usage: synthesize [OPTIONS] <TEXT>");
        eprintln!("  --model-dir <PATH>  Model directory (default: models/qwen3-tts-0.6b)");
        eprintln!("  --language <LANG>   Language code (default: en)");
        eprintln!("  --output <PATH>     Output WAV path (default: output.wav)");
        std::process::exit(1);
    }

    eprintln!("Loading model from {} ...", model_dir.display());
    let tts = Qwen3Tts::new(&model_dir).expect("failed to load model");

    let request = SynthesizeRequest::new(&text).with_language(&language);

    eprintln!("Synthesizing: \"{text}\" (language={language})");
    let audio = tts.synthesize(&request).expect("synthesis failed");

    eprintln!(
        "Generated {} samples at {} Hz ({:.2}s)",
        audio.len(),
        audio.sample_rate(),
        audio.duration_secs()
    );

    // Write WAV
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate(),
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(&output, spec).expect("failed to create WAV file");
    for &sample in audio.samples() {
        writer.write_sample(sample).expect("failed to write sample");
    }
    writer.finalize().expect("failed to finalize WAV");

    eprintln!("Wrote {}", output.display());
}
