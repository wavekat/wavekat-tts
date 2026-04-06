//! Synthesize text to a WAV file using Qwen3-TTS.
//!
//! Usage:
//!   cargo run --example synthesize --features qwen3-tts,hound -- [OPTIONS] [TEXT]
//!
//! Options:
//!   --model-dir <PATH>   Model directory (default: auto-download to cache)
//!   --language <LANG>    Language code (default: en)
//!   --output <PATH>      Output WAV path (default: output.wav)
//!   -i, --interactive    Interactive mode: keep model loaded, read text from stdin
//!
//! Example:
//!   cargo run --example synthesize --features qwen3-tts,hound -- "Hello, world!"
//!   cargo run --example synthesize --features qwen3-tts,hound -- -i

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use wavekat_tts::backends::qwen3_tts::Qwen3Tts;
use wavekat_tts::{SynthesizeRequest, TtsBackend};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut model_dir: Option<PathBuf> = None;
    let mut language = "en".to_string();
    let mut output = PathBuf::from("output.wav");
    let mut interactive = false;
    let mut text_parts: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => {
                i += 1;
                model_dir = Some(PathBuf::from(&args[i]));
            }
            "--language" => {
                i += 1;
                language = args[i].clone();
            }
            "--output" => {
                i += 1;
                output = PathBuf::from(&args[i]);
            }
            "-i" | "--interactive" => interactive = true,
            other => text_parts.push(other.to_string()),
        }
        i += 1;
    }

    let text = text_parts.join(" ");
    if text.is_empty() && !interactive {
        eprintln!("Usage: synthesize [OPTIONS] [TEXT]");
        eprintln!("  --model-dir <PATH>  Model directory (default: auto-download)");
        eprintln!("  --language <LANG>   Language code (default: en)");
        eprintln!("  --output <PATH>     Output WAV path (default: output.wav)");
        eprintln!("  -i, --interactive   Interactive mode (read from stdin)");
        std::process::exit(1);
    }

    eprintln!("Loading model ...");
    let tts = match model_dir {
        Some(dir) => Qwen3Tts::from_dir(dir).expect("failed to load model"),
        None => Qwen3Tts::new().expect("failed to load model"),
    };

    if interactive {
        run_interactive(&tts, &language, &output);
    } else {
        synthesize_one(&tts, &text, &language, &output);
    }
}

fn run_interactive(tts: &Qwen3Tts, language: &str, default_output: &PathBuf) {
    eprintln!("Interactive mode. Type text to synthesize, empty line to quit.");
    let stdin = io::stdin();
    let mut count = 0u32;

    loop {
        eprint!("> ");
        io::stderr().flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let text = line.trim();
        if text.is_empty() {
            break;
        }

        count += 1;
        let output = if *default_output == PathBuf::from("output.wav") {
            PathBuf::from(format!("output_{count:03}.wav"))
        } else {
            default_output.clone()
        };

        synthesize_one(tts, text, language, &output);
    }
}

fn synthesize_one(tts: &Qwen3Tts, text: &str, language: &str, output: &PathBuf) {
    let request = SynthesizeRequest::new(text).with_language(language);

    eprintln!("Synthesizing: \"{text}\" (language={language})");
    let start = std::time::Instant::now();
    let audio = tts.synthesize(&request).expect("synthesis failed");
    let elapsed = start.elapsed();

    let duration = audio.duration_secs();
    let rtf = elapsed.as_secs_f64() / duration as f64;

    eprintln!(
        "Generated {} samples at {} Hz ({:.2}s) in {:.2}s (RTF: {:.2})",
        audio.len(),
        audio.sample_rate(),
        duration,
        elapsed.as_secs_f64(),
        rtf,
    );

    // Write WAV
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate(),
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(output, spec).expect("failed to create WAV file");
    for &sample in audio.samples() {
        writer.write_sample(sample).expect("failed to write sample");
    }
    writer.finalize().expect("failed to finalize WAV");

    eprintln!("Wrote {}", output.display());
}
