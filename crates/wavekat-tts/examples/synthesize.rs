//! Synthesize text to a WAV file using Qwen3-TTS.
//!
//! Usage:
//!   cargo run --example synthesize --features qwen3-tts -- [OPTIONS] [TEXT]
//!
//! Options:
//!   --model-dir <PATH>      Model directory (default: auto-download to cache)
//!   --precision <PREC>      Model precision: int4 (default) or fp32
//!   --language <LANG>       Language code (default: en)
//!   --instruction <TEXT>    Voice style instruction (VoiceDesign prompt)
//!                           Default: "Speak naturally and clearly."
//!   --output <PATH>         Output WAV path (default: output.wav)
//!   -i, --interactive       Interactive mode: keep model loaded, read text from stdin
//!
//! Interactive commands (prefix with /):
//!   /lang <code>            Switch language (e.g. /lang ja)
//!   /langs                  List supported language codes
//!   /instruct <text>        Change voice instruction (e.g. /instruct Speak slowly.)
//!   /instruct               Reset instruction to default
//!   /status                 Show current settings
//!   /help                   Show this command list
//!   Empty line or Ctrl-D    Quit
//!
//! Example:
//!   cargo run --example synthesize --features qwen3-tts -- "Hello, world!"
//!   cargo run --example synthesize --features qwen3-tts -- -i
//!   cargo run --example synthesize --features qwen3-tts -- --precision fp32 -i

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use wavekat_tts::backends::qwen3_tts::{ModelPrecision, Qwen3Tts};
use wavekat_tts::{SynthesizeRequest, TtsBackend};

const DEFAULT_INSTRUCTION: &str = "Speak naturally and clearly.";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut model_dir: Option<PathBuf> = None;
    let mut precision = ModelPrecision::Int4;
    let mut language = "en".to_string();
    let mut instruction: Option<String> = None;
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
            "--precision" => {
                i += 1;
                precision = match args[i].as_str() {
                    "int4" => ModelPrecision::Int4,
                    "fp32" => ModelPrecision::Fp32,
                    other => {
                        eprintln!("error: unknown precision \"{other}\", expected int4 or fp32");
                        std::process::exit(1);
                    }
                };
            }
            "--language" => {
                i += 1;
                language = args[i].clone();
            }
            "--instruction" => {
                i += 1;
                instruction = Some(args[i].clone());
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
        eprintln!("  --model-dir <PATH>       Model directory (default: auto-download)");
        eprintln!("  --precision <PREC>       Model precision: int4 (default) or fp32");
        eprintln!("  --language <LANG>        Language code (default: en)");
        eprintln!("  --instruction <TEXT>     Voice style instruction (VoiceDesign prompt)");
        eprintln!("                           Default: \"{DEFAULT_INSTRUCTION}\"");
        eprintln!("  --output <PATH>          Output WAV path (default: output.wav)");
        eprintln!("  -i, --interactive        Interactive mode (read from stdin)");
        std::process::exit(1);
    }

    if instruction.is_none() {
        eprintln!("note: no --instruction given, using default: \"{DEFAULT_INSTRUCTION}\"");
        instruction = Some(DEFAULT_INSTRUCTION.to_string());
    }

    eprintln!("Loading model ...");
    let tts = match model_dir {
        Some(dir) => Qwen3Tts::from_dir(dir, precision).expect("failed to load model"),
        None => Qwen3Tts::new_with_precision(precision).expect("failed to load model"),
    };

    if interactive {
        run_interactive(&tts, language, instruction.unwrap(), &output);
    } else {
        synthesize_one(&tts, &text, &language, instruction.as_deref(), &output);
    }
}

fn run_interactive(
    tts: &Qwen3Tts,
    mut language: String,
    mut instruction: String,
    default_output: &PathBuf,
) {
    let supported_langs: Vec<String> = tts
        .voices()
        .unwrap_or_default()
        .into_iter()
        .flat_map(|v| v.languages)
        .collect();

    eprintln!("Interactive mode. Type text to synthesize, /help for commands, empty line to quit.");
    eprintln!("  language={language}  instruction=\"{instruction}\"");

    let stdin = io::stdin();
    let mut count = 0u32;

    loop {
        eprint!("> ");
        io::stderr().flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            break;
        }

        if let Some(rest) = input.strip_prefix('/') {
            let (cmd, arg) = rest
                .split_once(' ')
                .map_or((rest, ""), |(c, a)| (c, a.trim()));
            match cmd {
                "lang" | "language" => {
                    if arg.is_empty() {
                        eprintln!("usage: /lang <code>  — type /langs to list supported codes");
                    } else if !supported_langs.iter().any(|l| l == arg) {
                        eprintln!("unsupported language: \"{arg}\"");
                        eprintln!("supported: {}", supported_langs.join(", "));
                    } else {
                        language = arg.to_string();
                        eprintln!("language set to: {language}");
                    }
                }
                "langs" | "languages" => {
                    eprintln!("supported languages: {}", supported_langs.join(", "));
                }
                "instruct" | "instruction" => {
                    if arg.is_empty() {
                        instruction = DEFAULT_INSTRUCTION.to_string();
                        eprintln!("instruction reset to default: \"{instruction}\"");
                    } else {
                        instruction = arg.to_string();
                        eprintln!("instruction set to: \"{instruction}\"");
                    }
                }
                "status" => {
                    eprintln!("  language={language}");
                    eprintln!("  instruction=\"{instruction}\"");
                    eprintln!("  supported languages: {}", supported_langs.join(", "));
                }
                "help" => {
                    eprintln!("  /lang <code>        Switch language");
                    eprintln!("  /langs              List supported language codes");
                    eprintln!("  /instruct <text>    Change voice instruction");
                    eprintln!("  /instruct           Reset instruction to default");
                    eprintln!("  /status             Show current settings");
                    eprintln!("  /help               Show this help");
                    eprintln!("  Empty line          Quit");
                }
                other => eprintln!("unknown command: /{other}  (type /help for commands)"),
            }
            continue;
        }

        count += 1;
        let output = if *default_output == PathBuf::from("output.wav") {
            PathBuf::from(format!("output_{count:03}.wav"))
        } else {
            default_output.clone()
        };

        synthesize_one(tts, input, &language, Some(&instruction), &output);
    }
}

fn synthesize_one(
    tts: &Qwen3Tts,
    text: &str,
    language: &str,
    instruction: Option<&str>,
    output: &PathBuf,
) {
    let mut request = SynthesizeRequest::new(text).with_language(language);
    if let Some(instr) = instruction {
        request = request.with_instruction(instr);
    }

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

    audio.write_wav(output).expect("failed to write WAV");

    eprintln!("Wrote {}", output.display());
}
