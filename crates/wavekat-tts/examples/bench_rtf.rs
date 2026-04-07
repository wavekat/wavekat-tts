//! RTF benchmark for Qwen3-TTS.
//!
//! Measures Real-Time Factor (RTF = synthesis_time / audio_duration) across
//! different text lengths. RTF < 1.0 means faster-than-real-time synthesis.
//!
//! Usage:
//!   cargo run --release --example bench_rtf --features "qwen3-tts,cuda" -- --provider cuda
//!
//! Options:
//!   --model-dir <PATH>   Model directory (default: auto-download)
//!   --precision <PREC>   int4 (default) | fp32
//!   --provider <EP>      cpu (default) | cuda | tensorrt | coreml
//!   --iterations <N>     Measured runs per sample (default: 5)
//!   --warmup <N>         Warmup runs before measurement (default: 1)
//!   --language <LANG>    Language code (default: en)
//!   --instruction <TEXT> Voice instruction
//!   --csv                Emit CSV rows instead of summary table
//!
//! Examples:
//!   cargo run --release --example bench_rtf --features qwen3-tts
//!   cargo run --release --example bench_rtf --features "qwen3-tts,cuda" -- --provider cuda
//!   cargo run --release --example bench_rtf --features "qwen3-tts,cuda" -- --provider cuda --csv > results.csv

use std::path::PathBuf;
use std::time::Instant;

use wavekat_tts::backends::qwen3_tts::{ExecutionProvider, ModelConfig, ModelPrecision, Qwen3Tts};
use wavekat_tts::{SynthesizeRequest, TtsBackend};

struct Sample {
    label: &'static str,
    text: &'static str,
}

const SAMPLES: &[Sample] = &[
    Sample {
        label: "short",
        text: "Hello, world! This is a quick test of the speech synthesis system.",
    },
    Sample {
        label: "medium",
        text: "The quick brown fox jumps over the lazy dog. \
               Speech synthesis has improved dramatically over the past few years. \
               Modern neural TTS systems can produce highly natural-sounding speech \
               that is nearly indistinguishable from human voice recordings.",
    },
    Sample {
        label: "long",
        text: "Artificial intelligence is transforming the way we interact with computers. \
               Voice interfaces powered by text-to-speech technology are now commonplace \
               in smartphones, smart speakers, and automotive systems. \
               The latest generation of neural TTS models uses transformer architectures \
               trained on thousands of hours of human speech to capture the subtle nuances \
               of natural spoken language, including prosody, rhythm, and intonation. \
               These models can generate high-quality audio at sample rates of twenty-four \
               kilohertz or higher, enabling crisp and clear voice output across a wide range \
               of applications from accessibility tools to interactive voice assistants.",
    },
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut model_dir: Option<PathBuf> = None;
    let mut precision = ModelPrecision::Int4;
    let mut provider = ExecutionProvider::Cpu;
    let mut iterations: usize = 5;
    let mut warmup: usize = 1;
    let mut language = "en".to_string();
    let mut instruction = "Speak naturally and clearly.".to_string();
    let mut csv_mode = false;

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
            "--provider" => {
                i += 1;
                provider = match args[i].as_str() {
                    "cpu" => ExecutionProvider::Cpu,
                    "cuda" => ExecutionProvider::Cuda,
                    "tensorrt" => ExecutionProvider::TensorRt,
                    "coreml" => ExecutionProvider::CoreMl,
                    other => {
                        eprintln!(
                            "error: unknown provider \"{other}\", \
                             expected cpu, cuda, tensorrt, or coreml"
                        );
                        std::process::exit(1);
                    }
                };
            }
            "--iterations" => {
                i += 1;
                iterations = args[i]
                    .parse()
                    .expect("--iterations must be a positive integer");
            }
            "--warmup" => {
                i += 1;
                warmup = args[i]
                    .parse()
                    .expect("--warmup must be a non-negative integer");
            }
            "--language" => {
                i += 1;
                language = args[i].clone();
            }
            "--instruction" => {
                i += 1;
                instruction = args[i].clone();
            }
            "--csv" => csv_mode = true,
            "--help" | "-h" => {
                print_usage();
                return;
            }
            other => {
                eprintln!("error: unknown argument \"{other}\"  (use --help for usage)");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    eprintln!(
        "Loading model (precision={:?}, provider={:?}) ...",
        precision, provider
    );
    let mut config = ModelConfig::default()
        .with_precision(precision)
        .with_execution_provider(provider);
    if let Some(dir) = model_dir {
        config = config.with_dir(dir);
    }
    let tts = Qwen3Tts::from_config(config).expect("failed to load model");
    eprintln!("Model loaded.\n");

    if csv_mode {
        println!("sample,chars,iteration,synth_secs,audio_secs,rtf");
    } else {
        eprintln!(
            "Benchmark: {} warmup + {} measured iterations per sample\n",
            warmup, iterations
        );
    }

    let mut summary: Vec<(&'static str, usize, Vec<RunResult>)> = Vec::new();

    for sample in SAMPLES {
        let request = SynthesizeRequest::new(sample.text)
            .with_language(&language)
            .with_instruction(&instruction);

        // Warmup runs (not counted).
        for w in 0..warmup {
            eprint!(
                "  [{:6}] warmup {}/{} ...\r",
                sample.label,
                w + 1,
                warmup
            );
            tts.synthesize(&request).expect("warmup synthesis failed");
        }
        if warmup > 0 {
            eprintln!();
        }

        // Measured runs.
        let mut runs = Vec::with_capacity(iterations);
        for it in 0..iterations {
            let t0 = Instant::now();
            let audio = tts.synthesize(&request).expect("synthesis failed");
            let synth_secs = t0.elapsed().as_secs_f64();
            let audio_secs = audio.duration_secs();
            let rtf = synth_secs / audio_secs;

            eprintln!(
                "  [{:6}] iter {}/{}: synth={:.3}s  audio={:.2}s  RTF={:.3}",
                sample.label,
                it + 1,
                iterations,
                synth_secs,
                audio_secs,
                rtf,
            );

            if csv_mode {
                println!(
                    "{},{},{},{:.6},{:.6},{:.6}",
                    sample.label,
                    sample.text.len(),
                    it + 1,
                    synth_secs,
                    audio_secs,
                    rtf,
                );
            }

            runs.push(RunResult {
                synth_secs,
                audio_secs,
                rtf,
            });
        }

        summary.push((sample.label, sample.text.len(), runs));
        eprintln!();
    }

    if !csv_mode {
        print_table(&summary);
    }
}

struct RunResult {
    synth_secs: f64,
    audio_secs: f64,
    rtf: f64,
}

struct Stats {
    mean: f64,
    std: f64,
    min: f64,
    p50: f64,
    p95: f64,
    max: f64,
}

fn stats(values: &[f64]) -> Stats {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let pct = |p: f64| -> f64 {
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    };

    Stats {
        mean,
        std,
        min: sorted[0],
        p50: pct(50.0),
        p95: pct(95.0),
        max: *sorted.last().unwrap(),
    }
}

fn print_table(summary: &[(&'static str, usize, Vec<RunResult>)]) {
    let w = 82;
    println!("\n{}", "=".repeat(w));
    println!("  Qwen3-TTS RTF Benchmark");
    println!("{}", "=".repeat(w));
    println!(
        "{:<8}  {:>5}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>8}  {:>8}",
        "sample", "chars", "rtf_mean", "rtf_std", "rtf_min", "rtf_p50", "rtf_p95", "rtf_max",
        "audio_s", "synth_s"
    );
    println!("{}", "-".repeat(w));

    for (label, chars, runs) in summary {
        let rtf_vals: Vec<f64> = runs.iter().map(|r| r.rtf).collect();
        let audio_vals: Vec<f64> = runs.iter().map(|r| r.audio_secs).collect();
        let synth_vals: Vec<f64> = runs.iter().map(|r| r.synth_secs).collect();

        let rtf = stats(&rtf_vals);
        let audio_mean = audio_vals.iter().sum::<f64>() / audio_vals.len() as f64;
        let synth_mean = synth_vals.iter().sum::<f64>() / synth_vals.len() as f64;

        println!(
            "{:<8}  {:>5}  {:>7.3}  {:>7.3}  {:>7.3}  {:>7.3}  {:>7.3}  {:>7.3}  {:>8.2}  {:>8.2}",
            label,
            chars,
            rtf.mean,
            rtf.std,
            rtf.min,
            rtf.p50,
            rtf.p95,
            rtf.max,
            audio_mean,
            synth_mean,
        );
    }

    println!("{}", "=".repeat(w));
    println!("RTF < 1.0 = faster-than-real-time.  synth_s / audio_s = RTF.");
}

fn print_usage() {
    eprintln!(
        "bench_rtf — RTF benchmark for Qwen3-TTS

Usage:
  cargo run --release --example bench_rtf --features qwen3-tts [-- OPTIONS]

Options:
  --model-dir <PATH>   Model directory (default: auto-download to HF cache)
  --precision <PREC>   int4 (default) | fp32
  --provider <EP>      cpu (default) | cuda | tensorrt | coreml
  --iterations <N>     Measured runs per sample (default: 5)
  --warmup <N>         Warmup runs before measurement (default: 1)
  --language <LANG>    Language code (default: en)
  --instruction <TEXT> Voice instruction (default: \"Speak naturally and clearly.\")
  --csv                Emit CSV rows to stdout instead of summary table

Examples:
  # CPU benchmark
  cargo run --release --example bench_rtf --features qwen3-tts

  # CUDA benchmark (T4)
  cargo run --release --example bench_rtf --features \"qwen3-tts,cuda\" -- --provider cuda

  # Save CSV for further analysis
  cargo run --release --example bench_rtf --features \"qwen3-tts,cuda\" \\
    -- --provider cuda --csv > results.csv"
    );
}
