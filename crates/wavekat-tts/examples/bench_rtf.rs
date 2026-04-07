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
//!   --backend <NAME>     Backend identifier written to CSV (default: qwen3-tts)
//!   --precision <PREC>   int4 (default) | fp32
//!   --provider <EP>      cpu (default) | cuda | tensorrt | coreml
//!   --hardware <NAME>    Hardware label written to CSV, e.g. t4, a10g (default: unknown)
//!   --iterations <N>     Measured runs per sample (default: 5)
//!   --warmup <N>         Warmup runs before measurement (default: 1)
//!   --language <LANG>    Language code (default: en)
//!   --instruction <TEXT> Voice instruction
//!   --csv                Emit CSV rows instead of summary table
//!
//! Examples:
//!   cargo run --release --example bench_rtf --features qwen3-tts
//!   cargo run --release --example bench_rtf --features "qwen3-tts,cuda" -- --provider cuda --hardware t4
//!   cargo run --release --example bench_rtf --features "qwen3-tts,cuda" -- --provider cuda --hardware t4 --csv > results.csv

use std::path::PathBuf;
use std::time::Instant;

use wavekat_tts::backends::qwen3_tts::{ExecutionProvider, ModelConfig, ModelPrecision, Qwen3Tts};
use wavekat_tts::{SynthesizeRequest, TtsBackend};

struct Sample {
    label: &'static str,
    /// Pool of texts rotated across iterations so each run uses different content.
    texts: &'static [&'static str],
}

const SAMPLES: &[Sample] = &[
    Sample {
        label: "short",
        texts: &[
            "Hello, world! This is a quick test of the speech synthesis system.",
            "The weather today is sunny with a high of twenty-three degrees Celsius.",
            "Your package has been delivered to the front door of your building.",
            "Please turn left in five hundred meters, then continue for two miles.",
            "Your appointment is confirmed for Tuesday at three fifteen in the afternoon.",
            "An update is available for your device. Would you like to install it now?",
        ],
    },
    Sample {
        label: "medium",
        texts: &[
            "The quick brown fox jumps over the lazy dog. \
             Speech synthesis has improved dramatically over the past few years. \
             Modern neural TTS systems can produce highly natural-sounding speech \
             that is nearly indistinguishable from human voice recordings.",
            "Scientists have confirmed that regular physical activity reduces the risk \
             of chronic conditions including heart disease and diabetes. \
             A thirty-minute walk each day can improve cardiovascular health \
             and support mental well-being across all age groups.",
            "The global demand for renewable energy is accelerating as countries commit \
             to reducing carbon emissions. Solar and wind installations have grown rapidly \
             in recent years, making clean electricity more affordable than ever \
             for homes and businesses worldwide.",
            "Advances in robotics are transforming manufacturing, logistics, and healthcare. \
             Modern robots can perform delicate surgical procedures, navigate warehouse \
             environments, and assist elderly patients with daily tasks, \
             often working alongside human colleagues.",
            "The history of the internet stretches back to the nineteen sixties, \
             when researchers first connected computers across university campuses. \
             What began as a small academic network has grown into a global infrastructure \
             connecting billions of people and devices.",
        ],
    },
    Sample {
        label: "long",
        texts: &[
            "Artificial intelligence is transforming the way we interact with computers. \
             Voice interfaces powered by text-to-speech technology are now commonplace \
             in smartphones, smart speakers, and automotive systems. \
             The latest generation of neural TTS models uses transformer architectures \
             trained on thousands of hours of human speech to capture the subtle nuances \
             of natural spoken language, including prosody, rhythm, and intonation. \
             These models can generate high-quality audio at sample rates of twenty-four \
             kilohertz or higher, enabling crisp and clear voice output across a wide range \
             of applications from accessibility tools to interactive voice assistants.",
            "Ocean exploration remains one of the most challenging frontiers in modern science. \
             More than eighty percent of the world's oceans have never been mapped, explored, \
             or studied in detail, leaving vast regions of our planet largely unknown. \
             Deep sea research vessels and remotely operated underwater vehicles are slowly \
             changing this picture, discovering new ecosystems, geological formations, \
             and species previously unknown to science. These discoveries have important \
             implications for medicine, materials science, and our understanding of how \
             life evolved on Earth and potentially on other worlds in the solar system.",
            "Urban transportation networks are undergoing a fundamental transformation \
             driven by electrification, automation, and new mobility services. \
             Electric buses, trams, and bicycles are replacing fossil-fuel vehicles \
             in many cities, reducing air pollution and greenhouse gas emissions. \
             Ride-sharing platforms and micro-mobility services are changing how people \
             think about car ownership, particularly among younger generations who prefer \
             flexible access over the fixed costs of owning a vehicle. \
             City planners are reimagining streets to prioritize pedestrians and cyclists, \
             creating more livable environments while reducing traffic congestion.",
            "The development of quantum computing promises to solve problems that are \
             intractable for classical computers, including simulating molecular interactions \
             for drug discovery and breaking certain cryptographic algorithms. \
             Current quantum processors must operate near absolute zero to maintain \
             the fragile quantum states that give them their computational power. \
             Researchers around the world are racing to build systems with enough \
             stable qubits to demonstrate a clear advantage over classical hardware \
             in real-world applications, a milestone often referred to as quantum advantage.",
            "Throughout history, libraries have served as the guardians of human knowledge, \
             preserving texts and manuscripts that might otherwise have been lost to time. \
             The transition from physical collections to digital archives has dramatically \
             expanded access to information, allowing anyone with an internet connection \
             to read documents that were once available only to scholars. \
             Digitization projects at major institutions have made millions of books, \
             maps, and historical records freely available online, democratizing access \
             to cultural heritage and enabling new forms of research across disciplines.",
        ],
    },
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut model_dir: Option<PathBuf> = None;
    let mut backend = "qwen3-tts".to_string();
    let mut precision = ModelPrecision::Int4;
    let mut provider = ExecutionProvider::Cpu;
    let mut hardware = "unknown".to_string();
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
            "--backend" => {
                i += 1;
                backend = args[i].clone();
            }
            "--hardware" => {
                i += 1;
                hardware = args[i].clone();
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

    let precision_str = match precision {
        ModelPrecision::Int4 => "int4",
        ModelPrecision::Fp32 => "fp32",
    };
    let provider_str = match provider {
        ExecutionProvider::Cpu => "cpu",
        ExecutionProvider::Cuda => "cuda",
        ExecutionProvider::TensorRt => "tensorrt",
        ExecutionProvider::CoreMl => "coreml",
    };
    let date = today_iso();

    if csv_mode {
        println!("backend,precision,provider,hardware,date,sample,chars,iteration,synth_secs,audio_secs,rtf");
    } else {
        eprintln!(
            "Benchmark: {} warmup + {} measured iterations per sample\n",
            warmup, iterations
        );
    }

    let mut summary: Vec<(&'static str, usize, Vec<RunResult>)> = Vec::new();

    for sample in SAMPLES {
        // Warmup runs (not counted) — rotate through the text pool.
        for w in 0..warmup {
            let text = sample.texts[w % sample.texts.len()];
            let request = SynthesizeRequest::new(text)
                .with_language(&language)
                .with_instruction(&instruction);
            eprint!("  [{:6}] warmup {}/{} ...\r", sample.label, w + 1, warmup);
            tts.synthesize(&request).expect("warmup synthesis failed");
        }
        if warmup > 0 {
            eprintln!();
        }

        // Measured runs — each iteration uses a different text from the pool.
        let mut runs = Vec::with_capacity(iterations);
        let mut total_chars: usize = 0;
        for it in 0..iterations {
            let text = sample.texts[it % sample.texts.len()];
            let request = SynthesizeRequest::new(text)
                .with_language(&language)
                .with_instruction(&instruction);

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
                    "{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6}",
                    backend,
                    precision_str,
                    provider_str,
                    hardware,
                    date,
                    sample.label,
                    text.len(),
                    it + 1,
                    synth_secs,
                    audio_secs,
                    rtf,
                );
            }

            total_chars += text.len();
            runs.push(RunResult {
                synth_secs,
                audio_secs,
                rtf,
            });
        }

        let avg_chars = if iterations > 0 {
            total_chars / iterations
        } else {
            0
        };
        summary.push((sample.label, avg_chars, runs));
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
        "sample",
        "chars",
        "rtf_mean",
        "rtf_std",
        "rtf_min",
        "rtf_p50",
        "rtf_p95",
        "rtf_max",
        "audio_s",
        "synth_s"
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
  --backend <NAME>     Backend label in CSV (default: qwen3-tts)
  --precision <PREC>   int4 (default) | fp32
  --provider <EP>      cpu (default) | cuda | tensorrt | coreml
  --hardware <NAME>    Hardware label in CSV, e.g. t4, a10g (default: unknown)
  --iterations <N>     Measured runs per sample (default: 5)
  --warmup <N>         Warmup runs before measurement (default: 1)
  --language <LANG>    Language code (default: en)
  --instruction <TEXT> Voice instruction (default: \"Speak naturally and clearly.\")
  --csv                Emit CSV rows to stdout instead of summary table

Examples:
  # CPU benchmark
  cargo run --release --example bench_rtf --features qwen3-tts

  # CUDA benchmark on a T4
  cargo run --release --example bench_rtf --features \"qwen3-tts,cuda\" \\
    -- --provider cuda --hardware t4

  # Save CSV for tracking and README auto-update
  cargo run --release --example bench_rtf --features \"qwen3-tts,cuda\" \\
    -- --provider cuda --hardware t4 --csv > bench/results/cuda-t4-int4.csv"
    );
}

/// Return today's date as YYYY-MM-DD (UTC) without any external dependency.
fn today_iso() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut days = secs / 86400;
    let mut year = 1970u32;
    loop {
        let in_year = if is_leap(year) { 366 } else { 365 };
        if days < in_year {
            break;
        }
        days -= in_year;
        year += 1;
    }
    let month_lengths = if is_leap(year) {
        [31u64, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31u64, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1u32;
    for &ml in &month_lengths {
        if days < ml {
            break;
        }
        days -= ml;
        month += 1;
    }
    format!("{:04}-{:02}-{:02}", year, month, days + 1)
}

fn is_leap(year: u32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}
