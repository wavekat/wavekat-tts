//! Synthesize text to a WAV file with Kokoro.
//!
//! Usage:
//!   cargo run --release --example kokoro --features kokoro -- [OPTIONS] TEXT
//!
//! Options:
//!   --voice <ID>       Voice id (default: af_heart)
//!   --speed <X>        Speed multiplier (default: 1.0)
//!   --variant <NAME>   fp16 (default), fp32, q8f16, quantized
//!   --output <PATH>    Output WAV (default: kokoro.wav)

use std::time::Instant;

use wavekat_tts::backends::kokoro::{Kokoro, KokoroConfig, ModelVariant};
use wavekat_tts::{SynthesizeRequest, TtsBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut voice = "af_heart".to_string();
    let mut speed = 1.0f32;
    let mut variant = ModelVariant::default();
    let mut output = "kokoro.wav".to_string();
    let mut text = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--voice" => voice = args.next().ok_or("--voice needs a value")?,
            "--speed" => speed = args.next().ok_or("--speed needs a value")?.parse()?,
            "--variant" => {
                let v = args.next().ok_or("--variant needs a value")?;
                variant = ModelVariant::parse(&v).ok_or("unknown variant")?;
            }
            "--output" => output = args.next().ok_or("--output needs a value")?,
            _ => text.push(a),
        }
    }
    let text = if text.is_empty() {
        "Hello! This is Kokoro, running locally.".to_string()
    } else {
        text.join(" ")
    };

    let t = Instant::now();
    let tts = Kokoro::from_config(KokoroConfig::default().with_variant(variant))?;
    tts.load_voice(&voice)?;
    let load = t.elapsed();

    let t = Instant::now();
    let audio = tts.synthesize(
        &SynthesizeRequest::new(&text)
            .with_voice(&voice)
            .with_speed(speed),
    )?;
    let synth = t.elapsed();
    audio.write_wav(&output)?;

    eprintln!(
        "load {:.2}s, synth {:.2}s for {:.2}s of audio (RTF {:.3}) -> {output}",
        load.as_secs_f64(),
        synth.as_secs_f64(),
        audio.duration_secs(),
        synth.as_secs_f64() / audio.duration_secs()
    );
    Ok(())
}
