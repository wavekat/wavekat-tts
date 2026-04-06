# Proposed addition to wavekat-core: WAV I/O

## Problem

Every crate in the WaveKat ecosystem that reads or writes WAV files (wavekat-vad,
wavekat-turn, wavekat-tts) independently reaches for `hound` and repeats the same
boilerplate. Any spec change (e.g. bits-per-sample, channel count) must be updated
in multiple places.

```rust
// Current: every crate does this manually
let spec = hound::WavSpec {
    channels: 1,
    sample_rate: audio.sample_rate(),
    bits_per_sample: 32,
    sample_format: hound::SampleFormat::Float,
};
let mut writer = hound::WavWriter::create(path, spec)?;
for &sample in audio.samples() {
    writer.write_sample(sample)?;
}
writer.finalize()?;
```

## Proposed change

Add a `hound` feature flag to wavekat-core that extends `AudioFrame` with two
methods:

```toml
# wavekat-core Cargo.toml
[features]
hound = ["dep:hound"]

[dependencies]
hound = { version = "3.5", optional = true }
```

```rust
// In audio.rs, behind #[cfg(feature = "hound")]:

impl AudioFrame<'_> {
    /// Write this frame to a WAV file at `path`.
    ///
    /// Always writes mono f32 PCM at the frame's native sample rate.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use wavekat_core::AudioFrame;
    ///
    /// let frame = AudioFrame::from_vec(vec![0.0f32; 16000], 16000);
    /// frame.write_wav("output.wav").unwrap();
    /// ```
    #[cfg(feature = "hound")]
    pub fn write_wav(&self, path: impl AsRef<std::path::Path>) -> Result<(), hound::Error> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;
        for &sample in self.samples() {
            writer.write_sample(sample)?;
        }
        writer.finalize()
    }
}

impl AudioFrame<'static> {
    /// Read a mono WAV file and return an owned `AudioFrame`.
    ///
    /// Accepts both f32 and i16 WAV files. i16 samples are normalised to
    /// `[-1.0, 1.0]` (divided by 32768).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use wavekat_core::AudioFrame;
    ///
    /// let frame = AudioFrame::from_wav("input.wav").unwrap();
    /// println!("{} Hz, {} samples", frame.sample_rate(), frame.len());
    /// ```
    #[cfg(feature = "hound")]
    pub fn from_wav(path: impl AsRef<std::path::Path>) -> Result<Self, hound::Error> {
        let mut reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                reader.samples::<f32>().map(|s| s.unwrap()).collect()
            }
            hound::SampleFormat::Int => {
                reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect()
            }
        };
        Ok(AudioFrame::from_vec(samples, sample_rate))
    }
}
```

## Impact

- **Zero breaking changes** — purely additive, opt-in via feature flag
- Consumers opt in: `wavekat-core = { version = "0.0.5", features = ["hound"] }`
- All examples and tests across wavekat-vad, wavekat-turn, wavekat-tts can drop
  their own `hound` boilerplate and use the canonical implementation
- One place to maintain the WAV spec (mono, f32, native sample rate)

## Tests

```rust
#[cfg(feature = "hound")]
#[test]
fn wav_round_trip() {
    let original = AudioFrame::from_vec(vec![0.5f32, -0.5, 0.0, 1.0], 16000);
    let path = std::env::temp_dir().join("wavekat_test.wav");
    original.write_wav(&path).unwrap();
    let loaded = AudioFrame::from_wav(&path).unwrap();
    assert_eq!(loaded.sample_rate(), 16000);
    for (a, b) in original.samples().iter().zip(loaded.samples()) {
        assert!((a - b).abs() < 1e-6, "sample mismatch: {a} vs {b}");
    }
}
```
