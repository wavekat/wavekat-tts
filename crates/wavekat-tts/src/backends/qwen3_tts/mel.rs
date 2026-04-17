//! Mel-spectrogram computation matching the Qwen3-TTS reference (librosa).
//!
//! Parameters: sr=24000, n_fft=1024, hop=256, win=1024, n_mels=128,
//! fmin=0, fmax=12000, center=False, power=2.0, log on top.

use ndarray::Array2;
use realfft::num_complex::Complex;
use realfft::RealFftPlanner;

const SR: f32 = 24000.0;
const N_FFT: usize = 1024;
const HOP: usize = 256;
const WIN: usize = 1024;
const N_MELS: usize = 128;
const FMIN: f32 = 0.0;
const FMAX: f32 = 12000.0;
const N_BINS: usize = N_FFT / 2 + 1; // 513

/// Pre-computed mel filterbank and Hann window for repeated use.
pub struct MelSpectrogram {
    window: Vec<f32>,
    filterbank: Array2<f32>, // (N_MELS, N_BINS)
}

impl MelSpectrogram {
    pub fn new() -> Self {
        Self {
            window: hann_window(WIN),
            filterbank: mel_filterbank(N_MELS, N_FFT, SR, FMIN, FMAX),
        }
    }

    /// Compute log-mel spectrogram. Returns `(T_mel, 128)` f32.
    pub fn compute(&self, audio: &[f32]) -> Array2<f32> {
        let n_frames = if audio.len() >= WIN {
            1 + (audio.len() - WIN) / HOP
        } else {
            0
        };

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        let mut mel = Array2::<f32>::zeros((n_frames, N_MELS));
        let mut frame_buf = vec![0.0f32; N_FFT];
        let mut spectrum = vec![Complex::new(0.0f32, 0.0f32); N_BINS];

        for i in 0..n_frames {
            let start = i * HOP;

            // Window the frame
            for j in 0..WIN {
                frame_buf[j] = audio[start + j] * self.window[j];
            }
            // FFT
            fft.process(&mut frame_buf, &mut spectrum).unwrap();

            // Power spectrum → mel filterbank → log
            for m in 0..N_MELS {
                let mut sum = 0.0f32;
                for (k, s) in spectrum.iter().enumerate() {
                    let power = s.re * s.re + s.im * s.im;
                    sum += self.filterbank[[m, k]] * power;
                }
                mel[[i, m]] = sum.max(1e-5).ln();
            }
        }

        mel
    }
}

/// Periodic Hann window (matches librosa's STFT default).
fn hann_window(length: usize) -> Vec<f32> {
    let n = length as f32;
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n).cos()))
        .collect()
}

// ---------------------------------------------------------------------------
// Slaney mel scale (matches librosa default, htk=False)
// ---------------------------------------------------------------------------

const F_SP: f32 = 200.0 / 3.0; // 66.667 Hz per mel (linear region)
const MIN_LOG_HZ: f32 = 1000.0;
const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP; // 15.0

/// log(6.4) / 27 ≈ 0.06875
fn logstep() -> f32 {
    6.4f32.ln() / 27.0
}

fn hz_to_mel(freq: f32) -> f32 {
    if freq < MIN_LOG_HZ {
        freq / F_SP
    } else {
        MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() / logstep()
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * (logstep() * (mel - MIN_LOG_MEL)).exp()
    }
}

/// Build a `(n_mels, n_bins)` triangular mel filterbank (no normalization).
fn mel_filterbank(n_mels: usize, n_fft: usize, sr: f32, fmin: f32, fmax: f32) -> Array2<f32> {
    let n_bins = n_fft / 2 + 1;
    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);

    // n_mels + 2 mel-spaced center frequencies
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_to_hz(min_mel + (max_mel - min_mel) * i as f32 / (n_mels + 1) as f32))
        .collect();

    // FFT bin center frequencies
    let fft_freqs: Vec<f32> = (0..n_bins).map(|k| k as f32 * sr / n_fft as f32).collect();

    let mut fb = Array2::<f32>::zeros((n_mels, n_bins));

    for m in 0..n_mels {
        let f_left = mel_points[m];
        let f_center = mel_points[m + 1];
        let f_right = mel_points[m + 2];

        let d_left = f_center - f_left;
        let d_right = f_right - f_center;

        for k in 0..n_bins {
            let f = fft_freqs[k];
            if f >= f_left && f <= f_center && d_left > 0.0 {
                fb[[m, k]] = (f - f_left) / d_left;
            } else if f > f_center && f <= f_right && d_right > 0.0 {
                fb[[m, k]] = (f_right - f) / d_right;
            }
        }
    }

    fb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_scale_roundtrip() {
        for &freq in &[0.0, 500.0, 1000.0, 4000.0, 12000.0] {
            let m = hz_to_mel(freq);
            let f = mel_to_hz(m);
            assert!(
                (f - freq).abs() < 0.01,
                "roundtrip failed for {freq}: got {f}"
            );
        }
    }

    #[test]
    fn filterbank_shape() {
        let fb = mel_filterbank(128, 1024, 24000.0, 0.0, 12000.0);
        assert_eq!(fb.shape(), &[128, 513]);
    }

    #[test]
    fn filterbank_non_negative() {
        let fb = mel_filterbank(128, 1024, 24000.0, 0.0, 12000.0);
        assert!(fb.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn mel_output_shape() {
        let mel = MelSpectrogram::new();
        // 1 second of audio at 24kHz
        let audio = vec![0.0f32; 24000];
        let result = mel.compute(&audio);
        // (24000 - 1024) / 256 + 1 = 89 + 1 = 90 frames
        let expected_frames = 1 + (24000 - WIN) / HOP;
        assert_eq!(result.shape(), &[expected_frames, 128]);
    }

    #[test]
    fn hann_window_properties() {
        let w = hann_window(1024);
        assert_eq!(w.len(), 1024);
        assert!((w[0] - 0.0).abs() < 1e-6); // starts near zero
        assert!(w[512] > 0.99); // peak near middle
    }
}
