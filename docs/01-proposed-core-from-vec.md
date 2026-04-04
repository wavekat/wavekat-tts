# Proposed addition to wavekat-core

## Problem

`AudioFrame` currently only supports construction from borrowed data via `new()`.
TTS backends (and future ASR backends) **produce** owned audio data. The current
workaround requires an unnecessary clone:

```rust
// Current: borrows then clones — O(n) wasted allocation
let samples: Vec<f32> = tts_engine.generate();
let frame = AudioFrame::new(samples.as_slice(), 24000).into_owned();
//                          ^^^^^^^^^ Cow::Borrowed
//                                                     ^^^^^^^^^^^ clones into Cow::Owned
```

## Proposed change

Add a single constructor method to `AudioFrame`:

```rust
// In audio.rs, add to impl AudioFrame<'a>:

impl AudioFrame<'static> {
    /// Construct an owned frame directly from a Vec.
    ///
    /// Zero-copy — wraps as `Cow::Owned` without cloning.
    /// Intended for audio producers (TTS, ASR) that generate owned data.
    ///
    /// # Example
    ///
    /// ```
    /// use wavekat_core::AudioFrame;
    ///
    /// let samples = vec![0.5f32, -0.5, 0.3];
    /// let frame = AudioFrame::from_vec(samples, 24000);
    /// assert_eq!(frame.sample_rate(), 24000);
    /// assert_eq!(frame.len(), 3);
    /// ```
    pub fn from_vec(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples: Cow::Owned(samples),
            sample_rate,
        }
    }
}
```

## Impact

- **Zero breaking changes** — purely additive
- **Zero new dependencies**
- Enables `wavekat-tts` (and future `wavekat-asr`) to return `AudioFrame<'static>`
  without the borrow-then-clone overhead
- Completes the symmetry: `new()` for consumers (borrow), `from_vec()` for producers (own)

## Test

```rust
#[test]
fn from_vec_is_zero_copy() {
    let samples = vec![0.5f32, -0.5];
    let ptr = samples.as_ptr();
    let frame = AudioFrame::from_vec(samples, 24000);
    // Cow::Owned — the pointer should be the same (no reallocation)
    assert_eq!(frame.samples().as_ptr(), ptr);
    assert_eq!(frame.sample_rate(), 24000);
}
```
