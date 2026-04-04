# Qwen3-TTS Backend Implementation Plan

## Overview

Implement `Qwen3Tts` as the first TTS backend in wavekat-tts, using ONNX Runtime
(`ort` crate) to run the Qwen3-TTS-12Hz-0.6B model. The backend implements both
`TtsBackend` (batch) and `StreamingTtsBackend` (chunked vocoder decode).

## Why 0.6B-12Hz

Qwen3-TTS ships 4 variants (2 sizes × 2 codec rates):

| | 0.6B | 1.7B |
|---|---|---|
| **12 Hz** | Qwen3-TTS-12Hz-0.6B | Qwen3-TTS-12Hz-1.7B |
| **25 Hz** | Qwen3-TTS-25Hz-0.6B | Qwen3-TTS-25Hz-1.7B |

We target **0.6B-12Hz** because:

- **Pre-exported ONNX models exist** — `elbruno/Qwen3-TTS-12Hz-0.6B-Base-ONNX`
  on HuggingFace. Other variants would require running the export scripts ourselves.
- **Smaller footprint** — ~3.8 GB total ONNX vs ~7 GB for 1.7B.
- **Fewer decode steps** — 12 Hz = half the autoregressive steps of 25 Hz for the
  same audio duration, directly improving latency.

## Model architecture

Qwen3-TTS is a discrete multi-codebook language model. Inference requires
**4 components** chained in sequence:

```
Text ──► BPE Tokenizer ──► Talker LM (prefill + decode) ──► Code Predictor (×15) ──► Vocoder ──► 24 kHz PCM
```

| Component         | ONNX file              | Size (0.6B) | Notes                                      |
|-------------------|------------------------|-------------|---------------------------------------------|
| Talker prefill    | `talker_prefill.onnx`  | ~1.7 GB     | Processes full input sequence                |
| Talker decode     | `talker_decode.onnx`   | ~1.7 GB     | Autoregressive single-step decode            |
| Code predictor    | `code_predictor.onnx`  | ~440 MB     | Generates codebook groups 1-15 per step      |
| Vocoder           | `vocoder.onnx`         | ~2.7 MB     | Causal ConvNet, 1920× upsample to 24 kHz    |

Plus:
- **Embedding tables** (~1.4 GB) — text + codec embeddings as `.npy` files
- **BPE tokenizer** — `vocab.json` + `merges.txt` (Qwen3 standard, 151,936 tokens)

Pre-exported ONNX models: `elbruno/Qwen3-TTS-12Hz-0.6B-Base-ONNX` on HuggingFace.

## Key model parameters (0.6B)

| Parameter       | Value  |
|-----------------|--------|
| Hidden dim      | 1024   |
| Layers (Talker) | 28     |
| Attn heads      | 16     |
| KV heads        | 8      |
| Head dim        | 128    |
| FFN dim         | 3072   |
| Codec vocab     | 3072   |
| Text vocab      | 151936 |
| Output rate     | 24 kHz |
| Codec frame rate | 12.5 Hz |
| RVQ codebooks   | 16 (1 semantic + 15 acoustic) |

## Dependencies

```toml
# Cargo.toml additions (under [dependencies])
tokenizers = { version = "0.21", optional = true }  # HuggingFace BPE tokenizer
npyz = { version = "0.8", optional = true }          # Load .npy embedding files

# Update feature gate
qwen3-tts = ["dep:ort", "dep:ndarray", "dep:tokenizers", "dep:npyz"]
```

## File layout

```
src/backends/
├── mod.rs              # existing — feature-gated module declarations
├── onnx.rs             # NEW — shared ONNX session builder (from 02-execution-providers)
└── qwen3_tts/
    ├── mod.rs          # Qwen3Tts struct, TtsBackend + StreamingTtsBackend impls
    ├── model.rs        # ONNX session management, inference loop
    ├── tokenizer.rs    # BPE tokenization + prompt construction
    └── sampler.rs      # Top-p / temperature sampling for Talker + Code Predictor
```

## Implementation phases

### Phase 1: Scaffold + batch synthesis

Goal: `Qwen3Tts::new(path)` → `tts.synthesize(&request)` → `AudioFrame<'static>`.

#### 1a. Model directory convention

Expect a flat directory with all model files:

```
models/qwen3-tts-0.6b/
├── speaker_encoder.onnx
├── talker_prefill.onnx
├── talker_decode.onnx
├── code_predictor.onnx
├── vocoder.onnx
├── text_embedding.npy        # (151936, 1024) f32
├── codec_embedding.npy       # (3072, 1024) f32
├── codec_lm_head_weight.npy  # (3072, 1024) f32
├── vocab.json
└── merges.txt
```

Constructor: `Qwen3Tts::new(model_dir: impl AsRef<Path>) -> Result<Self, TtsError>`

#### 1b. Tokenizer + prompt construction

Build the input sequence expected by the Talker LM:

```
<|text_start|> [BPE tokens...] <|text_end|>
<|lang_id|>                                     # e.g., 2050 for English, 2055 for Chinese
<|codec_start|>
```

Use the `tokenizers` crate to load `vocab.json` + `merges.txt` and encode text.

Special token IDs (from model config):
- `<|text_start|>` = 151646
- `<|text_end|>`   = 151647
- `<|codec_start|>`= 151668
- `<|codec_end|>`  = 151669
- Language IDs: English=2050, Chinese=2055, Japanese=2048, Korean=2052

#### 1c. Embedding lookup

Construct input embeddings by:
1. Looking up text token IDs in `text_embedding.npy`
2. Looking up codec token IDs in `codec_embedding.npy`
3. Concatenating into a `(1, T, 1024)` tensor

Load `.npy` files with `npyz` at construction time into `ndarray::Array2<f32>`.

#### 1d. Talker LM inference loop

```
1. Prefill: run talker_prefill.onnx with full input embeddings
   → logits (1, T, 3072), KV-cache for 28 layers

2. Decode loop:
   a. Sample token from logits (top-p, temperature)
   b. If token == <|codec_end|> → stop
   c. Look up token embedding in codec_embedding
   d. Run talker_decode.onnx with single-step embedding + KV-cache
   → next logits, updated KV-cache
   e. Store token as codebook-group-0 for this timestep
```

KV-cache shape per layer: `(1, 8, T, 128)` — grows by 1 each decode step.

#### 1e. Code Predictor inference

For each Talker timestep, run the code predictor 15 times to fill codebook
groups 1-15:

```
For each timestep t:
  hidden_state = talker_hidden_states[t]  # (1, 1, 1024)
  For group g in 1..=15:
    input = [hidden_state, codec_embed(prev_group_token)]  # (1, 2, 1024)
    logits = code_predictor(input)                          # (1, 1, 2048)
    token = sample(logits)
    codes[g][t] = token
```

Code predictor has its own small KV-cache (5 layers), reset per timestep.

#### 1f. Vocoder

Feed the full `(1, 16, T)` code matrix to the vocoder:

```rust
let codes: Array3<i64> = /* shape (1, 16, num_steps) */;
let pcm: Array3<f32> = vocoder.run(codes)?;  // (1, 1, num_steps * 1920)
let samples: Vec<f32> = pcm.into_raw_vec();
let frame = AudioFrame::new(&samples, 24000).into_owned();
```

#### 1g. `voices()` implementation

Return a single default voice for the base model (no voice cloning in phase 1):

```rust
fn voices(&self) -> Result<Vec<VoiceInfo>, TtsError> {
    Ok(vec![VoiceInfo {
        id: "default".into(),
        name: "Qwen3-TTS Default".into(),
        languages: vec!["en".into(), "zh".into(), "ja".into(), "ko".into()],
        gender: None,
    }])
}
```

### Phase 2: Streaming synthesis

Implement `StreamingTtsBackend::stream()` by running the vocoder incrementally:

- Run Talker + Code Predictor as in batch mode, collecting codes
- Feed codes to the vocoder in chunks (e.g., every N timesteps)
- Each chunk produces `N * 1920` samples → one `AudioFrame`
- The vocoder is causal, so chunked decode is valid

Chunk size trade-off: smaller = lower latency, larger = less overhead.
Start with 10 timesteps per chunk (~0.8s audio per chunk).

### Phase 3: Execution providers

Apply the design from `02-execution-providers.md`:
- `BackendConfig` struct with `ExecutionProvider` enum
- `Qwen3Tts::with_config(model_dir, config)` constructor
- CoreML for macOS, CUDA/TensorRT for NVIDIA

## Sampling strategy

The Talker LM and Code Predictor both need token sampling:

| Parameter   | Talker LM | Code Predictor |
|-------------|-----------|----------------|
| Temperature | 0.7       | 0.2            |
| Top-p       | 0.8       | 0.5            |
| Repetition penalty | 1.0 | 1.0          |

These defaults match the reference implementation. Make them configurable later
if needed.

## Testing strategy

### Unit tests (no model files needed)
- Tokenizer: BPE encode/decode round-trip
- Prompt construction: verify special token sequence
- Sampler: deterministic with fixed seed

### Integration tests (`test-qwen3` make target, needs model files)
- Batch synthesis: generate audio, check sample rate = 24000, duration > 0
- Write to WAV with `hound` crate for manual listening
- Round-trip: synthesize → check AudioFrame properties

Model files are large (~3.8 GB total). Tests should skip gracefully if the
model directory doesn't exist:

```rust
#[test]
fn synthesize_hello() {
    let model_dir = std::env::var("QWEN3_TTS_MODEL_DIR")
        .unwrap_or_else(|_| "models/qwen3-tts-0.6b".into());
    if !Path::new(&model_dir).exists() {
        eprintln!("Skipping: model dir not found at {model_dir}");
        return;
    }
    // ...
}
```

## Reference implementations

| Project | Language | Notes |
|---------|----------|-------|
| [elbruno/ElBruno.QwenTTS](https://github.com/elbruno/ElBruno.QwenTTS) | C# / .NET | Complete ONNX pipeline, best reference for tensor I/O |
| [zukky/Qwen3-TTS-ONNX-DLL](https://huggingface.co/zukky/Qwen3-TTS-ONNX-DLL) | Rust DLL | Rust preprocessing + ONNX inference |
| [TrevorS/qwen3-tts-rs](https://github.com/TrevorS/qwen3-tts-rs) | Rust (Candle) | Pure Rust, no ONNX — useful for understanding the math |
| [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) | C++ (GGML) | Low-level implementation, useful for tensor shapes |

## Open questions

1. **`from_vec` on wavekat-core** — Should we land `01-proposed-core-from-vec`
   before starting, or use the `new().into_owned()` workaround for now?
   → Workaround is fine for phase 1. Upgrade later.

2. **Embedding loading** — `.npy` files are simple but large. Consider memory-mapped
   loading for the 1.4 GB embedding tables to reduce startup RSS.

3. **KV-cache memory** — 28 layers × 8 KV heads × 128 dim × 2 (K+V) × 4 bytes
   = ~229 KB per timestep. For a 10-second utterance (~125 steps), that's ~28 MB.
   Manageable, but worth profiling.

4. **Thread safety** — `ort::Session` is `Send + Sync`. The `Qwen3Tts` struct holds
   5 sessions + embedding tables. All immutable after construction → naturally
   thread-safe. KV-cache is per-inference, not shared.
