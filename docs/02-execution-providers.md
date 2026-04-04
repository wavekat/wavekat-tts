# Execution Provider Design

## Context

All TTS backends (Qwen3-TTS, CosyVoice) use ONNX Runtime via the `ort` crate.
By default models run on CPU. Users can opt into hardware acceleration (CoreML
on macOS, CUDA on NVIDIA) at runtime via a config struct, with Cargo features
gating compilation of each provider.

## Cargo features

```toml
# Enable a backend (pulls in ort + ndarray)
qwen3-tts = ["dep:ort", "dep:ndarray"]
cosyvoice = ["dep:ort", "dep:ndarray"]

# Execution providers (composable with any backend)
coreml   = ["ort?/coreml"]      # macOS Metal / ANE
cuda     = ["ort?/cuda"]        # NVIDIA GPU
tensorrt = ["ort?/tensorrt"]    # NVIDIA TensorRT
```

Usage examples:

```toml
# CPU only (default)
wavekat-tts = { version = "0.0.1", features = ["qwen3-tts"] }

# CoreML on macOS
wavekat-tts = { version = "0.0.1", features = ["qwen3-tts", "coreml"] }

# CUDA on Linux/Windows
wavekat-tts = { version = "0.0.1", features = ["qwen3-tts", "cuda"] }
```

## Runtime API

```rust
use wavekat_tts::{BackendConfig, ExecutionProvider};

// CPU (default)
let config = BackendConfig::default();

// CoreML with 4 threads
let config = BackendConfig::default()
    .with_provider(ExecutionProvider::CoreMl)
    .with_intra_threads(4);

// Pass to backend constructor
let tts = Qwen3Tts::with_config("model.onnx", config)?;
```

### ExecutionProvider enum

| Variant    | Feature required | Platform        |
|------------|-----------------|-----------------|
| `Cpu`      | (always)        | All             |
| `CoreMl`   | `coreml`        | macOS / iOS     |
| `Cuda`     | `cuda`          | NVIDIA GPU      |
| `TensorRt` | `tensorrt`      | NVIDIA TensorRT |

### BackendConfig struct

| Field                | Type                | Default |
|----------------------|---------------------|---------|
| `execution_provider` | `ExecutionProvider` | `Cpu`   |
| `intra_threads`      | `usize`             | `1`     |
| `inter_threads`      | `usize`             | `1`     |

## Implementation files

| File | Purpose |
|------|---------|
| `src/types.rs` | `ExecutionProvider` enum + `BackendConfig` struct |
| `src/backends/onnx.rs` | Shared session builder with provider selection |
| `src/backends/mod.rs` | Conditional `onnx` module gate |
| `Cargo.toml` | Feature flags for providers |
