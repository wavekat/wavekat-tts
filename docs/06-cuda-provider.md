# CUDA Execution Provider

## Status

**In progress** — the `cuda` Cargo feature is being wired up.
CPU inference already works; this adds NVIDIA GPU acceleration via ORT's CUDA EP.

## Goal

Enable NVIDIA GPU inference for Qwen3-TTS (and future ONNX-based backends)
by activating ORT's built-in CUDA execution provider. The `TtsBackend` trait
surface is unchanged — callers opt in via `BackendConfig`.

## Why CUDA over CPU

ORT's CUDA EP offloads transformer operations (matmul, attention, KV cache
reads/writes) to the GPU. For a 1.7B-parameter model the decode loop is the
bottleneck; GPU parallelism reduces per-step latency by ~15× on a T4.

CoreML is not viable here — see `05-mlx-backend.md` for why.

## Cargo feature

```toml
# crates/wavekat-tts/Cargo.toml
[features]
cuda     = ["ort?/cuda"]
tensorrt = ["ort?/tensorrt"]   # optional; higher throughput, longer compile
```

These compose with any ONNX-based backend feature:

```toml
# CPU only (default)
wavekat-tts = { version = "0.0.1", features = ["qwen3-tts"] }

# CUDA
wavekat-tts = { version = "0.0.1", features = ["qwen3-tts", "cuda"] }

# TensorRT (higher throughput, requires trtexec engine build)
wavekat-tts = { version = "0.0.1", features = ["qwen3-tts", "tensorrt"] }
```

The `ort` crate bundles its own CUDA libraries — no manual `LD_LIBRARY_PATH`
configuration is needed as long as the host has a compatible CUDA driver.

## Runtime API

```rust
use wavekat_tts::{BackendConfig, ExecutionProvider};

let config = BackendConfig::default()
    .with_provider(ExecutionProvider::Cuda);

let tts = Qwen3Tts::with_config("models/qwen3-tts-1.7b", config)?;
```

ORT falls back to CPU automatically if no compatible GPU is found at runtime.
Set `ORT_LOG_LEVEL=1` to confirm which EP is active:

```
[I:ort:session] [CUDAExecutionProvider] Created CUDA EP on device 0
```

### ExecutionProvider variants

| Variant    | Cargo feature | Requirement                        |
|------------|---------------|------------------------------------|
| `Cpu`      | (always)      | —                                  |
| `Cuda`     | `cuda`        | NVIDIA GPU, CUDA driver ≥ 11.8     |
| `TensorRt` | `tensorrt`    | CUDA + TensorRT 8+ installed       |

## Build

```bash
cargo build --release --features "qwen3-tts,cuda"
cargo run --release --example synthesize --features "qwen3-tts,cuda" -- \
  --text "Hello from GPU" --out output.wav
```

## Implementation

The only files that change:

| File | Change |
|------|--------|
| `crates/wavekat-tts/Cargo.toml` | Add `cuda` and `tensorrt` features |
| `src/backends/onnx.rs` | Match `ExecutionProvider::Cuda` → add CUDA EP to session builder |
| `src/types.rs` | `ExecutionProvider` enum already has `Cuda` and `TensorRt` variants |

`src/backends/onnx.rs` session builder (pseudocode):

```rust
let mut builder = Session::builder()?;
match config.execution_provider {
    ExecutionProvider::Cpu      => {}
    ExecutionProvider::CoreMl   => { builder = builder.with_execution_providers([CoreMLExecutionProvider::default()])?; }
    ExecutionProvider::Cuda     => { builder = builder.with_execution_providers([CUDAExecutionProvider::default()])?; }
    ExecutionProvider::TensorRt => { builder = builder.with_execution_providers([TensorRTExecutionProvider::default()])?; }
}
```

## Expected performance (NVIDIA T4, 1.7B model)

| Segment length | CPU     | CUDA (T4) | Speedup |
|----------------|---------|-----------|---------|
| 5 s audio      | ~120 s  | ~8 s      | ~15×    |
| 30 s audio     | ~700 s  | ~45 s     | ~15×    |

*Estimates based on ORT CUDA EP throughput for similarly-sized transformer
decode loops. Actual numbers depend on VRAM bandwidth and batch size.*

## Google Colab setup

Colab's free tier provides a T4 GPU (CUDA 12.x) — enough to run the 1.7B model
at real-time or faster with no local NVIDIA hardware required.

### 1. Enable GPU runtime

Runtime → Change runtime type → Hardware accelerator: **T4 GPU** → Save.

Verify:

```python
!nvidia-smi
```

### 2. Install Rust

```python
!curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
import os
os.environ["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + os.environ["PATH"]
!rustc --version
```

### 3. Clone and build

```python
!git clone https://github.com/wavekat/wavekat-tts.git
%cd wavekat-tts
!cargo build --release --features "qwen3-tts,cuda"
```

### 4. Model weights

The HF Hub downloader will fetch weights automatically on first run.
To persist across sessions, mount Google Drive and set `WAVEKAT_MODEL_DIR`:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
import os
os.environ["WAVEKAT_MODEL_DIR"] = "/content/drive/MyDrive/wavekat-models"
!cargo run --release --example synthesize --features "qwen3-tts,cuda" -- \
  --text "Hello from GPU" --out /content/output.wav
```

### 5. Download output

```python
from google.colab import files
files.download('/content/output.wav')
```

### Notes

- `/content` is wiped on disconnect — pin model weights to Drive to avoid
  re-downloading each session.
- ORT bundles its own CUDA libraries; no manual driver configuration is needed
  beyond selecting the T4 runtime.

## Open questions

- **ORT CUDA version pinning** — ORT 2.0.0-rc.12 bundles specific CUDA/cuDNN
  versions. Verify compatibility with the target driver before shipping.
- **TensorRT engine caching** — TRT requires a one-time engine build per
  (model, GPU, precision) tuple. Decide whether to ship pre-built engines or
  build on first run.
- **Multi-GPU** — `CUDAExecutionProvider::default()` uses device 0.
  Expose a `device_id` field in `BackendConfig` if needed.
