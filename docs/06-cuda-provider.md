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

The `ort` crate bundles `libonnxruntime_providers_cuda.so` but **not** the
underlying CUDA runtime libraries (cuBLAS, cuDNN, cuFFT, etc.). Those must be
installed on the host separately — see the platform-specific setup sections below.

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
  --provider cuda --output output.wav "Hello from GPU"
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

### 3. Install ORT system libraries

The prebuilt ORT CUDA binaries from `ort-sys` require glibc 2.38+, but Colab
runs Ubuntu 22.04 (glibc 2.35). Use `ORT_STRATEGY=system` with the pip-installed
`onnxruntime-gpu` instead, which is compiled for Ubuntu 22.04.

**Version requirement:** `ort-sys` 2.0.0-rc.12 requests ORT C API version 24.
The pip package version must match — `onnxruntime-gpu` N.x ships API version N
(e.g. `1.24.x` → API 24). Install the matching version:

```bash
pip install onnxruntime-gpu==1.24.0   # adjust patch if needed
```

Or install the latest and verify:

```bash
pip install -U onnxruntime-gpu
python -c "import onnxruntime; print(onnxruntime.__version__)"
```

After installing, create the symlinks the linker needs (the pip package ships
only the versioned `.so`, not the plain or major-version names):

**Notebook cell:**

```python
import onnxruntime, os

capi_dir = os.path.join(os.path.dirname(onnxruntime.__file__), "capi")
so_versioned = os.path.join(capi_dir, f"libonnxruntime.so.{onnxruntime.__version__}")

# Force-create symlinks — os.path.lexists catches stale/broken symlinks
# that os.path.exists would miss (e.g. left over from a previous ORT version).
for link in [
    os.path.join(capi_dir, "libonnxruntime.so"),    # ort-sys build script
    os.path.join(capi_dir, "libonnxruntime.so.1"),  # runtime ELF SONAME
]:
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(so_versioned, link)

os.environ["ORT_STRATEGY"]            = "system"
os.environ["ORT_LIB_LOCATION"]        = capi_dir
os.environ["ORT_PREFER_DYNAMIC_LINK"] = "1"
os.environ["LD_LIBRARY_PATH"]         = capi_dir + ":" + os.environ.get("LD_LIBRARY_PATH", "")
```

**Terminal (Colab shell):**

```bash
CAPI=/usr/local/lib/python3.12/dist-packages/onnxruntime/capi
ORT_VER=$(python -c "import onnxruntime; print(onnxruntime.__version__)")
ln -sf $CAPI/libonnxruntime.so.$ORT_VER $CAPI/libonnxruntime.so.1
ln -sf $CAPI/libonnxruntime.so.$ORT_VER $CAPI/libonnxruntime.so

export ORT_STRATEGY=system
export ORT_LIB_LOCATION=$CAPI
export ORT_PREFER_DYNAMIC_LINK=1
export LD_LIBRARY_PATH=$CAPI:$LD_LIBRARY_PATH
```

These env vars must be set before `cargo build` so that `ort-sys` finds the
system ORT instead of downloading its own prebuilt binaries.

### 5. Clone and build

```python
!git clone https://github.com/wavekat/wavekat-tts.git
%cd wavekat-tts
!cargo build --release --features "qwen3-tts,cuda"
```

### 6. Model weights

Mount Drive for persistent storage, then copy the model to local `/content/`
before loading. ORT 1.24 added a security check that rejects `.onnx.data`
external data paths resolving outside the model directory — HF Hub stores files
as symlinks (`int4/talker_prefill.onnx.data → ../../blobs/...`) which trigger
this check when accessed via the Drive FUSE mount. Copying with `cp -rL`
dereferences symlinks into real files. `/content/` is local NVMe so loading is
also faster than from Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
import glob, os, shutil

LOCAL    = "/content/wavekat-model"
DRIVE    = "/content/drive/MyDrive/wavekat-models"

if not os.path.isfile(f"{LOCAL}/config.json"):
    snapshots = glob.glob(f"{DRIVE}/models--*/snapshots/*/")
    if snapshots:
        snapshot = snapshots[0]
        print(f"Copying {snapshot} → {LOCAL} (resolving symlinks)...")
        shutil.copytree(snapshot, LOCAL, symlinks=False, dirs_exist_ok=True)
        print("Done.")
    else:
        print("Drive cache not found — will download to Drive on first run.")
        os.environ["WAVEKAT_MODEL_DIR"] = DRIVE

if os.path.isfile(f"{LOCAL}/config.json"):
    os.environ["WAVEKAT_MODEL_DIR"] = LOCAL
    print(f"WAVEKAT_MODEL_DIR={LOCAL}")
```

```bash
cargo run --release --example synthesize --features "qwen3-tts,cuda" -- \
  --provider cuda --output /content/output.wav "Hello from GPU"
```

### 7. Download output

```python
from google.colab import files
files.download('/content/output.wav')
```

### Notes

- `/content` is wiped on disconnect — pin model weights to Drive to avoid
  re-downloading each session.
- Colab's T4 runtime includes cuBLAS, cuDNN, and the CUDA driver pre-installed —
  no extra CUDA library setup needed beyond selecting the GPU runtime.

## Azure Ubuntu 24.04 GPU (T4) setup

Azure's **Standard_NC4as_T4_v3** SKU provides a single NVIDIA T4 (16 GB VRAM)
on Ubuntu 24.04 LTS. Ubuntu 24.04 ships glibc 2.39, so ORT's prebuilt CUDA
binaries work without the `ORT_STRATEGY=system` workaround required on Colab.

### 1. Provision the VM

```bash
az vm create \
  --resource-group <rg> \
  --name wavekat-gpu \
  --image Ubuntu2404 \
  --size Standard_NC4as_T4_v3 \
  --admin-username azureuser \
  --generate-ssh-keys
```

Open SSH if needed:

```bash
az vm open-port --resource-group <rg> --name wavekat-gpu --port 22
```

### 2. Install NVIDIA drivers and CUDA runtime libraries

```bash
ssh azureuser@<public-ip>

sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```

After reconnecting, verify the driver and add the NVIDIA CUDA repository to install
the runtime libraries that ORT's CUDA provider requires (cuBLAS, cuDNN, etc.):

```bash
# Verify driver
nvidia-smi   # expect "Tesla T4", CUDA driver ≥ 12.x

# Add NVIDIA CUDA 12 repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA 12 runtime libraries (cuBLAS, cuDNN, cuFFT, cuSolver, cuSPARSE)
# ort-sys 2.0.0-rc.12 targets cu12 by default; any 12.x minor version works.
sudo apt install -y cuda-libraries-12-6 libcudnn9-cuda-12
```

### 3. (Optional) Install TensorRT

Required only when building with the `tensorrt` feature. ORT bundles
`libonnxruntime_providers_tensorrt.so` but dynamically loads
`libnvinfer.so.10` at runtime — install TensorRT 10 from the NVIDIA CUDA
repository added in step 2:

```bash
sudo apt install -y libnvinfer10 libnvinfer-plugin10 libnvonnxparsers10
```

Verify:

```bash
ldconfig -p | grep nvinfer
# expect libnvinfer.so.10 → /usr/lib/x86_64-linux-gnu/libnvinfer.so.10
```

Skip this step if you only need the `cuda` feature.

### 4. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
source "$HOME/.cargo/env"
rustc --version
```

### 5. Install build dependencies

```bash
sudo apt install -y pkg-config libssl-dev
```

### 6. Clone and build

```bash
git clone https://github.com/wavekat/wavekat-tts.git
cd wavekat-tts
cargo build --release --features "qwen3-tts,cuda"
```

ORT will download its prebuilt CUDA libraries automatically (no extra env vars
needed on Ubuntu 24.04). For TensorRT, swap the feature flag for
`"qwen3-tts,tensorrt"` after completing step 3.

### 7. Model weights

The app auto-downloads weights from Hugging Face Hub on first run. Point
`HF_HOME` at a data disk so the cache doesn't fill the small OS disk on `/`:

```bash
sudo mkdir -p /checkpoints/huggingface
sudo chown -R $USER:$USER /checkpoints
export HF_HOME=/checkpoints/huggingface
```

### 8. Run

```bash
cargo run --release --example synthesize --features "qwen3-tts,cuda" -- \
  --provider cuda --output ~/output.wav "Hello from Azure GPU"
```

Confirm the CUDA EP is active:

```bash
ORT_LOG_LEVEL=1 cargo run --release --example synthesize --features "qwen3-tts,cuda" -- \
  --provider cuda --output ~/output.wav "Hello from Azure GPU" 2>&1 | grep -i cuda
# [I:ort:session] [CUDAExecutionProvider] Created CUDA EP on device 0
```

### Notes

- Ubuntu 24.04 has glibc 2.39 — no `ORT_STRATEGY=system` or symlink patching needed.
- ORT bundles `libonnxruntime_providers_cuda.so` and
  `libonnxruntime_providers_tensorrt.so` but **not** cuBLAS/cuDNN or
  `libnvinfer`. Steps 2 and 3 install those from the NVIDIA CUDA repository.
- The `Standard_NC4as_T4_v3` SKU is available in East US, West US 2, and several
  European regions. Check availability with `az vm list-skus`.
- Stop the VM when idle to avoid billing: `az vm deallocate --resource-group <rg> --name wavekat-gpu`.

## Open questions

- **ORT CUDA version pinning** — ORT 2.0.0-rc.12 bundles specific CUDA/cuDNN
  versions. Verify compatibility with the target driver before shipping.
- **TensorRT engine caching** — TRT requires a one-time engine build per
  (model, GPU, precision) tuple. Decide whether to ship pre-built engines or
  build on first run.
- **Multi-GPU** — `CUDAExecutionProvider::default()` uses device 0.
  Expose a `device_id` field in `BackendConfig` if needed.
