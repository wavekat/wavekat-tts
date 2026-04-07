# Benchmarking

## What is RTF

**Real-Time Factor (RTF)** = synthesis time / audio duration.

- RTF `0.35` → the model produces 1 s of audio in 0.35 s (2.9× faster than real-time)
- RTF `2.0`  → the model takes 2 s to produce 1 s of audio (2× slower than real-time)
- RTF `1.0`  → exactly real-time

RTF is the primary performance metric because it is independent of text length and
directly answers "can this hardware keep up with a live conversation?"

---

## Running the benchmark

### Quick human-readable run

```bash
make bench-rtf           # CPU (int4)
make bench-rtf-cuda      # CUDA — requires --features "qwen3-tts,cuda"
make bench-rtf-trt       # TensorRT — requires --features "qwen3-tts,tensorrt"
```

Output is a summary table printed to stdout:

```
sample    chars  rtf_mean  rtf_std  rtf_min  rtf_p50  rtf_p95  rtf_max   audio_s   synth_s
short        66    1.981    0.043    1.949    1.967    2.065    2.065      3.97      7.87
medium      243    2.038    0.018    2.015    2.039    2.068    2.068     20.64     42.08
long        655    2.345    0.116    2.157    2.375    2.503    2.503     42.80    102.28
```

### Saving results as CSV

```bash
make bench-csv           # saves to bench/results/cpu-int4.csv
make bench-csv-cuda      # saves to bench/results/cuda-t4-int4.csv
make bench-csv-trt       # saves to bench/results/trt-t4-int4.csv
```

These targets pipe `--csv` output to the appropriate file in `bench/results/`.
Cargo/bench progress goes to stderr; only the CSV rows go to the file.

### Advanced options

All options are passed after `--`:

```bash
cargo run --release --example bench_rtf --features "qwen3-tts,cuda" -- \
  --provider cuda \
  --precision fp32 \
  --warmup 2 \
  --iterations 10 \
  --language zh \
  --csv > bench/results/cuda-t4-fp32.csv
```

| Flag | Default | Description |
|------|---------|-------------|
| `--provider` | `cpu` | `cpu` \| `cuda` \| `tensorrt` \| `coreml` |
| `--precision` | `int4` | `int4` \| `fp32` |
| `--iterations` | `5` | Measured runs per sample |
| `--warmup` | `1` | Warmup runs (not counted) |
| `--language` | `en` | Language code |
| `--instruction` | `"Speak naturally and clearly."` | VoiceDesign prompt |
| `--model-dir` | _(auto-download)_ | Local model directory |
| `--csv` | _(off)_ | Emit CSV to stdout |

---

## Text samples

The benchmark uses three fixed English samples:

| Label | Chars | ~Words | ~Audio duration |
|-------|------:|------:|----------------|
| `short` | 66 | 11 | 3–5 s |
| `medium` | 243 | 42 | 18–25 s |
| `long` | 655 | 109 | 35–65 s |

These are intentionally varied to expose whether RTF scales with sequence length
(it does slightly, due to KV cache growth in the decode loop).

---

## Saving and committing results

1. Run on the target machine (e.g. Azure T4):
   ```bash
   make bench-csv-cuda
   ```
2. Commit the CSV:
   ```bash
   git add bench/results/cuda-t4-int4.csv
   git commit -m "bench: add T4 CUDA int4 results"
   git push
   ```
3. On push to `main`, the `update-bench` GitHub Actions workflow
   (`workflows/update-bench.yml`) detects the changed CSV, runs
   `scripts/update_bench_table.py`, and commits an updated `## Performance`
   table to `README.md` automatically.

CSV files live in `bench/results/` and are named `<provider>-<hardware>-<precision>.csv`.
Known names and their README labels:

| Filename | Label |
|----------|-------|
| `cpu-int4.csv` | CPU · int4 |
| `cpu-fp32.csv` | CPU · fp32 |
| `cuda-t4-int4.csv` | CUDA T4 · int4 |
| `cuda-t4-fp32.csv` | CUDA T4 · fp32 |
| `trt-t4-int4.csv` | TensorRT T4 · int4 |
| `trt-t4-fp32.csv` | TensorRT T4 · fp32 |

Unknown filenames fall back to a title-cased label derived from the stem.

---

## Reading the results

### Capacity planning

A single GPU running at RTF `0.35` spends 0.35 s of compute per 1 s of audio.
The theoretical maximum concurrent streams on one GPU is:

```
max_streams ≈ floor(1 / RTF)
```

At RTF `0.35` → **2–3 concurrent requests** before queuing builds up.
At RTF `2.0` (CPU baseline) the model cannot keep up with a single real-time stream.

### Latency

`synth_s` (mean) for a given sample length is the wall-clock latency a caller
experiences. Use the `p95` column for SLA planning — it bounds worst-case latency
under normal conditions.

### Provider comparison

| Provider | Expected RTF (T4, int4) | Notes |
|----------|:----------------------:|-------|
| CPU | ~2.0 | Baseline, no GPU needed |
| CUDA | ~0.3–0.5 | ORT CUDA EP, requires CUDA 12 + cuDNN 9 |
| TensorRT | ~0.15–0.3 | Higher setup cost, best throughput |

CUDA results to be added after T4 run. See `06-cuda-provider.md` for setup.

---

## Updating the README table manually

If you need to regenerate the table without pushing:

```bash
python3 scripts/update_bench_table.py
```

To check whether README.md is in sync with the current CSVs (useful in CI):

```bash
python3 scripts/update_bench_table.py --check
```

Exit code `1` means the table is stale; `0` means it is up to date.
