# ONNX Export & Quantization Guide

This document explains how the published INT4 ONNX models are produced and how
to reproduce or customize the process.

## Overview

The pipeline is: **PyTorch → FP32 ONNX → INT4 ONNX**.

1. Export each model component (talker, code predictor, vocoder) to FP32 ONNX
2. Validate ONNX outputs match PyTorch (per-stage + end-to-end)
3. Quantize FP32 weights to INT4

## Quantization method

**INT4 weight-only RTN (Round-To-Nearest)** via ONNX Runtime's
[`MatMulNBitsQuantizer`](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

| Setting | Value |
|---------|-------|
| Bits | 4 |
| Block size | 128 |
| Symmetric | Yes |
| Accuracy level | 4 (int8 input, int32 accumulator) |
| Calibration data | Not required |
| Vocoder | Kept at FP32 (quantization degrades audio quality) |

RTN is the simplest weight-only method — it rounds each weight block to the
nearest 4-bit value. No calibration dataset is needed, which makes it
straightforward to reproduce. The tradeoff is slightly lower quality compared
to calibration-based methods (e.g., GPTQ, AWQ), but for Qwen3-TTS the
difference is negligible in practice.

## Quick start

```bash
cd tools/qwen3-tts-onnx
pip install -r requirements.txt

# 1.7B VoiceDesign: export FP32 ONNX, validate, quantize to INT4
make all

# 0.6B Base (voice clone): export + quantize
make clone-all
```

## Custom quantization

The quantization script can be run standalone:

```bash
cd tools/qwen3-tts-onnx

# Smaller block size = higher quality, larger files
python quantize_int4.py --block-size 64

# Also quantize the vocoder (may reduce audio quality)
python quantize_int4.py --include-vocoder
```

See [`tools/qwen3-tts-onnx/quantize_int4.py`](../tools/qwen3-tts-onnx/quantize_int4.py)
for all options.

## Model components

Each Qwen3-TTS model is split into 4 ONNX files:

| Model | Description | FP32 Size (1.7B) | INT4 Size (1.7B) |
|-------|-------------|-------------------|-------------------|
| `talker_prefill.onnx` | Full sequence prefill with KV cache output | 5.3 GB | 1.4 GB |
| `talker_decode.onnx` | Single-step decode with KV cache | 5.3 GB | 1.4 GB |
| `code_predictor.onnx` | Predict codebook groups 1-15 | 440 MB | 322 MB |
| `vocoder.onnx` | Codes to 24 kHz waveform | 876 MB | 558 MB (FP32 copy) |

Embeddings and tokenizer files are shared between FP32 and INT4 variants.

## References

- [ONNX Runtime Quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [`tools/qwen3-tts-onnx/`](../tools/qwen3-tts-onnx/) — full export and quantization toolkit
- [`tools/qwen3-tts-onnx/PLAN.md`](../tools/qwen3-tts-onnx/PLAN.md) — detailed export design notes
