---
language:
  - en
  - zh
  - ja
  - ko
  - de
  - fr
  - es
  - it
  - pt
  - ru
license: apache-2.0
tags:
  - text-to-speech
  - tts
  - onnx
  - qwen3-tts
  - voice-cloning
library_name: onnxruntime
pipeline_tag: text-to-speech
base_model: Qwen/Qwen3-TTS-12Hz-0.6B-Base
---

<p align="center">
  <a href="https://github.com/wavekat/wavekat-tts">
    <img src="https://github.com/wavekat/wavekat-brand/raw/main/assets/banners/wavekat-tts-narrow.svg" alt="WaveKat TTS">
  </a>
</p>

# Qwen3-TTS 0.6B Base — Voice Clone (ONNX)

ONNX export of [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) for **voice cloning** with ONNX Runtime. No PyTorch required at inference time.

Provide a short reference audio clip and its transcript, and the model synthesizes new text in the same voice using In-Context Learning (ICL).

Both FP32 and INT4 (weight-only, RTN) variants are included.

> Exported and maintained by [WaveKat](https://github.com/wavekat) as part of the [wavekat-tts](https://github.com/wavekat/wavekat-tts) voice pipeline.

## Quick Start

```bash
pip install -r requirements.txt

# FP32
python generate_clone_onnx.py \
  --ref-audio ref.wav --ref-text "Transcript of the reference audio." \
  --text "New text to synthesize in the cloned voice." \
  -o output_fp32.wav

# INT4 (~4x smaller, faster)
python generate_clone_onnx.py --variant int4 \
  --ref-audio ref.wav --ref-text "Transcript of the reference audio." \
  --text "New text to synthesize in the cloned voice." \
  -o output_int4.wav
```

Reference audio should be **mono 24 kHz WAV** with a clear, single-speaker recording (3–10 seconds works well). The script will resample automatically via librosa if needed.

## Model Architecture

Qwen3-TTS 0.6B Base uses a 6-model voice-clone pipeline:

```
Ref audio --> [Speaker Encoder]      ECAPA-TDNN → 1024-d speaker embedding
         \-> [Tokenizer Encoder]     Mimi encoder → 16-group ref codes (12 Hz)

Text + Ref text + Speaker embed + Ref codes
             |
             v  (ICL prefill)
     [Talker LM]           28 layers, 1024 hidden
     predicts codebook group 0
             |
             v
     [Code Predictor]      5 layers, 1024 hidden
     predicts groups 1-15
             |
             v
     [Vocoder]             single forward pass
     concat(ref_codes, gen_codes) → 24 kHz waveform → trim ref portion
```

The pipeline is split into 6 ONNX models:

| Model | Description | Precision |
|-------|-------------|-----------|
| `speaker_encoder.onnx` | ECAPA-TDNN: mel → 1024-d speaker embedding | FP32 only |
| `tokenizer_encoder.onnx` | Mimi encoder: audio → 16-group codec codes | FP32 only |
| `talker_prefill.onnx` | Full sequence prefill with KV cache output | FP32 / INT4 |
| `talker_decode.onnx` | Single-step decode with KV cache | FP32 / INT4 |
| `code_predictor.onnx` | Predict codebook groups 1-15 | FP32 / INT4 |
| `vocoder.onnx` | Codes to 24 kHz waveform | FP32 / INT4 |

> Speaker encoder and tokenizer encoder are always FP32 — they run once per request and are small.

## Repository Structure

```
.
├── config.json                # Model config (dimensions, token IDs, language map)
├── speaker_encoder.onnx       # ECAPA-TDNN speaker encoder (FP32)
├── tokenizer_encoder.onnx     # Mimi speech tokenizer encoder (FP32)
├── tokenizer/                 # Text tokenizer (vocab, merges)
├── embeddings/                # Pre-extracted embedding weights (.npy)
├── fp32/                      # FP32 ONNX models
│   ├── talker_prefill.onnx
│   ├── talker_decode.onnx
│   ├── code_predictor.onnx
│   └── vocoder.onnx
├── int4/                      # INT4 weight-only quantized models
│   ├── talker_prefill.onnx
│   ├── talker_decode.onnx
│   ├── code_predictor.onnx
│   └── vocoder.onnx
├── generate_clone_onnx.py     # Reference ONNX-only voice clone script
└── requirements.txt           # Inference dependencies
```

## How It Works

1. **Speaker encoding** — the reference audio is converted to a log-mel spectrogram and passed through an ECAPA-TDNN encoder to produce a 1024-d speaker embedding.
2. **Reference code extraction** — the same audio is encoded by a Mimi tokenizer encoder into 16-group discrete codes at 12 Hz.
3. **ICL prefill** — the talker LM is prefilled with an interleaved sequence: text embeddings (ref transcript + target text) paired with codec embeddings (speaker embed + reference codes).
4. **Autoregressive decode** — the talker generates group-0 codec tokens, and the code predictor fills in groups 1-15 per frame.
5. **Vocoder** — reference codes are prepended to generated codes, the vocoder decodes the combined sequence, and the leading reference portion is trimmed proportionally.

## Supported Languages

English, Chinese, Japanese, Korean, German, French, Spanish, Italian, Portuguese, Russian.

## Reproducing the Export

The export scripts are in the [wavekat-tts](https://github.com/wavekat/wavekat-tts) repository:

```bash
cd tools/qwen3-tts-onnx
pip install -r requirements.txt

# Export FP32, quantize INT4, and package for HF
make clone-all
```

## About WaveKat

[WaveKat](https://github.com/wavekat) builds open-source voice pipeline components in Rust.
This ONNX export is maintained as part of [wavekat-tts](https://github.com/wavekat/wavekat-tts), which provides unified TTS inference across multiple backends.

## Acknowledgements

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) by the Qwen team at Alibaba Cloud
