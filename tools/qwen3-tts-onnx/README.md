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
library_name: onnxruntime
pipeline_tag: text-to-speech
base_model: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
---

<p align="center">
  <a href="https://github.com/wavekat/wavekat-tts">
    <img src="https://github.com/wavekat/wavekat-brand/raw/main/assets/banners/wavekat-tts-narrow.svg" alt="WaveKat TTS">
  </a>
</p>

# Qwen3-TTS 1.7B VoiceDesign (ONNX)

ONNX export of [Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) for inference with ONNX Runtime. No PyTorch required at inference time.

Both FP32 and INT4 (weight-only, RTN) variants are included.

> Exported and maintained by [WaveKat](https://github.com/wavekat) as part of the [wavekat-tts](https://github.com/wavekat/wavekat-tts) voice pipeline.

## Quick Start

```bash
pip install -r requirements.txt

# FP32
python generate_onnx.py --text "Give every small business the voice of a big one." \
  --instruct "Speak in a warm and friendly female voice" \
  -o output_fp32.wav

# INT4 (~4x smaller, faster)
python generate_onnx.py --variant int4 \
  --text "Give every small business the voice of a big one." \
  --instruct "Speak in a warm and friendly female voice" \
  -o output_int4.wav

# Chinese
python generate_onnx.py --variant int4 --lang chinese \
  --text "让每一家小企业，都拥有大企业的声音。" \
  --instruct "Speak in a warm and professional female voice" \
  -o output_zh.wav
```

## Model Architecture

Qwen3-TTS is a three-stage autoregressive pipeline:

```
Text --> [Tokenizer + Embedding Construction] --> inputs_embeds
             |
             v
     [Talker LM]           28 layers, 2048 hidden
     predicts codebook group 0
             |
             v
     [Code Predictor]      5 layers, 1024 hidden
     predicts groups 1-15
             |
             v
     [Vocoder]             single forward pass
     16 codebook groups --> 24kHz waveform
```

The pipeline is split into 4 ONNX models:

| Model | Description | FP32 Size | INT4 Size |
|-------|-------------|-----------|-----------|
| `talker_prefill.onnx` | Full sequence prefill with KV cache output | 5.3 GB | 1.4 GB |
| `talker_decode.onnx` | Single-step decode with KV cache | 5.3 GB | 1.4 GB |
| `code_predictor.onnx` | Predict codebook groups 1-15 | 440 MB | 322 MB |
| `vocoder.onnx` | Codes to 24kHz waveform | 876 MB | 558 MB |

## Repository Structure

```
.
├── config.json              # Model config (dimensions, token IDs, language map)
├── tokenizer/               # Text tokenizer (vocab, merges, config)
├── embeddings/              # Pre-extracted embedding weights (.npy)
├── fp32/                    # FP32 ONNX models
│   ├── talker_prefill.onnx
│   ├── talker_decode.onnx
│   ├── code_predictor.onnx
│   └── vocoder.onnx
├── int4/                    # INT4 weight-only quantized models
│   ├── talker_prefill.onnx
│   ├── talker_decode.onnx
│   ├── code_predictor.onnx
│   └── vocoder.onnx
├── generate_onnx.py         # Reference ONNX-only inference script
└── requirements.txt         # Inference dependencies
```

## Supported Languages

English, Chinese, Japanese, Korean, German, French, Spanish, Italian, Portuguese, Russian.

## Reproducing the Export

The export scripts are in the [wavekat-tts](https://github.com/wavekat/wavekat-tts) repository:

```bash
cd tools/qwen3-tts-onnx
pip install -r requirements.txt

# Export FP32, validate, and quantize INT4
make all
```

## About WaveKat

[WaveKat](https://github.com/wavekat) builds open-source voice pipeline components in Rust.
This ONNX export is maintained as part of [wavekat-tts](https://github.com/wavekat/wavekat-tts), which provides unified TTS inference across multiple backends.

## Acknowledgements

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) by the Qwen team at Alibaba Cloud
