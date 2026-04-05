# ONNX Export Plan for Qwen3-TTS

## Goal

Create ONNX export scripts for the official Qwen3-TTS PyTorch models, starting with
`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`. Validate that ONNX inference produces
identical output to PyTorch inference. Designed to support all model variants later.

## Model Architecture

Qwen3-TTS is a three-stage pipeline:

```
Text ──► [Tokenizer + Embedding Construction] ──► inputs_embeds
                                                      │
                                                      ▼
                                              [Talker LM]  ◄── autoregressive
                                              predicts 1st codebook token
                                                      │
                                                      ▼
                                            [Code Predictor]  ◄── autoregressive
                                            predicts codebook tokens 2-16
                                                      │
                                                      ▼
                                              [Vocoder]  ◄── single forward pass
                                              codes → 24kHz waveform
```

### Component Dimensions (1.7B VoiceDesign)

| Component | Param | Value |
|-----------|-------|-------|
| **Talker** | hidden_size | 2048 |
| | num_layers | 24 |
| | num_attention_heads | 32 |
| | num_kv_heads | 4 |
| | head_dim | 128 |
| | vocab_size | 3072 (2048 codec + 1024 control) |
| | text_hidden_size | 2048 |
| **Code Predictor** | hidden_size | 1024 |
| | num_layers | 5 |
| | num_attention_heads | 16 |
| | num_kv_heads | 8 |
| | head_dim | 128 |
| | vocab_size | 2048 |
| | num_code_groups | 16 (talker predicts group 0, CP predicts 1-15) |
| **Projection** | small_to_mtp_projection | Linear(2048 → 1024, bias=True) |
| **Text Projection** | SiLU-gated MLP | 2048 → 2048 → 2048 |
| **Vocoder** | num_quantizers | 16 |
| | codebook_size | 2048 |
| | total_upsample | 1920x (12 Hz codes → 24kHz audio) |

### What VoiceDesign Doesn't Have

- No speaker encoder (speakers are pre-defined as codec token ID sequences in config)
- No voice cloning / ICL prompt path

## Inference Flow (What Host Code Does)

The embedding construction and autoregressive loop are complex Python logic that
**cannot** be captured in ONNX. This stays in host code (Rust/Python):

### 1. Build Prefill Embeddings

```
Position  Content                              Computation
────────  ─────────────────────────────────     ─────────��────────────────
[0..2]    <|im_start|> assistant \n             text_projection(text_embedding[tokens])
[3..N]    codec_think, think_bos, lang_id,     tts_pad_embed + codec_embedding[tokens]
          think_eos, [speaker_embed]
[N+1]     tts_bos                              text_projection(tts_bos) + codec_embedding[codec_pad]
[N+2]     first_text_token                     text_projection(text[0]) + codec_embedding[codec_bos]

Non-streaming mode: all remaining text tokens are prepended with codec_pad embeddings.
Streaming mode: remaining text tokens fed incrementally as trailing_text_hidden.
```

### 2. Talker Prefill

```
talker_prefill(inputs_embeds, attention_mask, position_ids)
  → logits, hidden_states, KV cache (24 layers)
```

### 3. Autoregressive Decode Loop

```
for each step:
  1. Sample group-0 token from talker logits (top-k=50, top-p=1.0, temp=0.9, rep_penalty=1.05)
     → stop if codec_eos_token_id
  2. Run code predictor for groups 1-15:
     - Input: concat(talker_hidden_state, group0_codec_embed) projected through small_to_mtp
     - Autoregressively predict 15 tokens (one per codebook group)
     - Sub-talker sampling: top-k=50, top-p=1.0, temp=0.9
  3. Build next talker input:
     next_embed = sum(all 16 codec embeddings) + trailing_text_hidden[step]
  4. Run talker decode step → next logits + hidden_state
```

### 4. Vocoder

```
vocoder(codes)  # codes shape: (1, 16, num_steps)
  → waveform at 24kHz
```

## ONNX Models to Export

### Model 1: `talker_prefill.onnx`

Processes the full prefill sequence, outputs logits + hidden states + KV cache.

| Direction | Name | Shape | Type |
|-----------|------|-------|------|
| Input | `inputs_embeds` | `(1, T, 2048)` | float32 |
| Input | `attention_mask` | `(1, T)` | int64 |
| Input | `position_ids` | `(3, 1, T)` | int64 |
| Output | `logits` | `(1, T, 3072)` | float32 |
| Output | `hidden_states` | `(1, T, 2048)` | float32 |
| Output | `present_key_0` .. `present_key_23` | `(1, 4, T, 128)` each | float32 |
| Output | `present_value_0` .. `present_value_23` | `(1, 4, T, 128)` each | float32 |

Dynamic axes: `T` (sequence length).

### Model 2: `talker_decode.onnx`

Single-step decode with existing KV cache.

| Direction | Name | Shape | Type |
|-----------|------|-------|------|
| Input | `inputs_embeds` | `(1, 1, 2048)` | float32 |
| Input | `attention_mask` | `(1, total)` | int64 |
| Input | `position_ids` | `(3, 1, 1)` | int64 |
| Input | `past_keys` | `(24, 1, 4, past, 128)` | float32 |
| Input | `past_values` | `(24, 1, 4, past, 128)` | float32 |
| Output | `logits` | `(1, 1, 3072)` | float32 |
| Output | `hidden_states` | `(1, 1, 2048)` | float32 |
| Output | `present_keys` | `(24, 1, 4, total, 128)` | float32 |
| Output | `present_values` | `(24, 1, 4, total, 128)` | float32 |

Dynamic axes: `past`, `total` (total = past + 1).

### Model 3: `code_predictor.onnx`

Predicts codebook tokens for groups 1-15, one group per call.

| Direction | Name | Shape | Type |
|-----------|------|-------|------|
| Input | `inputs_embeds` | `(1, S, 1024)` | float32 |
| Input | `generation_steps` | `(1,)` | int64 |
| Input | `past_keys` | `(5, 1, 8, past, 128)` | float32 |
| Input | `past_values` | `(5, 1, 8, past, 128)` | float32 |
| Output | `logits` | `(1, S, 2048)` | float32 |
| Output | `present_keys` | `(5, 1, 8, total, 128)` | float32 |
| Output | `present_values` | `(5, 1, 8, total, 128)` | float32 |

Dynamic axes: `S`, `past`, `total`.

Host code applies `small_to_mtp_projection` before feeding `inputs_embeds`.
The `generation_steps` input selects which `lm_head` to use (0-14 → groups 1-15).

### Model 4: `vocoder.onnx`

Converts discrete codes to audio waveform.

| Direction | Name | Shape | Type |
|-----------|------|-------|------|
| Input | `codes` | `(1, 16, T)` | int64 |
| Output | `waveform` | `(1, 1, T*1920)` | float32 |

Dynamic axes: `T`.

## Embedding Files (.npy)

Extracted from PyTorch weights, used by host code for embedding construction.

| File | Shape | Source |
|------|-------|--------|
| `text_embedding.npy` | `(151936, 2048)` | `model.talker.model.text_embedding.weight` |
| `text_projection_fc1_weight.npy` | `(2048, 2048)` | `model.talker.text_projection.linear_fc1.weight` |
| `text_projection_fc1_bias.npy` | `(2048,)` | `model.talker.text_projection.linear_fc1.bias` |
| `text_projection_fc2_weight.npy` | `(2048, 2048)` | `model.talker.text_projection.linear_fc2.weight` |
| `text_projection_fc2_bias.npy` | `(2048,)` | `model.talker.text_projection.linear_fc2.bias` |
| `talker_codec_embedding.npy` | `(3072, 2048)` | `model.talker.model.codec_embedding.weight` |
| `cp_codec_embedding_0..14.npy` | `(2048, 2048)` x15 | `model.talker.code_predictor.model.codec_embedding[i].weight` |
| `small_to_mtp_projection_weight.npy` | `(1024, 2048)` | `model.talker.code_predictor.small_to_mtp_projection.weight` |
| `small_to_mtp_projection_bias.npy` | `(1024,)` | `model.talker.code_predictor.small_to_mtp_projection.bias` |

Text projection is a SiLU-gated MLP:
```
hidden = fc1_weight @ input + fc1_bias    # (2048,)
output = fc2_weight @ (hidden * sigmoid(hidden)) + fc2_bias  # (2048,)
```

## Export Technical Details

### Avoiding torch.vmap SDPA Issues

Load model with `attn_implementation="eager"`. This forces the eager attention code path
(`repeat_kv` + explicit matmul) instead of `torch.nn.functional.scaled_dot_product_attention`,
which uses `torch.vmap` for GQA repeat and produces `Expand` ops incompatible with some
ONNX runtimes.

### KV Cache Handling

HuggingFace uses `DynamicCache` internally. Our export wrappers:

- **Prefill wrapper**: Let model create DynamicCache, then extract per-layer key/value
  tensors from it as individual outputs.
- **Decode wrapper**: Reconstruct DynamicCache from stacked input tensors, run model
  for 1 step, re-stack updated KV as output tensors.

### Code Predictor lm_head Selection

The CP has 15 separate `nn.Linear` heads (one per codebook group). ONNX can't trace
dynamic `ModuleList` indexing. Solution: stack all head weights into a single buffer and
use `generation_steps` to index:

```python
all_weights = torch.stack([h.weight for h in self.lm_heads])  # (15, 2048, 1024)
weight = all_weights[generation_steps[0]]
logits = F.linear(hidden_states, weight)
```

### Vocoder Codebook Precomputation

`EuclideanCodebook.decode()` divides `embedding_sum / cluster_usage` at runtime.
Before export, precompute: `embedding = embedding_sum / cluster_usage.clamp(min=eps)[:, None]`
and replace with a direct `F.embedding` lookup.

### External Data Consolidation

Large models produce scattered weight files during ONNX export. Consolidate into a
single `.onnx.data` file per model using:
```python
onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True)
```

## Script Details

### `export_embeddings.py`

```
Args: --model-id, --output-dir
1. Load model from HuggingFace (float32)
2. Extract all embedding tensors listed above as .npy
3. Serialize config.json with token IDs, dimensions, speaker/language maps
4. Copy tokenizer files (vocab.json, merges.txt)
```

### `export_talker.py`

```
Args: --model-id, --output-dir
1. Load model with attn_implementation="eager", float32
2. Create TalkerPrefillWrapper(model.talker.model, model.talker.codec_head)
3. Export with dummy inputs, opset 17, dynamic axes on seq dim
4. Create TalkerDecodeWrapper (same model, with KV cache I/O)
5. Export with dummy inputs including past KV cache
6. Consolidate external data
7. Quick validation: run both models, check output shapes
```

### `export_code_predictor.py`

```
Args: --model-id, --output-dir
1. Load model, extract model.talker.code_predictor
2. Create CodePredictorWrapper (CP transformer + stacked lm_heads)
3. Export with dummy inputs, opset 17
4. Consolidate external data
5. Quick validation: check output shapes, verify lm_head selection works
```

### `export_vocoder.py`

```
Args: --model-id, --output-dir
1. Load speech tokenizer from model
2. Extract decoder (Qwen3TTSTokenizerV2Decoder)
3. Precompute EuclideanCodebook embeddings (replace runtime division)
4. Patch pre_transformer to use_cache=False
5. Create VocoderWrapper
6. Export with dummy codes input, opset 17
7. Consolidate external data
8. Quick validation: compare PyTorch vs ONNX output on random codes
```

### `validate.py`

```
Args: --model-id, --onnx-dir
Stage 1: Embedding validation (NumPy vs PyTorch, atol=1e-5)
Stage 2: Talker prefill validation (logits, hidden_states, KV cache, atol=1e-4)
Stage 3: Talker decode validation (single step, atol=1e-4)
Stage 4: Code predictor validation (all 15 groups, atol=1e-4)
Stage 5: Vocoder validation (waveform, atol=1e-3, report SNR)
Stage 6: End-to-end (greedy decode same text, compare code sequences + audio)
         Save both WAV files for manual listening
```

### `run_all.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL_ID="${1:-Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
OUTPUT_DIR="${2:-./output/qwen3-tts-1.7b-voicedesign}"

python export_embeddings.py  --model-id "$MODEL_ID" --output-dir "$OUTPUT_DIR"
python export_talker.py      --model-id "$MODEL_ID" --output-dir "$OUTPUT_DIR"
python export_code_predictor.py --model-id "$MODEL_ID" --output-dir "$OUTPUT_DIR"
python export_vocoder.py     --model-id "$MODEL_ID" --output-dir "$OUTPUT_DIR"
python validate.py           --model-id "$MODEL_ID" --onnx-dir "$OUTPUT_DIR"
```

## Output Directory Structure

```
output/qwen3-tts-1.7b-voicedesign/
  talker_prefill.onnx
  talker_prefill.onnx.data
  talker_decode.onnx
  talker_decode.onnx.data
  code_predictor.onnx
  code_predictor.onnx.data
  vocoder.onnx
  vocoder.onnx.data
  embeddings/
    text_embedding.npy
    text_projection_fc1_weight.npy
    text_projection_fc1_bias.npy
    text_projection_fc2_weight.npy
    text_projection_fc2_bias.npy
    talker_codec_embedding.npy
    cp_codec_embedding_0.npy
    ...
    cp_codec_embedding_14.npy
    small_to_mtp_projection_weight.npy
    small_to_mtp_projection_bias.npy
  tokenizer/
    vocab.json
    merges.txt
  config.json
```

## Supporting All Models Later

The scripts accept `--model-id` and read dimensions from the model's config. Key
differences across variants:

| Variant | hidden | layers | kv_heads | text_proj | Speaker Encoder |
|---------|--------|--------|----------|-----------|-----------------|
| 0.6B-VoiceDesign | 1024 | 20 | 2 | 2048→1024 | No |
| 0.6B-Base | 1024 | 20 | 2 | 2048→1024 | Yes (ECAPA-TDNN) |
| 1.7B-VoiceDesign | 2048 | 24 | 4 | 2048→2048 | No |
| 1.7B-CustomVoice | 2048 | 24 | 4 | 2048→2048 | No |
| 1.7B-Base | 2048 | 24 | 4 | 2048→2048 | Yes (ECAPA-TDNN) |

For Base models, an additional `export_speaker_encoder.py` will be needed later.
The CP dimensions (hidden=1024, layers=5, heads=16, kv_heads=8) are the same across all variants.

## Verification Criteria

- Per-stage max absolute error < tolerance (1e-4 for transformers, 1e-3 for vocoder)
- End-to-end greedy decode: code sequences must be **identical**
- Audio SNR > 40dB between PyTorch and ONNX outputs
- Manual listening: saved WAV files sound the same
