#!/usr/bin/env python3
"""Generate WAV files using only ONNX models (no PyTorch at inference time).

This is the reference implementation for a full ONNX-only TTS pipeline,
matching the official Qwen3-TTS inference flow. It uses:
  - talker_prefill.onnx   (prefill sequence → logits + KV cache)
  - talker_decode.onnx     (single-step decode with KV cache)
  - code_predictor.onnx    (predict codebook groups 1-15)
  - vocoder.onnx           (codes → 24kHz waveform)
  - embeddings/*.npy       (text/codec embeddings for input construction)
  - tokenizer/             (text tokenization)
  - config.json            (model dimensions, token IDs, sampling config)

Examples:
  # Basic generation (FP32)
  python generate_onnx.py --text "Give every small business the voice of a big one." \
    --instruct "Speak in a warm and friendly female voice"

  # Use INT4 quantized models
  python generate_onnx.py --variant int4 \
    --text "Give every small business the voice of a big one." \
    --instruct "Speak in a warm and friendly female voice"

  # Chinese text
  python generate_onnx.py --variant int4 --lang chinese \
    --text "让每一家小企业，都拥有大企业的声音。" \
    --instruct "Speak in a warm and professional female voice"

  # Custom output + params
  python generate_onnx.py --text "AI phone answering for small businesses." \
    -o demo.wav --temperature 0.7
"""

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Embedding helpers (NumPy only, no PyTorch)
# ---------------------------------------------------------------------------

def text_project_numpy(token_ids, text_emb, fc1_w, fc1_b, fc2_w, fc2_b):
    """SiLU-gated MLP text projection in NumPy."""
    embeds = text_emb[token_ids]  # (N, hidden)
    hidden = embeds @ fc1_w.T + fc1_b
    activated = hidden * (1.0 / (1.0 + np.exp(-hidden)))  # SiLU
    return activated @ fc2_w.T + fc2_b


def load_embeddings(onnx_dir):
    """Load exported .npy embedding files."""
    edir = os.path.join(onnx_dir, "embeddings")
    d = {}
    for name in [
        "text_embedding",
        "text_projection_fc1_weight", "text_projection_fc1_bias",
        "text_projection_fc2_weight", "text_projection_fc2_bias",
        "talker_codec_embedding",
    ]:
        d[name] = np.load(os.path.join(edir, f"{name}.npy"))

    d["cp_codec_embeddings"] = []
    i = 0
    while True:
        path = os.path.join(edir, f"cp_codec_embedding_{i}.npy")
        if not os.path.exists(path):
            break
        d["cp_codec_embeddings"].append(np.load(path))
        i += 1

    return d


def load_config(onnx_dir):
    with open(os.path.join(onnx_dir, "config.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_top_k(logits, top_k, temperature):
    """Sample from logits with top-k and temperature."""
    if temperature != 1.0:
        logits = logits / temperature

    if top_k > 0 and top_k < len(logits):
        top_k_idx = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[top_k_idx] = logits[top_k_idx]
        logits = mask

    # Softmax
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / probs.sum()

    return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# ONNX inference pipeline
# ---------------------------------------------------------------------------

def generate_onnx(
    model_dir: str,
    variant: str,
    text: str,
    instruct: str | None,
    language: str,
    output_path: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    seed: int | None,
):
    if seed is not None:
        np.random.seed(seed)

    onnx_dir = os.path.join(model_dir, variant)
    config = load_config(model_dir)
    emb = load_embeddings(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))

    # Load ONNX sessions
    print(f"Loading ONNX models ({variant})...")
    prefill_sess = ort.InferenceSession(os.path.join(onnx_dir, "talker_prefill.onnx"))
    decode_sess = ort.InferenceSession(os.path.join(onnx_dir, "talker_decode.onnx"))
    cp_sess = ort.InferenceSession(os.path.join(onnx_dir, "code_predictor.onnx"))
    vocoder_sess = ort.InferenceSession(os.path.join(onnx_dir, "vocoder.onnx"))

    # Embedding arrays
    text_emb = emb["text_embedding"]
    fc1_w = emb["text_projection_fc1_weight"]
    fc1_b = emb["text_projection_fc1_bias"]
    fc2_w = emb["text_projection_fc2_weight"]
    fc2_b = emb["text_projection_fc2_bias"]
    codec_emb = emb["talker_codec_embedding"]
    cp_codec_embs = emb["cp_codec_embeddings"]

    def text_proj(token_ids):
        return text_project_numpy(token_ids, text_emb, fc1_w, fc1_b, fc2_w, fc2_b)

    # Model dimensions from config
    num_layers = config["talker_num_layers"]
    hidden_size = config["talker_hidden_size"]
    num_code_groups = config["talker_num_code_groups"]
    cp_num_layers = config["cp_num_layers"]
    cp_num_kv_heads = config["cp_num_kv_heads"]
    cp_head_dim = config["cp_head_dim"]
    vocab_size = config["talker_vocab_size"]
    codec_eos = config["codec_eos_token_id"]

    # Tokenize text
    chat_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(chat_text, add_special_tokens=False)

    # Tokenize instruct (if provided)
    instruct_tokens = None
    if instruct:
        instruct_text = f"<|im_start|>user\n{instruct}<|im_end|>\n"
        instruct_tokens = tokenizer.encode(instruct_text, add_special_tokens=False)

    print(f"  Text: '{text}'")
    if instruct:
        print(f"  Instruct: '{instruct}'")
    print(f"  Language: {language}")
    print(f"  Sampling: temp={temperature}, top_k={top_k}, rep_penalty={repetition_penalty}")

    # --- Build prefill embeddings ---
    # Codec prefix tokens
    language_id = config["codec_language_id"].get(language.lower())
    if language_id is not None:
        codec_prefix_ids = [
            config["codec_think_id"],
            config["codec_think_bos_id"],
            language_id,
            config["codec_think_eos_id"],
        ]
    else:
        codec_prefix_ids = [
            config["codec_nothink_id"],
            config["codec_think_bos_id"],
            config["codec_think_eos_id"],
        ]

    # Speaker embed (for preset speaker models; None for VoiceDesign)
    speaker_embed = None
    spk_id = config.get("spk_id", {})

    # Special embeddings
    tts_pad_embed = text_proj([config["tts_pad_token_id"]])[0]
    tts_bos_embed = text_proj([config["tts_bos_token_id"]])[0]
    tts_eos_embed = text_proj([config["tts_eos_token_id"]])[0]
    codec_pad_embed = codec_emb[config["codec_pad_id"]]
    codec_bos_embed = codec_emb[config["codec_bos_id"]]

    embeds_list = []

    # 1. Instruct embeddings (VoiceDesign: prepended before role)
    if instruct_tokens is not None:
        instruct_embed = text_proj(instruct_tokens)  # (N, hidden)
        embeds_list.append(instruct_embed)

    # 2. Role prefix: first 3 tokens of input_ids (text projection only, no codec)
    role_embed = text_proj(input_ids[:3])
    embeds_list.append(role_embed)

    # 3. Codec prefix: tts_pad + codec_embed for each prefix token
    for cid in codec_prefix_ids:
        embeds_list.append((tts_pad_embed + codec_emb[cid]).reshape(1, -1))

    # 4. Speaker (only for preset speaker models)
    if speaker_embed is not None:
        embeds_list.append((tts_pad_embed + speaker_embed.sum(axis=0)).reshape(1, -1))

    # 5. Transition: tts_bos + codec_pad
    embeds_list.append((tts_bos_embed + codec_pad_embed).reshape(1, -1))

    # 6. Non-streaming: all text tokens + tts_eos, each paired with codec_pad
    text_tokens = input_ids[3:-5]
    for tid in text_tokens:
        embeds_list.append((text_proj([tid])[0] + codec_pad_embed).reshape(1, -1))
    embeds_list.append((tts_eos_embed + codec_pad_embed).reshape(1, -1))

    # 7. Final: tts_pad + codec_bos
    embeds_list.append((tts_pad_embed + codec_bos_embed).reshape(1, -1))

    prefill_embeds = np.concatenate(embeds_list, axis=0)[np.newaxis, :, :].astype(np.float32)
    T = prefill_embeds.shape[1]
    attention_mask = np.ones((1, T), dtype=np.int64)
    position_ids = np.arange(T).reshape(1, 1, T).repeat(3, axis=0)

    # --- Run prefill ---
    print(f"  Prefill: {T} tokens")
    t0 = time.time()

    prefill_out = prefill_sess.run(None, {
        "inputs_embeds": prefill_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    })

    logits = prefill_out[0]
    hidden_states = prefill_out[1]
    kv_outputs = prefill_out[2:]
    past_keys = np.stack([kv_outputs[i * 2] for i in range(num_layers)])
    past_values = np.stack([kv_outputs[i * 2 + 1] for i in range(num_layers)])

    trailing_hidden = tts_pad_embed.reshape(1, -1)

    # --- Decode loop ---
    # Token suppression: control tokens [vocab_size-1024, vocab_size) except codec_eos
    suppress_mask = np.zeros(vocab_size, dtype=bool)
    suppress_mask[vocab_size - 1024:vocab_size] = True
    suppress_mask[codec_eos] = False

    all_codes = []
    current_pos = T
    generated_tokens = []

    for step in range(max_new_tokens):
        last_logits = logits[0, -1, :].copy()
        last_logits[suppress_mask] = -np.inf

        # Enforce min_new_tokens=2
        if step < 2:
            last_logits[codec_eos] = -np.inf

        # Repetition penalty
        if repetition_penalty != 1.0 and generated_tokens:
            seen = np.array(generated_tokens)
            scores = last_logits[seen]
            scores = np.where(scores > 0, scores / repetition_penalty,
                              scores * repetition_penalty)
            last_logits[seen] = scores

        # Sample group 0 token
        group0_token = sample_top_k(last_logits, top_k, temperature)

        if group0_token == codec_eos:
            break

        generated_tokens.append(group0_token)

        # Code predictor: groups 1-15
        frame_codes = [group0_token]
        talker_hidden = hidden_states[0, -1:, :]
        group0_embed = codec_emb[group0_token].reshape(1, -1)

        cp_input = np.concatenate([talker_hidden, group0_embed], axis=0)
        cp_input = cp_input[np.newaxis, :, :].astype(np.float32)
        cp_past_keys = np.zeros((cp_num_layers, 1, cp_num_kv_heads, 0, cp_head_dim), dtype=np.float32)
        cp_past_values = np.zeros((cp_num_layers, 1, cp_num_kv_heads, 0, cp_head_dim), dtype=np.float32)

        for g in range(num_code_groups - 1):
            cp_out = cp_sess.run(None, {
                "inputs_embeds": cp_input,
                "generation_steps": np.array([g], dtype=np.int64),
                "past_keys": cp_past_keys,
                "past_values": cp_past_values,
            })
            cp_logits = cp_out[0]
            cp_past_keys = cp_out[1]
            cp_past_values = cp_out[2]

            # Sample sub-talker token
            cp_last = cp_logits[0, -1, :]
            token = sample_top_k(cp_last, top_k, temperature)
            frame_codes.append(token)

            cp_embed = cp_codec_embs[g][token].reshape(1, 1, -1).astype(np.float32)
            cp_input = cp_embed

        all_codes.append(frame_codes)

        # Build next talker input
        next_embed = codec_emb[group0_token].copy()
        for g in range(num_code_groups - 1):
            next_embed = next_embed + cp_codec_embs[g][frame_codes[g + 1]]
        next_embed = next_embed + trailing_hidden[0]
        next_embed = next_embed.reshape(1, 1, -1).astype(np.float32)

        # Talker decode step
        decode_mask = np.ones((1, current_pos + 1), dtype=np.int64)
        decode_pos = np.array([[[current_pos]]]).repeat(3, axis=0)

        decode_out = decode_sess.run(None, {
            "inputs_embeds": next_embed,
            "attention_mask": decode_mask,
            "position_ids": decode_pos,
            "past_keys": past_keys,
            "past_values": past_values,
        })

        logits = decode_out[0]
        hidden_states = decode_out[1]
        past_keys = decode_out[2]
        past_values = decode_out[3]
        current_pos += 1

        # Progress
        if (step + 1) % 50 == 0:
            print(f"    ... {step + 1} frames")

    gen_time = time.time() - t0
    num_frames = len(all_codes)
    print(f"  Generated {num_frames} frames in {gen_time:.1f}s")

    if num_frames == 0:
        print("  ERROR: no frames generated")
        return

    # --- Vocoder: codes → waveform ---
    codes_arr = np.array(all_codes, dtype=np.int64)  # (T, 16)
    codes_input = codes_arr.T[np.newaxis, :, :]  # (1, 16, T)

    t0 = time.time()
    wav = vocoder_sess.run(None, {"codes": codes_input})[0].flatten()
    voc_time = time.time() - t0

    duration = len(wav) / 24000
    print(f"  Vocoder: {voc_time:.1f}s, audio: {duration:.1f}s")

    sf.write(output_path, wav, 24000)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate WAV using ONNX-only Qwen3-TTS pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python generate_onnx.py --text "Give every small business the voice of a big one." --instruct "Speak in a warm and friendly female voice"
  python generate_onnx.py --variant int4 --text "AI phone answering for small businesses." --instruct "Speak in a professional male voice"
  python generate_onnx.py --variant int4 --lang chinese --text "让每一家小企业，都拥有大企业的声音。"
  python generate_onnx.py --text "Hello" -o demo.wav --temperature 0.7
""",
    )
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--instruct", default=None,
                        help="Voice design instruction (e.g. 'Speak slowly in a deep male voice')")
    parser.add_argument("--lang", default="english",
                        help="Language (default: english)")
    parser.add_argument("--model-dir", default="./output/qwen3-tts-1.7b-voicedesign",
                        help="Root model directory (contains config.json, tokenizer/, embeddings/, fp32/, int4/)")
    parser.add_argument("--variant", default="fp32",
                        help="ONNX variant: fp32 or int4 (default: fp32)")
    parser.add_argument("-o", "--output", default="output.wav",
                        help="Output WAV path (default: output.wav)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max codec frames to generate (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature (default: 0.9)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling (default: 50)")
    parser.add_argument("--repetition-penalty", type=float, default=1.05,
                        help="Repetition penalty (default: 1.05)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_onnx(
        model_dir=args.model_dir,
        variant=args.variant,
        text=args.text,
        instruct=args.instruct,
        language=args.lang,
        output_path=args.output,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
