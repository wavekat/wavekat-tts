#!/usr/bin/env python3
"""Validate ONNX exports against PyTorch output for Qwen3-TTS.

Runs per-stage comparisons and an end-to-end greedy decode test.
"""

import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from transformers import AutoTokenizer

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def text_project_numpy(token_ids, text_embedding, fc1_w, fc1_b, fc2_w, fc2_b):
    """Replicate text_projection MLP (SiLU-gated) in NumPy."""
    # token_ids: list of int
    embeds = text_embedding[token_ids]  # (T, text_hidden)
    hidden = embeds @ fc1_w.T + fc1_b  # (T, intermediate)
    activated = hidden * (1.0 / (1.0 + np.exp(-hidden)))  # SiLU
    output = activated @ fc2_w.T + fc2_b  # (T, output)
    return output


def load_embeddings(output_dir):
    """Load exported .npy embedding files."""
    edir = os.path.join(output_dir, "embeddings")
    d = {}
    for name in [
        "text_embedding",
        "text_projection_fc1_weight", "text_projection_fc1_bias",
        "text_projection_fc2_weight", "text_projection_fc2_bias",
        "talker_codec_embedding",
    ]:
        d[name] = np.load(os.path.join(edir, f"{name}.npy"))

    # CP codec embeddings
    d["cp_codec_embeddings"] = []
    i = 0
    while True:
        path = os.path.join(edir, f"cp_codec_embedding_{i}.npy")
        if not os.path.exists(path):
            break
        d["cp_codec_embeddings"].append(np.load(path))
        i += 1

    # Projection (may not exist if Identity)
    proj_w_path = os.path.join(edir, "small_to_mtp_projection_weight.npy")
    if os.path.exists(proj_w_path):
        d["small_to_mtp_projection_weight"] = np.load(proj_w_path)
        d["small_to_mtp_projection_bias"] = np.load(
            os.path.join(edir, "small_to_mtp_projection_bias.npy")
        )
    return d


def load_config(output_dir):
    with open(os.path.join(output_dir, "config.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Stage 1: Embeddings
# ---------------------------------------------------------------------------

def validate_embeddings(model, output_dir):
    """Compare text_projection output: PyTorch vs NumPy from .npy files."""
    print("\n=== Stage 1: Embedding Validation ===")
    emb = load_embeddings(output_dir)

    # Pick a few token IDs to test
    test_ids = [151644, 77091, 198, 151671, 151672, 151673, 100, 200, 300]

    with torch.no_grad():
        pt_text_emb = model.talker.model.text_embedding(
            torch.tensor([test_ids])
        )  # (1, N, text_hidden)
        pt_projected = model.talker.text_projection(pt_text_emb)  # (1, N, hidden)
        pt_projected = pt_projected.squeeze(0).numpy()

    np_projected = text_project_numpy(
        test_ids,
        emb["text_embedding"],
        emb["text_projection_fc1_weight"],
        emb["text_projection_fc1_bias"],
        emb["text_projection_fc2_weight"],
        emb["text_projection_fc2_bias"],
    )

    max_err = np.max(np.abs(pt_projected - np_projected))
    mean_err = np.mean(np.abs(pt_projected - np_projected))
    print(f"  Text projection: max_err={max_err:.6e}, mean_err={mean_err:.6e}")
    ok = max_err < 1e-4
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Stage 2: Talker Prefill
# ---------------------------------------------------------------------------

def validate_talker_prefill(model, output_dir):
    """Compare talker prefill: PyTorch vs ONNX."""
    print("\n=== Stage 2: Talker Prefill Validation ===")
    cfg = model.config.talker_config
    hidden_size = cfg.hidden_size

    # Build a simple inputs_embeds
    T = 8
    torch.manual_seed(42)
    inputs_embeds = torch.randn(1, T, hidden_size)
    attention_mask = torch.ones(1, T, dtype=torch.int64)
    position_ids = torch.arange(T).unsqueeze(0).unsqueeze(0).expand(3, 1, T)
    cache_position = torch.arange(T)

    with torch.no_grad():
        pt_out = model.talker.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            cache_position=cache_position,
        )
        pt_hidden = pt_out.last_hidden_state.numpy()
        pt_logits = model.talker.codec_head(pt_out.last_hidden_state).numpy()

    # ONNX
    onnx_path = os.path.join(output_dir, "talker_prefill.onnx")
    if not os.path.exists(onnx_path):
        print("  SKIP: talker_prefill.onnx not found")
        return True

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {
        "inputs_embeds": inputs_embeds.numpy(),
        "attention_mask": attention_mask.numpy(),
        "position_ids": position_ids.numpy(),
    })

    ort_logits = ort_out[0]
    ort_hidden = ort_out[1]

    logits_err = np.max(np.abs(pt_logits - ort_logits))
    hidden_err = np.max(np.abs(pt_hidden - ort_hidden))
    print(f"  Logits max_err={logits_err:.6e}")
    print(f"  Hidden max_err={hidden_err:.6e}")
    ok = logits_err < 1e-3 and hidden_err < 1e-3
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Stage 3: Talker Decode
# ---------------------------------------------------------------------------

def validate_talker_decode(model, output_dir):
    """Compare single-step talker decode: PyTorch vs ONNX."""
    print("\n=== Stage 3: Talker Decode Validation ===")
    from transformers.cache_utils import DynamicCache

    cfg = model.config.talker_config
    hidden_size = cfg.hidden_size
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim

    # Run prefill first to get real KV cache
    T = 6
    torch.manual_seed(42)
    prefill_embeds = torch.randn(1, T, hidden_size)
    prefill_mask = torch.ones(1, T, dtype=torch.int64)
    prefill_pos = torch.arange(T).unsqueeze(0).unsqueeze(0).expand(3, 1, T)

    with torch.no_grad():
        prefill_out = model.talker.model(
            inputs_embeds=prefill_embeds,
            attention_mask=prefill_mask,
            position_ids=prefill_pos,
            use_cache=True,
            cache_position=torch.arange(T),
        )
        past_kv = prefill_out.past_key_values

    # Build stacked KV for ONNX
    past_keys_np = np.stack([past_kv.key_cache[i].numpy() for i in range(num_layers)])
    past_values_np = np.stack([past_kv.value_cache[i].numpy() for i in range(num_layers)])

    # Single decode step
    decode_embeds = torch.randn(1, 1, hidden_size)
    decode_mask = torch.ones(1, T + 1, dtype=torch.int64)
    decode_pos = torch.tensor([[[T]]]).expand(3, 1, 1)

    with torch.no_grad():
        pt_decode = model.talker.model(
            inputs_embeds=decode_embeds,
            attention_mask=decode_mask,
            position_ids=decode_pos,
            past_key_values=past_kv,
            use_cache=True,
            cache_position=torch.tensor([T]),
        )
        pt_logits = model.talker.codec_head(pt_decode.last_hidden_state).numpy()
        pt_hidden = pt_decode.last_hidden_state.numpy()

    # ONNX
    onnx_path = os.path.join(output_dir, "talker_decode.onnx")
    if not os.path.exists(onnx_path):
        print("  SKIP: talker_decode.onnx not found")
        return True

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {
        "inputs_embeds": decode_embeds.numpy(),
        "attention_mask": decode_mask.numpy(),
        "position_ids": decode_pos.numpy(),
        "past_keys": past_keys_np,
        "past_values": past_values_np,
    })

    logits_err = np.max(np.abs(pt_logits - ort_out[0]))
    hidden_err = np.max(np.abs(pt_hidden - ort_out[1]))
    print(f"  Logits max_err={logits_err:.6e}")
    print(f"  Hidden max_err={hidden_err:.6e}")
    ok = logits_err < 1e-3 and hidden_err < 1e-3
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Stage 4: Code Predictor
# ---------------------------------------------------------------------------

def validate_code_predictor(model, output_dir):
    """Compare code predictor: PyTorch vs ONNX for a single group prediction."""
    print("\n=== Stage 4: Code Predictor Validation ===")

    cfg = model.config.talker_config
    cp_cfg = cfg.code_predictor_config
    talker_hidden = cfg.hidden_size

    # Simulate: talker hidden state + group0 codec embed
    torch.manual_seed(42)
    hidden_state = torch.randn(1, 1, talker_hidden)
    group0_embed = torch.randn(1, 1, talker_hidden)
    inputs_embeds = torch.cat([hidden_state, group0_embed], dim=1)  # (1, 2, talker_hidden)

    # PyTorch: use the code predictor's forward
    cp = model.talker.code_predictor
    with torch.no_grad():
        projected = cp.small_to_mtp_projection(inputs_embeds)
        cp_out = cp.model(
            inputs_embeds=projected,
            use_cache=True,
            cache_position=torch.arange(2),
        )
        pt_hidden = cp_out.last_hidden_state
        pt_logits = cp.lm_head[0](pt_hidden).numpy()  # group 0 → lm_head[0]

    # ONNX
    onnx_path = os.path.join(output_dir, "code_predictor.onnx")
    if not os.path.exists(onnx_path):
        print("  SKIP: code_predictor.onnx not found")
        return True

    sess = ort.InferenceSession(onnx_path)
    cp_num_layers = cp_cfg.num_hidden_layers
    cp_num_kv_heads = cp_cfg.num_key_value_heads
    cp_head_dim = cp_cfg.head_dim

    ort_out = sess.run(None, {
        "inputs_embeds": inputs_embeds.numpy(),
        "generation_steps": np.array([0], dtype=np.int64),
        "past_keys": np.zeros((cp_num_layers, 1, cp_num_kv_heads, 0, cp_head_dim), dtype=np.float32),
        "past_values": np.zeros((cp_num_layers, 1, cp_num_kv_heads, 0, cp_head_dim), dtype=np.float32),
    })

    logits_err = np.max(np.abs(pt_logits - ort_out[0]))
    print(f"  Logits max_err={logits_err:.6e}, shape={ort_out[0].shape}")
    ok = logits_err < 1e-3
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Stage 5: Vocoder
# ---------------------------------------------------------------------------

def validate_vocoder(model, output_dir):
    """Compare vocoder: PyTorch vs ONNX."""
    print("\n=== Stage 5: Vocoder Validation ===")

    speech_tokenizer = model.speech_tokenizer
    decoder = speech_tokenizer.model.decoder
    num_q = decoder.config.num_quantizers

    torch.manual_seed(42)
    codes = torch.randint(0, 2048, (1, num_q, 5), dtype=torch.int64)

    with torch.no_grad():
        pt_wav = decoder(codes).numpy()

    onnx_path = os.path.join(output_dir, "vocoder.onnx")
    if not os.path.exists(onnx_path):
        print("  SKIP: vocoder.onnx not found")
        return True

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {"codes": codes.numpy()})
    ort_wav = ort_out[0]

    max_err = np.max(np.abs(pt_wav - ort_wav))
    if np.max(np.abs(pt_wav)) > 0:
        signal_power = np.mean(pt_wav ** 2)
        noise_power = np.mean((pt_wav - ort_wav) ** 2)
        snr = 10 * np.log10(signal_power / max(noise_power, 1e-30))
    else:
        snr = float("inf")

    print(f"  Waveform max_err={max_err:.6e}, SNR={snr:.1f} dB")
    print(f"  PT shape={pt_wav.shape}, ONNX shape={ort_wav.shape}")
    ok = max_err < 1e-2
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Stage 6: End-to-end greedy decode
# ---------------------------------------------------------------------------

def validate_end_to_end(model, output_dir):
    """End-to-end: PyTorch greedy decode vs ONNX greedy decode on same text."""
    print("\n=== Stage 6: End-to-End Validation ===")

    config = load_config(output_dir)
    emb = load_embeddings(output_dir)
    talker_cfg = model.config.talker_config
    cp_cfg = talker_cfg.code_predictor_config

    # Load ONNX sessions
    onnx_dir = output_dir
    required_files = ["talker_prefill.onnx", "talker_decode.onnx",
                      "code_predictor.onnx", "vocoder.onnx"]
    for f in required_files:
        if not os.path.exists(os.path.join(onnx_dir, f)):
            print(f"  SKIP: {f} not found")
            return True

    prefill_sess = ort.InferenceSession(os.path.join(onnx_dir, "talker_prefill.onnx"))
    decode_sess = ort.InferenceSession(os.path.join(onnx_dir, "talker_decode.onnx"))
    cp_sess = ort.InferenceSession(os.path.join(onnx_dir, "code_predictor.onnx"))
    vocoder_sess = ort.InferenceSession(os.path.join(onnx_dir, "vocoder.onnx"))

    # ---- Run PyTorch inference ----
    test_text = "Hello, this is a test."
    language = "english"
    speaker = list(talker_cfg.spk_id.keys())[0]  # first available speaker
    print(f"  Text: '{test_text}', language={language}, speaker={speaker}")

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(output_dir, "tokenizer")
    )

    # Format text as chat template
    chat_text = f"<|im_start|>assistant\n{test_text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(chat_text, add_special_tokens=False)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.int64)

    with torch.no_grad():
        # Greedy decode: top_k=1 means argmax
        pt_codes, pt_hidden = model.generate(
            input_ids=[input_ids_tensor],
            languages=[language],
            speakers=[speaker],
            max_new_tokens=20,  # short for validation
            do_sample=False,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            subtalker_dosample=False,
            subtalker_top_k=1,
            subtalker_top_p=1.0,
            subtalker_temperature=1.0,
            repetition_penalty=1.0,
        )
    pt_codes_arr = pt_codes[0].cpu().numpy()  # (num_steps, num_code_groups)
    print(f"  PyTorch generated {pt_codes_arr.shape[0]} frames")

    # ---- Run ONNX inference (same greedy decode) ----
    onnx_codes = _onnx_greedy_decode(
        model, config, emb,
        input_ids, language, speaker,
        prefill_sess, decode_sess, cp_sess,
        max_steps=20,
    )
    print(f"  ONNX generated {len(onnx_codes)} frames")

    # Compare code sequences
    min_len = min(len(onnx_codes), pt_codes_arr.shape[0])
    if min_len == 0:
        print("  WARNING: one or both produced 0 frames")
        return False

    onnx_codes_arr = np.array(onnx_codes[:min_len])
    pt_codes_trimmed = pt_codes_arr[:min_len]

    code_match = np.array_equal(onnx_codes_arr, pt_codes_trimmed)
    if not code_match:
        diffs = np.sum(onnx_codes_arr != pt_codes_trimmed)
        total = onnx_codes_arr.size
        print(f"  Code mismatch: {diffs}/{total} elements differ")
        # Show first difference
        for i in range(min_len):
            if not np.array_equal(onnx_codes_arr[i], pt_codes_trimmed[i]):
                print(f"    First diff at frame {i}: ONNX={onnx_codes_arr[i]}, PT={pt_codes_trimmed[i]}")
                break
    else:
        print(f"  Codes match perfectly ({min_len} frames)")

    # Decode both to audio via vocoder if codes exist
    if min_len > 0:
        num_groups = talker_cfg.num_code_groups
        # Vocoder expects (1, num_quantizers, T) where num_quantizers = num_groups
        pt_vocoder_input = pt_codes_trimmed.T[np.newaxis, :, :]  # (1, groups, T)
        onnx_vocoder_input = onnx_codes_arr.T[np.newaxis, :, :]

        pt_wav = vocoder_sess.run(None, {"codes": pt_vocoder_input.astype(np.int64)})[0]
        onnx_wav = vocoder_sess.run(None, {"codes": onnx_vocoder_input.astype(np.int64)})[0]

        # Save WAV files
        wav_dir = os.path.join(output_dir, "validation")
        os.makedirs(wav_dir, exist_ok=True)
        sf.write(os.path.join(wav_dir, "pytorch.wav"), pt_wav.flatten(), 24000)
        sf.write(os.path.join(wav_dir, "onnx.wav"), onnx_wav.flatten(), 24000)
        print(f"  Saved audio to {wav_dir}/")

        if code_match:
            # Audio should be identical since we used the same vocoder ONNX
            wav_err = np.max(np.abs(pt_wav - onnx_wav))
            print(f"  Audio max_err={wav_err:.6e} (same codes, same vocoder)")

    ok = code_match
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def _onnx_greedy_decode(
    model, config, emb,
    input_ids, language, speaker,
    prefill_sess, decode_sess, cp_sess,
    max_steps=20,
):
    """Run ONNX-based greedy decode, replicating the official inference flow."""
    talker_cfg = model.config.talker_config
    cp_cfg = talker_cfg.code_predictor_config
    num_layers = talker_cfg.num_hidden_layers
    num_kv_heads = talker_cfg.num_key_value_heads
    head_dim = talker_cfg.head_dim
    hidden_size = talker_cfg.hidden_size
    num_code_groups = talker_cfg.num_code_groups
    cp_num_layers = cp_cfg.num_hidden_layers
    cp_num_kv_heads = cp_cfg.num_key_value_heads
    cp_head_dim = cp_cfg.head_dim

    text_emb = emb["text_embedding"]
    fc1_w = emb["text_projection_fc1_weight"]
    fc1_b = emb["text_projection_fc1_bias"]
    fc2_w = emb["text_projection_fc2_weight"]
    fc2_b = emb["text_projection_fc2_bias"]
    codec_emb = emb["talker_codec_embedding"]
    cp_codec_embs = emb["cp_codec_embeddings"]

    def text_proj(token_ids):
        return text_project_numpy(token_ids, text_emb, fc1_w, fc1_b, fc2_w, fc2_b)

    # Build prefill embeddings
    # input_ids = [im_start, assistant, \n, text..., im_end, \n, im_start, assistant, \n]
    # Role prefix: first 3 tokens
    role_embed = text_proj(input_ids[:3])  # (3, hidden)

    # Codec prefix
    language_id = talker_cfg.codec_language_id.get(language.lower())
    if language_id is not None:
        codec_prefix_ids = [
            talker_cfg.codec_think_id,
            talker_cfg.codec_think_bos_id,
            language_id,
            talker_cfg.codec_think_eos_id,
        ]
    else:
        codec_prefix_ids = [
            talker_cfg.codec_nothink_id,
            talker_cfg.codec_think_bos_id,
            talker_cfg.codec_think_eos_id,
        ]

    # Speaker embed
    spk_id_tokens = talker_cfg.spk_id[speaker.lower()]
    speaker_embed = codec_emb[spk_id_tokens]  # (num_spk_tokens, hidden) or single token
    if len(speaker_embed.shape) == 1:
        speaker_embed = speaker_embed.reshape(1, -1)

    # Codec embeddings for prefix
    codec_prefix_embed = codec_emb[codec_prefix_ids]  # (N, hidden)

    # tts_pad, tts_bos, tts_eos embeddings
    tts_pad_embed = text_proj([config["tts_pad_token_id"]])[0]  # (hidden,)
    tts_bos_embed = text_proj([config["tts_bos_token_id"]])[0]
    tts_eos_embed = text_proj([config["tts_eos_token_id"]])[0]

    codec_pad_embed = codec_emb[talker_cfg.codec_pad_id]
    codec_bos_embed = codec_emb[talker_cfg.codec_bos_id]

    # Build the prefill sequence:
    # [role(3)] [tts_pad + codec_prefix] [tts_pad + speaker] [tts_bos + codec_pad] [text[3] + codec_bos]
    # Then for non-streaming: [text[4:-5] + codec_pad each] [tts_eos + codec_pad] [tts_pad + codec_bos]
    embeds_list = []

    # Role: text projection only
    embeds_list.append(role_embed)  # (3, hidden)

    # Codec prefix: tts_pad_embed + codec_embed for each
    for cid in codec_prefix_ids:
        embeds_list.append((tts_pad_embed + codec_emb[cid]).reshape(1, -1))

    # Speaker
    embeds_list.append((tts_pad_embed + speaker_embed.sum(axis=0)).reshape(1, -1))

    # Transition: tts_bos + codec_pad
    embeds_list.append((tts_bos_embed + codec_pad_embed).reshape(1, -1))

    # Non-streaming: all text tokens + tts_eos, each paired with codec_pad
    text_tokens = input_ids[3:-5]  # text content (between role markers)
    for tid in text_tokens:
        embeds_list.append(
            (text_proj([tid])[0] + codec_pad_embed).reshape(1, -1)
        )
    # tts_eos + codec_pad
    embeds_list.append((tts_eos_embed + codec_pad_embed).reshape(1, -1))

    # Final: tts_pad + codec_bos
    embeds_list.append((tts_pad_embed + codec_bos_embed).reshape(1, -1))

    prefill_embeds = np.concatenate(embeds_list, axis=0)  # (T, hidden)
    prefill_embeds = prefill_embeds[np.newaxis, :, :].astype(np.float32)  # (1, T, hidden)

    T = prefill_embeds.shape[1]
    attention_mask = np.ones((1, T), dtype=np.int64)
    position_ids = np.arange(T).reshape(1, 1, T).repeat(3, axis=0)  # (3, 1, T)

    # Run prefill
    prefill_out = prefill_sess.run(None, {
        "inputs_embeds": prefill_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    })

    logits = prefill_out[0]  # (1, T, vocab)
    hidden_states = prefill_out[1]  # (1, T, hidden)
    # Stack KV cache from per-layer outputs
    kv_outputs = prefill_out[2:]
    past_keys = np.stack([kv_outputs[i * 2] for i in range(num_layers)])  # (layers, 1, kv_heads, T, head_dim)
    past_values = np.stack([kv_outputs[i * 2 + 1] for i in range(num_layers)])

    # Trailing text hidden (for non-streaming, it's just tts_pad_embed repeated)
    trailing_hidden = tts_pad_embed.reshape(1, -1)

    # Decode loop
    all_codes = []
    current_pos = T
    codec_eos = talker_cfg.codec_eos_token_id

    for step in range(max_steps):
        # Greedy: argmax on last position logits
        last_logits = logits[0, -1, :]  # (vocab,)
        group0_token = int(np.argmax(last_logits))

        if group0_token == codec_eos:
            break

        # Run code predictor for groups 1-15
        frame_codes = [group0_token]
        talker_hidden = hidden_states[0, -1:, :]  # (1, hidden)
        group0_embed = codec_emb[group0_token].reshape(1, -1)  # (1, hidden)

        # CP initial input: [talker_hidden, group0_embed] in talker hidden space
        cp_input = np.concatenate([talker_hidden, group0_embed], axis=0)  # (2, talker_hidden)
        cp_input = cp_input[np.newaxis, :, :].astype(np.float32)  # (1, 2, talker_hidden)

        cp_past_keys = np.zeros((cp_num_layers, 1, cp_num_kv_heads, 0, cp_head_dim), dtype=np.float32)
        cp_past_values = np.zeros((cp_num_layers, 1, cp_num_kv_heads, 0, cp_head_dim), dtype=np.float32)

        for g in range(num_code_groups - 1):
            cp_out = cp_sess.run(None, {
                "inputs_embeds": cp_input,
                "generation_steps": np.array([g], dtype=np.int64),
                "past_keys": cp_past_keys,
                "past_values": cp_past_values,
            })

            cp_logits = cp_out[0]  # (1, S, vocab)
            cp_past_keys = cp_out[1]
            cp_past_values = cp_out[2]

            # Greedy: argmax on last position
            token = int(np.argmax(cp_logits[0, -1, :]))
            frame_codes.append(token)

            # Next CP input: codec embed for this group (in talker hidden space)
            cp_embed = cp_codec_embs[g][token].reshape(1, 1, -1).astype(np.float32)
            cp_input = cp_embed

        all_codes.append(frame_codes)

        # Build next talker input: sum of all codec embeddings + trailing text
        next_embed = codec_emb[group0_token].copy()
        for g in range(num_code_groups - 1):
            next_embed = next_embed + cp_codec_embs[g][frame_codes[g + 1]]
        next_embed = next_embed + trailing_hidden[0]
        next_embed = next_embed.reshape(1, 1, -1).astype(np.float32)

        # Decode step
        decode_mask = np.ones((1, current_pos + 1), dtype=np.int64)
        decode_pos = np.array([[[current_pos]]]).repeat(3, axis=0)  # (3, 1, 1)

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

    return all_codes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate Qwen3-TTS ONNX exports")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--onnx-dir", default="./output/qwen3-tts-1.7b-voicedesign")
    parser.add_argument("--stages", default="1,2,3,4,5,6", help="Comma-separated stage numbers")
    args = parser.parse_args()

    stages = set(int(s) for s in args.stages.split(","))

    print(f"Loading PyTorch model: {args.model_id}")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    results = {}

    if 1 in stages:
        results[1] = validate_embeddings(model, args.onnx_dir)
    if 2 in stages:
        results[2] = validate_talker_prefill(model, args.onnx_dir)
    if 3 in stages:
        results[3] = validate_talker_decode(model, args.onnx_dir)
    if 4 in stages:
        results[4] = validate_code_predictor(model, args.onnx_dir)
    if 5 in stages:
        results[5] = validate_vocoder(model, args.onnx_dir)
    if 6 in stages:
        results[6] = validate_end_to_end(model, args.onnx_dir)

    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    all_pass = True
    stage_names = {
        1: "Embeddings", 2: "Talker Prefill", 3: "Talker Decode",
        4: "Code Predictor", 5: "Vocoder", 6: "End-to-End",
    }
    for stage_num in sorted(results.keys()):
        status = "PASS" if results[stage_num] else "FAIL"
        if not results[stage_num]:
            all_pass = False
        print(f"  Stage {stage_num} ({stage_names[stage_num]}): {status}")

    if all_pass:
        print("\nAll stages passed!")
    else:
        print("\nSome stages FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
