#!/usr/bin/env python3
"""Validate ONNX exports against PyTorch output for Qwen3-TTS.

Runs per-stage comparisons and an end-to-end greedy decode test.

Memory-efficient: generates all PyTorch reference outputs first, then unloads
the model before loading ONNX sessions for comparison.  Peak memory stays under
~7 GB (one model at a time) so validation runs on standard GitHub Actions runners.
"""

import argparse
import gc
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


def _extract_model_cfg(model):
    """Extract all config scalars needed for ONNX-only inference."""
    tc = model.config.talker_config
    cc = tc.code_predictor_config
    return {
        "num_hidden_layers": tc.num_hidden_layers,
        "num_key_value_heads": tc.num_key_value_heads,
        "head_dim": tc.head_dim,
        "hidden_size": tc.hidden_size,
        "num_code_groups": tc.num_code_groups,
        "cp_num_hidden_layers": cc.num_hidden_layers,
        "cp_num_key_value_heads": cc.num_key_value_heads,
        "cp_head_dim": cc.head_dim,
        "codec_language_id": dict(tc.codec_language_id) if tc.codec_language_id else {},
        "codec_think_id": tc.codec_think_id,
        "codec_think_bos_id": tc.codec_think_bos_id,
        "codec_think_eos_id": tc.codec_think_eos_id,
        "codec_nothink_id": tc.codec_nothink_id,
        "spk_id": dict(tc.spk_id) if tc.spk_id else {},
        "codec_pad_id": tc.codec_pad_id,
        "codec_bos_id": tc.codec_bos_id,
        "codec_eos_token_id": tc.codec_eos_token_id,
        "vocab_size": tc.vocab_size,
    }


# ---------------------------------------------------------------------------
# Stage 1: Embeddings (no ONNX — runs entirely during PyTorch phase)
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

def pytorch_talker_prefill(model):
    """Run PyTorch talker prefill and return reference data."""
    print("\n  Stage 2: generating PyTorch reference (talker prefill)...")
    cfg = model.config.talker_config
    hidden_size = cfg.hidden_size

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

    return {
        "pt_logits": pt_logits,
        "pt_hidden": pt_hidden,
        "inputs_embeds": inputs_embeds.numpy(),
        "attention_mask": attention_mask.numpy(),
        "position_ids": position_ids.numpy(),
    }


def compare_talker_prefill(refs, onnx_dir):
    """Compare talker prefill: saved PyTorch refs vs ONNX."""
    print("\n=== Stage 2: Talker Prefill Validation ===")
    onnx_path = os.path.join(onnx_dir, "talker_prefill.onnx")
    if not os.path.exists(onnx_path):
        print("  SKIP: talker_prefill.onnx not found")
        return True

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {
        "inputs_embeds": refs["inputs_embeds"],
        "attention_mask": refs["attention_mask"],
        "position_ids": refs["position_ids"],
    })
    del sess
    gc.collect()

    logits_err = np.max(np.abs(refs["pt_logits"] - ort_out[0]))
    hidden_err = np.max(np.abs(refs["pt_hidden"] - ort_out[1]))
    print(f"  Logits max_err={logits_err:.6e}")
    print(f"  Hidden max_err={hidden_err:.6e}")
    ok = logits_err < 1e-3 and hidden_err < 1e-3
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Stage 3: Talker Decode
# ---------------------------------------------------------------------------

def pytorch_talker_decode(model):
    """Run PyTorch talker decode and return reference data."""
    print("  Stage 3: generating PyTorch reference (talker decode)...")
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

    # Build stacked KV for ONNX (new Cache API: cache[i] -> (key, value))
    past_keys_np = np.stack([past_kv[i][0].numpy() for i in range(num_layers)])
    past_values_np = np.stack([past_kv[i][1].numpy() for i in range(num_layers)])

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

    return {
        "pt_logits": pt_logits,
        "pt_hidden": pt_hidden,
        "decode_embeds": decode_embeds.numpy(),
        "decode_mask": decode_mask.numpy(),
        "decode_pos": decode_pos.numpy(),
        "past_keys": past_keys_np,
        "past_values": past_values_np,
    }


def compare_talker_decode(refs, onnx_dir):
    """Compare single-step talker decode: saved PyTorch refs vs ONNX."""
    print("\n=== Stage 3: Talker Decode Validation ===")
    onnx_path = os.path.join(onnx_dir, "talker_decode.onnx")
    if not os.path.exists(onnx_path):
        print("  SKIP: talker_decode.onnx not found")
        return True

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {
        "inputs_embeds": refs["decode_embeds"],
        "attention_mask": refs["decode_mask"],
        "position_ids": refs["decode_pos"],
        "past_keys": refs["past_keys"],
        "past_values": refs["past_values"],
    })
    del sess
    gc.collect()

    logits_err = np.max(np.abs(refs["pt_logits"] - ort_out[0]))
    hidden_err = np.max(np.abs(refs["pt_hidden"] - ort_out[1]))
    print(f"  Logits max_err={logits_err:.6e}")
    print(f"  Hidden max_err={hidden_err:.6e}")
    ok = logits_err < 1e-3 and hidden_err < 1e-3
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Stage 4: Code Predictor
# ---------------------------------------------------------------------------

def pytorch_code_predictor(model):
    """Run PyTorch code predictor and return reference data."""
    print("  Stage 4: generating PyTorch reference (code predictor)...")
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
        pt_logits = cp.lm_head[0](pt_hidden).numpy()  # group 0 -> lm_head[0]

    return {
        "pt_logits": pt_logits,
        "inputs_embeds": inputs_embeds.numpy(),
        "cp_num_layers": cp_cfg.num_hidden_layers,
        "cp_num_kv_heads": cp_cfg.num_key_value_heads,
        "cp_head_dim": cp_cfg.head_dim,
    }


def compare_code_predictor(refs, onnx_dir):
    """Compare code predictor: saved PyTorch refs vs ONNX."""
    print("\n=== Stage 4: Code Predictor Validation ===")
    onnx_path = os.path.join(onnx_dir, "code_predictor.onnx")
    if not os.path.exists(onnx_path):
        print("  SKIP: code_predictor.onnx not found")
        return True

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {
        "inputs_embeds": refs["inputs_embeds"],
        "generation_steps": np.array([0], dtype=np.int64),
        "past_keys": np.zeros(
            (refs["cp_num_layers"], 1, refs["cp_num_kv_heads"], 0, refs["cp_head_dim"]),
            dtype=np.float32,
        ),
        "past_values": np.zeros(
            (refs["cp_num_layers"], 1, refs["cp_num_kv_heads"], 0, refs["cp_head_dim"]),
            dtype=np.float32,
        ),
    })
    del sess
    gc.collect()

    logits_err = np.max(np.abs(refs["pt_logits"] - ort_out[0]))
    print(f"  Logits max_err={logits_err:.6e}, shape={ort_out[0].shape}")
    ok = logits_err < 1e-3
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Stage 5: Vocoder
# ---------------------------------------------------------------------------

def pytorch_vocoder(model):
    """Run PyTorch vocoder and return reference data."""
    print("  Stage 5: generating PyTorch reference (vocoder)...")
    speech_tokenizer = model.speech_tokenizer
    decoder = speech_tokenizer.model.decoder
    num_q = decoder.config.num_quantizers

    test_cases = []
    for T in [50, 150, 299]:
        torch.manual_seed(42 + T)
        codes = torch.randint(0, 2048, (1, num_q, T), dtype=torch.int64)

        with torch.no_grad():
            pt_wav = decoder(codes).cpu().numpy()

        test_cases.append({
            "T": T,
            "codes": codes.numpy(),
            "pt_wav": pt_wav,
        })

    return {"test_cases": test_cases}


def compare_vocoder(refs, onnx_dir):
    """Compare vocoder: saved PyTorch refs vs ONNX at multiple sequence lengths."""
    print("\n=== Stage 5: Vocoder Validation ===")
    onnx_path = os.path.join(onnx_dir, "vocoder.onnx")
    if not os.path.exists(onnx_path):
        print("  SKIP: vocoder.onnx not found")
        return True

    sess = ort.InferenceSession(onnx_path)
    ok = True

    for tc in refs["test_cases"]:
        T = tc["T"]
        pt_wav = tc["pt_wav"]

        ort_out = sess.run(None, {"codes": tc["codes"]})
        ort_wav = ort_out[0]

        max_err = np.max(np.abs(pt_wav - ort_wav))
        if np.max(np.abs(pt_wav)) > 0:
            signal_power = np.mean(pt_wav ** 2)
            noise_power = np.mean((pt_wav - ort_wav) ** 2)
            snr = 10 * np.log10(signal_power / max(noise_power, 1e-30))
        else:
            snr = float("inf")

        t_ok = max_err < 1e-2
        print(f"  T={T}: max_err={max_err:.6e}, SNR={snr:.1f} dB, "
              f"shape={ort_wav.shape} {'PASS' if t_ok else 'FAIL'}")
        ok = ok and t_ok

    del sess
    gc.collect()

    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Stage 6: End-to-end greedy decode
# ---------------------------------------------------------------------------

def pytorch_end_to_end(model, root_dir):
    """Run PyTorch end-to-end inference and return reference data."""
    print("  Stage 6: generating PyTorch reference (end-to-end)...")
    talker_cfg = model.config.talker_config

    test_text = ("The sun rose slowly over the mountains, casting long golden shadows "
                 "across the valley below. Birds began to sing in the tall pine trees, "
                 "and a gentle breeze carried the scent of wildflowers through the crisp morning air.")
    language = "english"
    if talker_cfg.spk_id:
        speaker = list(talker_cfg.spk_id.keys())[0]
    else:
        speaker = None
    print(f"  Text: '{test_text}', language={language}, speaker={speaker}")

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(root_dir, "tokenizer")
    )
    chat_text = f"<|im_start|>assistant\n{test_text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(chat_text, add_special_tokens=False)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.int64)

    max_frames = 300

    # Greedy decode (deterministic — for ONNX comparison)
    with torch.no_grad():
        pt_codes, pt_hidden = model.generate(
            input_ids=[input_ids_tensor],
            languages=[language],
            speakers=[speaker],
            non_streaming_mode=True,
            max_new_tokens=max_frames,
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
    pt_codes_arr = pt_codes[0].cpu().numpy()
    print(f"  PyTorch greedy: {pt_codes_arr.shape[0]} frames")

    # Sampled decode (for clean listening sample)
    print("  Generating clean audio sample (native sampling config)...")
    with torch.no_grad():
        sample_codes, _ = model.generate(
            input_ids=[input_ids_tensor],
            languages=[language],
            speakers=[speaker],
            non_streaming_mode=True,
            max_new_tokens=max_frames,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
            repetition_penalty=1.05,
        )
    sample_codes_arr = sample_codes[0].cpu().numpy()
    print(f"  PyTorch sampled: {sample_codes_arr.shape[0]} frames")

    return {
        "pt_codes_arr": pt_codes_arr,
        "sample_codes_arr": sample_codes_arr,
        "model_cfg": _extract_model_cfg(model),
        "test_text": test_text,
        "language": language,
        "speaker": speaker,
        "input_ids": input_ids,
        "max_frames": max_frames,
    }


def compare_end_to_end(refs, root_dir, onnx_dir):
    """Compare end-to-end: saved PyTorch refs vs ONNX greedy decode."""
    print("\n=== Stage 6: End-to-End Validation ===")

    config = load_config(root_dir)
    emb = load_embeddings(root_dir)

    required_files = ["talker_prefill.onnx", "talker_decode.onnx",
                      "code_predictor.onnx", "vocoder.onnx"]
    for f in required_files:
        if not os.path.exists(os.path.join(onnx_dir, f)):
            print(f"  SKIP: {f} not found")
            return True

    pt_codes_arr = refs["pt_codes_arr"]

    # Run ONNX greedy decode (manages sessions internally for memory)
    onnx_codes = _onnx_greedy_decode(
        refs["model_cfg"], config, emb,
        refs["input_ids"], refs["language"], refs["speaker"],
        onnx_dir,
        max_steps=refs["max_frames"],
        repetition_penalty=1.0,
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
    len_match = len(onnx_codes) == pt_codes_arr.shape[0]

    # Find first divergence frame
    first_diff_frame = min_len
    for i in range(min_len):
        if not np.array_equal(onnx_codes_arr[i], pt_codes_trimmed[i]):
            first_diff_frame = i
            break

    if not code_match:
        diffs = np.sum(onnx_codes_arr != pt_codes_trimmed)
        total = onnx_codes_arr.size
        print(f"  Codes match for first {first_diff_frame}/{min_len} frames, "
              f"then {diffs}/{total} elements differ")
        print(f"  (float32 accumulation causes divergence in autoregressive decode)")
    else:
        print(f"  Codes match perfectly ({min_len} frames, "
              f"len_match={'yes' if len_match else 'no (%d vs %d)' % (len(onnx_codes), pt_codes_arr.shape[0])})")

    # Decode to audio using ONNX vocoder (loaded separately to save memory)
    wav_dir = os.path.join(root_dir, "validation")
    os.makedirs(wav_dir, exist_ok=True)

    vocoder_sess = ort.InferenceSession(os.path.join(onnx_dir, "vocoder.onnx"))

    if min_len > 0:
        pt_vocoder_input = pt_codes_trimmed.T[np.newaxis, :, :].astype(np.int64)
        onnx_vocoder_input = onnx_codes_arr.T[np.newaxis, :, :].astype(np.int64)

        pt_wav = vocoder_sess.run(None, {"codes": pt_vocoder_input})[0].flatten()
        onnx_wav = vocoder_sess.run(None, {"codes": onnx_vocoder_input})[0].flatten()

        sf.write(os.path.join(wav_dir, "pytorch_e2e.wav"), pt_wav, 24000)
        sf.write(os.path.join(wav_dir, "onnx_e2e.wav"), onnx_wav, 24000)
        print(f"  Saved audio to {wav_dir}/ ({len(pt_wav)/24000:.1f}s)")

        if code_match:
            wav_err = np.max(np.abs(pt_wav - onnx_wav))
            print(f"  Audio max_err={wav_err:.6e} (same codes -> identical audio)")
        else:
            signal_power = np.mean(pt_wav ** 2)
            noise_power = np.mean((pt_wav - onnx_wav) ** 2)
            snr = 10 * np.log10(signal_power / max(noise_power, 1e-30))
            print(f"  Audio SNR={snr:.1f} dB (different codes)")

    # With rep_penalty=1.0 and greedy decode, codes should match exactly
    # (or differ by at most 1 frame at EOS boundary).
    len_diff = abs(len(onnx_codes) - pt_codes_arr.shape[0])
    ok = code_match and len_diff <= 1
    if not len_match and code_match:
        print(f"  Length diff={len_diff} (EOS boundary sensitivity)")
    print(f"  {'PASS' if ok else 'FAIL'}")

    # Decode the clean sample with ONNX vocoder
    sample_codes_arr = refs["sample_codes_arr"]
    sample_input = sample_codes_arr.T[np.newaxis, :, :].astype(np.int64)
    sample_wav = vocoder_sess.run(None, {"codes": sample_input})[0].flatten()
    sf.write(os.path.join(wav_dir, "sample.wav"), sample_wav, 24000)
    print(f"  Saved sample.wav ({len(sample_wav)/24000:.1f}s, {sample_codes_arr.shape[0]} frames)")

    del vocoder_sess
    gc.collect()

    return ok


def _onnx_greedy_decode(
    model_cfg, config, emb,
    input_ids, language, speaker,
    onnx_dir,
    max_steps=20,
    repetition_penalty=1.0,
):
    """Run ONNX-based greedy decode, replicating the official inference flow.

    Loads ONNX sessions sequentially to keep peak memory low:
    prefill session -> unload -> decode + CP sessions -> unload.
    """
    num_layers = model_cfg["num_hidden_layers"]
    num_kv_heads = model_cfg["num_key_value_heads"]
    head_dim = model_cfg["head_dim"]
    hidden_size = model_cfg["hidden_size"]
    num_code_groups = model_cfg["num_code_groups"]
    cp_num_layers = model_cfg["cp_num_hidden_layers"]
    cp_num_kv_heads = model_cfg["cp_num_key_value_heads"]
    cp_head_dim = model_cfg["cp_head_dim"]

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
    role_embed = text_proj(input_ids[:3])

    language_id = model_cfg["codec_language_id"].get(language.lower())
    if language_id is not None:
        codec_prefix_ids = [
            model_cfg["codec_think_id"],
            model_cfg["codec_think_bos_id"],
            language_id,
            model_cfg["codec_think_eos_id"],
        ]
    else:
        codec_prefix_ids = [
            model_cfg["codec_nothink_id"],
            model_cfg["codec_think_bos_id"],
            model_cfg["codec_think_eos_id"],
        ]

    speaker_embed = None
    if speaker is not None and speaker != "":
        spk_id_tokens = model_cfg["spk_id"][speaker.lower()]
        speaker_embed = codec_emb[spk_id_tokens]
        if len(speaker_embed.shape) == 1:
            speaker_embed = speaker_embed.reshape(1, -1)

    tts_pad_embed = text_proj([config["tts_pad_token_id"]])[0]
    tts_bos_embed = text_proj([config["tts_bos_token_id"]])[0]
    tts_eos_embed = text_proj([config["tts_eos_token_id"]])[0]

    codec_pad_embed = codec_emb[model_cfg["codec_pad_id"]]
    codec_bos_embed = codec_emb[model_cfg["codec_bos_id"]]

    # Build the prefill sequence (non-streaming mode)
    embeds_list = []

    # Role: text projection only
    embeds_list.append(role_embed)

    # Codec prefix
    for cid in codec_prefix_ids:
        embeds_list.append((tts_pad_embed + codec_emb[cid]).reshape(1, -1))

    # Speaker
    if speaker_embed is not None:
        embeds_list.append((tts_pad_embed + speaker_embed.sum(axis=0)).reshape(1, -1))

    # Transition: tts_bos + codec_pad
    embeds_list.append((tts_bos_embed + codec_pad_embed).reshape(1, -1))

    # Text tokens + tts_eos
    text_tokens = input_ids[3:-5]
    for tid in text_tokens:
        embeds_list.append(
            (text_proj([tid])[0] + codec_pad_embed).reshape(1, -1)
        )
    embeds_list.append((tts_eos_embed + codec_pad_embed).reshape(1, -1))

    # Final: tts_pad + codec_bos
    embeds_list.append((tts_pad_embed + codec_bos_embed).reshape(1, -1))

    prefill_embeds = np.concatenate(embeds_list, axis=0)
    prefill_embeds = prefill_embeds[np.newaxis, :, :].astype(np.float32)

    T = prefill_embeds.shape[1]
    attention_mask = np.ones((1, T), dtype=np.int64)
    position_ids = np.arange(T).reshape(1, 1, T).repeat(3, axis=0)

    # --- Prefill phase: load session, run, unload ---
    prefill_sess = ort.InferenceSession(os.path.join(onnx_dir, "talker_prefill.onnx"))
    prefill_out = prefill_sess.run(None, {
        "inputs_embeds": prefill_embeds,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    })
    del prefill_sess
    gc.collect()

    logits = prefill_out[0]
    hidden_states = prefill_out[1]
    kv_outputs = prefill_out[2:]
    past_keys = np.stack([kv_outputs[i * 2] for i in range(num_layers)])
    past_values = np.stack([kv_outputs[i * 2 + 1] for i in range(num_layers)])

    trailing_hidden = tts_pad_embed.reshape(1, -1)

    # --- Decode phase: load decode + CP sessions ---
    decode_sess = ort.InferenceSession(os.path.join(onnx_dir, "talker_decode.onnx"))
    cp_sess = ort.InferenceSession(os.path.join(onnx_dir, "code_predictor.onnx"))

    all_codes = []
    current_pos = T
    codec_eos = model_cfg["codec_eos_token_id"]
    vocab_size = model_cfg["vocab_size"]

    suppress_mask = np.zeros(vocab_size, dtype=bool)
    suppress_mask[vocab_size - 1024:vocab_size] = True
    suppress_mask[codec_eos] = False

    generated_tokens = []

    for step in range(max_steps):
        last_logits = logits[0, -1, :].copy()
        last_logits[suppress_mask] = -np.inf
        if step < 2:
            last_logits[codec_eos] = -np.inf
        if repetition_penalty != 1.0 and generated_tokens:
            seen = np.array(generated_tokens)
            scores = last_logits[seen]
            scores = np.where(scores > 0, scores / repetition_penalty,
                              scores * repetition_penalty)
            last_logits[seen] = scores
        group0_token = int(np.argmax(last_logits))

        if group0_token == codec_eos:
            break

        generated_tokens.append(group0_token)

        # Run code predictor for groups 1-15
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

            token = int(np.argmax(cp_logits[0, -1, :]))
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

    del decode_sess, cp_sess
    gc.collect()

    return all_codes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate Qwen3-TTS ONNX exports")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--onnx-dir", default="./output/qwen3-tts-1.7b-voicedesign")
    parser.add_argument("--variant", default="fp32", help="ONNX variant subfolder (default: fp32)")
    parser.add_argument("--stages", default="1,2,3,4,5,6", help="Comma-separated stage numbers")
    args = parser.parse_args()

    stages = set(int(s) for s in args.stages.split(","))

    root_dir = args.onnx_dir
    onnx_subdir = os.path.join(root_dir, args.variant)

    results = {}
    refs = {}

    # ===================================================================
    # Phase 1: Load PyTorch model, generate all reference outputs
    # ===================================================================
    print(f"Loading PyTorch model: {args.model_id}")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    if 1 in stages:
        results[1] = validate_embeddings(model, root_dir)
    if 2 in stages:
        refs[2] = pytorch_talker_prefill(model)
    if 3 in stages:
        refs[3] = pytorch_talker_decode(model)
    if 4 in stages:
        refs[4] = pytorch_code_predictor(model)
    if 5 in stages:
        refs[5] = pytorch_vocoder(model)
    if 6 in stages:
        refs[6] = pytorch_end_to_end(model, root_dir)

    # ===================================================================
    # Unload PyTorch model — reclaim ~6.8 GB
    # ===================================================================
    print("\nUnloading PyTorch model to free memory...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ===================================================================
    # Phase 2: Load ONNX sessions one at a time, compare against refs
    # ===================================================================
    if 2 in stages:
        results[2] = compare_talker_prefill(refs[2], onnx_subdir)
    if 3 in stages:
        results[3] = compare_talker_decode(refs[3], onnx_subdir)
    if 4 in stages:
        results[4] = compare_code_predictor(refs[4], onnx_subdir)
    if 5 in stages:
        results[5] = compare_vocoder(refs[5], onnx_subdir)
    if 6 in stages:
        results[6] = compare_end_to_end(refs[6], root_dir, onnx_subdir)

    # ===================================================================
    # Summary
    # ===================================================================
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
