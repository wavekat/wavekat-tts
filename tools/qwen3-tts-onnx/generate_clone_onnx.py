#!/usr/bin/env python3
"""Generate WAV with voice cloning using ONNX models (no PyTorch at inference).

This is the end-to-end verification script for the 0.6B Base voice-clone ONNX
pipeline.  It chains all 6 exported models:

  1. tokenizer_encoder.onnx  — ref audio → ref codes (16 groups, 12 Hz)
  2. speaker_encoder.onnx    — ref mel   → speaker embedding (1024-d)
  3. talker_prefill.onnx     — ICL prefill → logits + KV cache
  4. talker_decode.onnx      — single-step decode loop
  5. code_predictor.onnx     — groups 1-15 per frame
  6. vocoder.onnx            — codes → 24 kHz waveform

Examples:
  python generate_clone_onnx.py \\
    --ref-audio clone.wav --ref-text "Okay. Yeah. I resent you." \\
    --text "Give every small business the voice of a big one." \\
    -o cloned.wav

  python generate_clone_onnx.py --variant int4 \\
    --ref-audio clone.wav --ref-text "Okay. Yeah. I resent you." \\
    --text "让每一家小企业，都拥有大企业的声音。" --lang chinese \\
    -o cloned_zh.wav
"""

import argparse
import json
import os
import time

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import AutoTokenizer

# Mel-spectrogram parameters (match the reference HiFi-GAN config).
MEL_SR = 24000
MEL_N_FFT = 1024
MEL_HOP = 256
MEL_WIN = 1024
MEL_N_MELS = 128
MEL_FMIN = 0
MEL_FMAX = 12000

# Tokenizer encoder canonical input length (must match export_tokenizer_encoder.py).
TOKENIZER_CANONICAL_SAMPLES = 10 * MEL_SR  # 240_000


# ---------------------------------------------------------------------------
# Audio / mel helpers
# ---------------------------------------------------------------------------

def load_ref_audio(path: str, target_sr: int = MEL_SR) -> np.ndarray:
    """Load + resample reference audio to mono float32 at target_sr."""
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32)


def compute_mel(audio: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram matching the Qwen3-TTS reference.

    Returns (1, T_mel, 128) float32.
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=MEL_SR, n_fft=MEL_N_FFT, hop_length=MEL_HOP,
        win_length=MEL_WIN, n_mels=MEL_N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX,
        center=False,
    )
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    # (n_mels, frames) → (1, frames, n_mels)
    return mel.T[np.newaxis, :, :].astype(np.float32)


def pad_for_tokenizer_encoder(audio: np.ndarray) -> tuple[np.ndarray, int]:
    """Zero-pad / truncate to the canonical sample count.

    Returns (padded_waveform (1, CANONICAL), original_sample_count).
    """
    n = len(audio)
    if n > TOKENIZER_CANONICAL_SAMPLES:
        audio = audio[:TOKENIZER_CANONICAL_SAMPLES]
        n = TOKENIZER_CANONICAL_SAMPLES
    padded = np.zeros(TOKENIZER_CANONICAL_SAMPLES, dtype=np.float32)
    padded[:n] = audio
    return padded[np.newaxis, :], n


# ---------------------------------------------------------------------------
# Embedding / sampling helpers (copied from generate_onnx.py)
# ---------------------------------------------------------------------------

def text_project_numpy(token_ids, text_emb, fc1_w, fc1_b, fc2_w, fc2_b):
    embeds = text_emb[token_ids]
    hidden = embeds @ fc1_w.T + fc1_b
    activated = hidden * (1.0 / (1.0 + np.exp(-hidden)))
    return activated @ fc2_w.T + fc2_b


def load_embeddings(onnx_dir):
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


def sample_top_k(logits, top_k, temperature):
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0 and top_k < len(logits):
        idx = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[idx] = logits[idx]
        logits = mask
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_clone_onnx(
    model_dir: str,
    variant: str,
    text: str,
    ref_audio_path: str,
    ref_text: str,
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

    # ------------------------------------------------------------------
    # 1. Load ONNX sessions
    # ------------------------------------------------------------------
    print(f"Loading ONNX models ({variant}) ...")
    prefill_sess = ort.InferenceSession(os.path.join(onnx_dir, "talker_prefill.onnx"))
    decode_sess = ort.InferenceSession(os.path.join(onnx_dir, "talker_decode.onnx"))
    cp_sess = ort.InferenceSession(os.path.join(onnx_dir, "code_predictor.onnx"))
    vocoder_sess = ort.InferenceSession(os.path.join(onnx_dir, "vocoder.onnx"))
    spk_sess = ort.InferenceSession(os.path.join(model_dir, "speaker_encoder.onnx"))
    tok_enc_sess = ort.InferenceSession(os.path.join(model_dir, "tokenizer_encoder.onnx"))

    # ------------------------------------------------------------------
    # 2. Precompute embedding accessors
    # ------------------------------------------------------------------
    text_emb = emb["text_embedding"]
    fc1_w = emb["text_projection_fc1_weight"]
    fc1_b = emb["text_projection_fc1_bias"]
    fc2_w = emb["text_projection_fc2_weight"]
    fc2_b = emb["text_projection_fc2_bias"]
    codec_emb = emb["talker_codec_embedding"]
    cp_codec_embs = emb["cp_codec_embeddings"]
    hidden_size = config["talker_hidden_size"]
    num_code_groups = config["talker_num_code_groups"]

    def text_proj(token_ids):
        return text_project_numpy(token_ids, text_emb, fc1_w, fc1_b, fc2_w, fc2_b)

    # ------------------------------------------------------------------
    # 3. Encode reference audio
    # ------------------------------------------------------------------
    print(f"  Ref audio: {ref_audio_path}")
    ref_audio = load_ref_audio(ref_audio_path)
    ref_duration = len(ref_audio) / MEL_SR
    print(f"  Ref duration: {ref_duration:.1f}s ({len(ref_audio)} samples)")

    # Speaker embedding
    mel = compute_mel(ref_audio)
    print(f"  Mel shape: {mel.shape}")
    spk_embed = spk_sess.run(None, {"mels": mel})[0]  # (1, 1024)
    print(f"  Speaker embedding: {spk_embed.shape}")

    # Reference codes
    padded_wav, orig_samples = pad_for_tokenizer_encoder(ref_audio)
    ref_codes_full = tok_enc_sess.run(None, {"waveform": padded_wav})[0]  # (1, 16, 125)
    # Trim to actual frames
    encode_downsample = 1920
    actual_frames = int(np.ceil(orig_samples / encode_downsample))
    ref_codes = ref_codes_full[:, :, :actual_frames]  # (1, 16, actual_frames)
    print(f"  Ref codes: {ref_codes.shape} (trimmed from {ref_codes_full.shape[2]})")

    # ------------------------------------------------------------------
    # 4. Tokenize text & ref_text
    # ------------------------------------------------------------------
    chat_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(chat_text, add_special_tokens=False)

    ref_chat = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
    ref_ids = tokenizer.encode(ref_chat, add_special_tokens=False)

    print(f"  Text: '{text}' ({len(input_ids)} tokens)")
    print(f"  Ref text: '{ref_text}' ({len(ref_ids)} tokens)")
    print(f"  Language: {language}")

    # ------------------------------------------------------------------
    # 5. Build prefill embeddings (ICL, non-streaming)
    # ------------------------------------------------------------------
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

    tts_pad_embed = text_proj([config["tts_pad_token_id"]])[0]   # (hidden,)
    tts_bos_embed = text_proj([config["tts_bos_token_id"]])[0]
    tts_eos_embed = text_proj([config["tts_eos_token_id"]])[0]
    codec_pad_id = config["codec_pad_id"]
    codec_bos_id = config["codec_bos_id"]
    codec_pad_embed = codec_emb[codec_pad_id]
    codec_bos_embed = codec_emb[codec_bos_id]

    embeds_list = []

    # 5a. Role prefix: first 3 tokens (text proj only)
    role_embed = text_proj(input_ids[:3])  # (3, hidden)
    embeds_list.append(role_embed)

    # 5b. Codec prefix: tts_pad + codec_embed for each prefix token
    for cid in codec_prefix_ids:
        embeds_list.append((tts_pad_embed + codec_emb[cid]).reshape(1, -1))

    # 5c. Speaker slot: speaker_embed (1024-d projected into hidden_size via sum)
    # The reference code does: speaker_embed.view(1, 1, -1) in the codec embedding position
    # Since enc_dim == hidden_size for 0.6B (both 1024), it's direct.
    embeds_list.append(spk_embed.reshape(1, -1))

    # 5d. Transition: codec_pad + codec_bos (tts_pad + tts_bos on the text side)
    embeds_list.append((tts_pad_embed + codec_pad_embed).reshape(1, -1))
    # Note: the reference builds this as:
    #   tts_pad * (num_codec_prefix + 1 speaker) | tts_bos  +  codec_prefix[:-1] | codec[last]
    # But we already laid out codec prefix above. The last two before ICL are:
    #   tts_bos + codec_pad  (transition into text/codec interleave)
    # Let me re-derive from the reference code...

    # Actually, let me redo this more carefully following the reference:
    # The reference builds:
    #   _talker_input_embed = tts_pad.expand(N-2) | tts_bos  + codec_input_embedding[:, :-1]
    # where codec_input_embedding = [codec_prefix(think,think_bos,lang,think_eos), speaker, codec_pad, codec_bos]
    # and [:, :-1] means all except codec_bos
    # Then the first text token is paired with codec_bos separately outside of ICL path.
    # But in ICL mode, the text path is different...

    # Let me restart the prefill construction cleanly.
    embeds_list = []

    # Part A: Role prefix (3 tokens, text_proj only)
    role_embed = text_proj(input_ids[:3])  # (3, hidden)
    embeds_list.append(role_embed)

    # Part B: Codec prefix
    # codec_input_embedding = [think, think_bos, lang, think_eos, speaker, codec_pad, codec_bos]
    #                                                              ^speaker slot
    # _talker_input_embed = concat(tts_pad × (len-2), tts_bos) + codec_input_embedding[:-1]
    # So text side: [tts_pad, tts_pad, tts_pad, tts_pad, tts_pad, tts_bos]
    # codec side:   [think,   think_bos, lang,  think_eos, speaker, codec_pad]
    # (codec_bos is the last element, used separately)

    codec_full = list(codec_prefix_ids)  # [think, think_bos, lang, think_eos]
    # Add speaker + codec_pad + codec_bos
    num_tts_pad = len(codec_full) + 1  # +1 for speaker slot, then tts_bos pairs with codec_pad
    # Total codec_input: [think, think_bos, lang, think_eos, speaker, codec_pad, codec_bos]
    # [:-1] = [think, think_bos, lang, think_eos, speaker, codec_pad]
    # text side = [tts_pad × 5, tts_bos]  (5 = len(codec_prefix) + 1 for speaker)

    for cid in codec_full:
        e = tts_pad_embed + codec_emb[cid]
        embeds_list.append(e.reshape(1, -1))

    # Speaker slot: tts_pad + speaker_embed
    embeds_list.append((tts_pad_embed + spk_embed[0]).reshape(1, -1))

    # Transition: tts_bos + codec_pad
    embeds_list.append((tts_bos_embed + codec_pad_embed).reshape(1, -1))

    # Part C: ICL block (non-streaming)
    # text_embed = text_proj(ref_ids[3:-2] ++ input_ids[3:-5]) | tts_eos_embed
    ref_content_ids = ref_ids[3:-2]    # strip role prefix + trailing <|im_end|>\n
    text_content_ids = input_ids[3:-5]  # strip role prefix + trailing markers
    combined_text_ids = list(ref_content_ids) + list(text_content_ids)
    text_embed = text_proj(combined_text_ids)  # (T1-1, hidden)
    text_embed = np.concatenate([text_embed, tts_eos_embed.reshape(1, -1)], axis=0)  # (T1, hidden)
    T1 = text_embed.shape[0]

    # codec_embed = [codec_bos] | sum_over_groups(codec_embed_g[ref_code])
    # For each ref code frame, sum all 16 group embeddings
    ref_frames = ref_codes.shape[2]
    codec_frame_embeds = np.zeros((ref_frames, hidden_size), dtype=np.float32)
    for f in range(ref_frames):
        # Group 0: talker codec embedding
        codec_frame_embeds[f] += codec_emb[ref_codes[0, 0, f]]
        # Groups 1-15: CP codec embeddings
        for g in range(num_code_groups - 1):
            codec_frame_embeds[f] += cp_codec_embs[g][ref_codes[0, g + 1, f]]

    # Prepend codec_bos
    codec_embed = np.concatenate([
        codec_bos_embed.reshape(1, -1),
        codec_frame_embeds,
    ], axis=0)  # (T2, hidden)  where T2 = 1 + ref_frames
    T2 = codec_embed.shape[0]

    # Non-streaming ICL interleave:
    #   icl_input = (text_embed + codec_pad_embed × T1) | (codec_embed + tts_pad × T2)
    text_with_pad = text_embed + np.tile(codec_pad_embed, (T1, 1))
    codec_with_pad = codec_embed + np.tile(tts_pad_embed, (T2, 1))
    icl_embed = np.concatenate([text_with_pad, codec_with_pad], axis=0)  # (T1+T2, hidden)
    embeds_list.append(icl_embed)

    # Trailing text hidden for decode loop = tts_pad_embed (non-streaming)
    trailing_hidden = tts_pad_embed.reshape(1, -1)

    # Stack prefill
    prefill_embeds = np.concatenate(embeds_list, axis=0)[np.newaxis, :, :].astype(np.float32)
    T = prefill_embeds.shape[1]
    attention_mask = np.ones((1, T), dtype=np.int64)
    position_ids = np.arange(T).reshape(1, 1, T).repeat(3, axis=0)

    print(f"  Prefill: {T} tokens (role=3, codec_prefix={len(codec_full)+2}, "
          f"ICL text={T1}, ICL codec={T2})")

    # ------------------------------------------------------------------
    # 6. Run prefill + decode loop (same as generate_onnx.py)
    # ------------------------------------------------------------------
    num_layers = config["talker_num_layers"]
    vocab_size = config["talker_vocab_size"]
    codec_eos = config["codec_eos_token_id"]
    cp_num_layers = config["cp_num_layers"]
    cp_num_kv_heads = config["cp_num_kv_heads"]
    cp_head_dim = config["cp_head_dim"]

    suppress_mask = np.zeros(vocab_size, dtype=bool)
    suppress_mask[vocab_size - 1024:vocab_size] = True
    suppress_mask[codec_eos] = False

    print("  Running prefill ...")
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

    all_codes = []
    current_pos = T
    generated_tokens = []

    print("  Decoding ...")
    for step in range(max_new_tokens):
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
            cp_past_keys = cp_out[1]
            cp_past_values = cp_out[2]
            token = sample_top_k(cp_out[0][0, -1, :], top_k, temperature)
            frame_codes.append(token)
            cp_input = cp_codec_embs[g][token].reshape(1, 1, -1).astype(np.float32)

        all_codes.append(frame_codes)

        # Next talker input
        next_embed = codec_emb[group0_token].copy()
        for g in range(num_code_groups - 1):
            next_embed += cp_codec_embs[g][frame_codes[g + 1]]
        next_embed += trailing_hidden[0]
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

        if (step + 1) % 50 == 0:
            print(f"    ... {step + 1} frames")

    gen_time = time.time() - t0
    num_gen = len(all_codes)
    print(f"  Generated {num_gen} frames in {gen_time:.1f}s")

    if num_gen == 0:
        print("  ERROR: no frames generated")
        return

    # ------------------------------------------------------------------
    # 7. Vocoder: prepend ref_codes, decode, trim reference portion
    # ------------------------------------------------------------------
    gen_codes = np.array(all_codes, dtype=np.int64)  # (gen_frames, 16)
    ref_codes_t = ref_codes[0].T  # (ref_frames, 16) — was (16, ref_frames)

    all_codes_arr = np.concatenate([ref_codes_t, gen_codes], axis=0)  # (total, 16)
    codes_input = all_codes_arr.T[np.newaxis, :, :]  # (1, 16, total)
    total_frames = codes_input.shape[2]

    print(f"  Vocoder: {total_frames} frames (ref={ref_frames}, gen={num_gen})")
    t0 = time.time()
    wav = vocoder_sess.run(None, {"codes": codes_input})[0].flatten()
    voc_time = time.time() - t0

    # Trim leading reference portion (proportional cut like the Python reference)
    cut = int(ref_frames / max(total_frames, 1) * len(wav))
    wav = wav[cut:]

    duration = len(wav) / MEL_SR
    print(f"  Vocoder: {voc_time:.1f}s, output: {duration:.1f}s (trimmed {cut} samples)")

    sf.write(output_path, wav, MEL_SR)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Voice clone via ONNX-only Qwen3-TTS 0.6B Base pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python generate_clone_onnx.py --ref-audio ref.wav --ref-text "Hello world" --text "New text"
  python generate_clone_onnx.py --variant int4 --ref-audio ref.wav --ref-text "Hello" --text "New"
""",
    )
    parser.add_argument("--text", required=True, help="Text to synthesize in cloned voice")
    parser.add_argument("--ref-audio", required=True, help="Reference audio WAV path")
    parser.add_argument("--ref-text", required=True, help="Transcript of reference audio")
    parser.add_argument("--lang", default="english", help="Language (default: english)")
    parser.add_argument("--model-dir", default="./output/qwen3-tts-0.6b-base",
                        help="Root model directory")
    parser.add_argument("--variant", default="fp32", help="fp32 or int4 (default: fp32)")
    parser.add_argument("-o", "--output", default="clone_output.wav", help="Output WAV path")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    generate_clone_onnx(
        model_dir=args.model_dir,
        variant=args.variant,
        text=args.text,
        ref_audio_path=args.ref_audio,
        ref_text=args.ref_text,
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
