#!/usr/bin/env python3
"""Extract embedding weights and config from Qwen3-TTS model as .npy files."""

import argparse
import json
import os
import shutil

import numpy as np
import torch
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


def export_embeddings(model_id: str, output_dir: str):
    print(f"Loading model: {model_id}")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    talker = model.talker
    config = model.config
    talker_config = config.talker_config
    cp_config = talker_config.code_predictor_config

    embed_dir = os.path.join(output_dir, "embeddings")
    tok_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(embed_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    # --- Text embedding ---
    w = talker.model.text_embedding.weight.detach().cpu().numpy()
    np.save(os.path.join(embed_dir, "text_embedding.npy"), w)
    print(f"  text_embedding: {w.shape}")

    # --- Text projection MLP weights ---
    for name, param_name in [
        ("text_projection_fc1_weight", "linear_fc1.weight"),
        ("text_projection_fc1_bias", "linear_fc1.bias"),
        ("text_projection_fc2_weight", "linear_fc2.weight"),
        ("text_projection_fc2_bias", "linear_fc2.bias"),
    ]:
        p = dict(talker.text_projection.named_parameters())[param_name]
        w = p.detach().cpu().numpy()
        np.save(os.path.join(embed_dir, f"{name}.npy"), w)
        print(f"  {name}: {w.shape}")

    # --- Talker codec embedding ---
    w = talker.model.codec_embedding.weight.detach().cpu().numpy()
    np.save(os.path.join(embed_dir, "talker_codec_embedding.npy"), w)
    print(f"  talker_codec_embedding: {w.shape}")

    # --- Code predictor codec embeddings (one per group 1-15) ---
    cp_embeddings = talker.code_predictor.model.codec_embedding
    num_cp_groups = cp_config.num_code_groups - 1
    for i in range(num_cp_groups):
        w = cp_embeddings[i].weight.detach().cpu().numpy()
        np.save(os.path.join(embed_dir, f"cp_codec_embedding_{i}.npy"), w)
        print(f"  cp_codec_embedding_{i}: {w.shape}")

    # --- small_to_mtp_projection (if it exists and is not Identity) ---
    proj = talker.code_predictor.small_to_mtp_projection
    if isinstance(proj, torch.nn.Linear):
        w = proj.weight.detach().cpu().numpy()
        b = proj.bias.detach().cpu().numpy()
        np.save(os.path.join(embed_dir, "small_to_mtp_projection_weight.npy"), w)
        np.save(os.path.join(embed_dir, "small_to_mtp_projection_bias.npy"), b)
        print(f"  small_to_mtp_projection_weight: {w.shape}")
        print(f"  small_to_mtp_projection_bias: {b.shape}")
    else:
        print("  small_to_mtp_projection: Identity (talker hidden == CP hidden)")

    # --- Config JSON ---
    cfg = {
        "model_id": model_id,
        "tts_model_type": config.tts_model_type,
        "tts_model_size": config.tts_model_size,
        # Token IDs
        "im_start_token_id": config.im_start_token_id,
        "im_end_token_id": config.im_end_token_id,
        "tts_pad_token_id": config.tts_pad_token_id,
        "tts_bos_token_id": config.tts_bos_token_id,
        "tts_eos_token_id": config.tts_eos_token_id,
        # Talker codec token IDs
        "codec_eos_token_id": talker_config.codec_eos_token_id,
        "codec_think_id": talker_config.codec_think_id,
        "codec_nothink_id": talker_config.codec_nothink_id,
        "codec_think_bos_id": talker_config.codec_think_bos_id,
        "codec_think_eos_id": talker_config.codec_think_eos_id,
        "codec_pad_id": talker_config.codec_pad_id,
        "codec_bos_id": talker_config.codec_bos_id,
        # Talker dimensions
        "talker_hidden_size": talker_config.hidden_size,
        "talker_num_layers": talker_config.num_hidden_layers,
        "talker_num_attention_heads": talker_config.num_attention_heads,
        "talker_num_kv_heads": talker_config.num_key_value_heads,
        "talker_head_dim": talker_config.head_dim,
        "talker_vocab_size": talker_config.vocab_size,
        "talker_text_hidden_size": talker_config.text_hidden_size,
        "talker_num_code_groups": talker_config.num_code_groups,
        # Code predictor dimensions
        "cp_hidden_size": cp_config.hidden_size,
        "cp_num_layers": cp_config.num_hidden_layers,
        "cp_num_attention_heads": cp_config.num_attention_heads,
        "cp_num_kv_heads": cp_config.num_key_value_heads,
        "cp_head_dim": cp_config.head_dim,
        "cp_vocab_size": cp_config.vocab_size,
        "cp_num_code_groups": cp_config.num_code_groups,
        # Speaker / language maps
        "spk_id": talker_config.spk_id,
        "spk_is_dialect": talker_config.spk_is_dialect,
        "codec_language_id": talker_config.codec_language_id,
        # Generation config
        "sample_rate": 24000,
    }
    if model.generate_config is not None:
        cfg["generate_config"] = model.generate_config

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  config.json written")

    # --- Tokenizer files ---
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(tok_dir)
    print(f"  tokenizer saved to {tok_dir}")

    print(f"\nAll embeddings exported to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS embeddings as .npy")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output-dir",
        default="./output/qwen3-tts-1.7b-voicedesign",
        help="Output directory",
    )
    args = parser.parse_args()
    export_embeddings(args.model_id, args.output_dir)


if __name__ == "__main__":
    main()
