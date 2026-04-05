#!/usr/bin/env python3
"""Export Qwen3-TTS code predictor as code_predictor.onnx."""

import argparse
import os

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from mask_patch import patch_causal_mask


class CodePredictorWrapper(nn.Module):
    """Wrapper for ONNX export of the code predictor with lm_head selection.

    Includes small_to_mtp_projection inside the model so that inputs are in
    talker hidden space (2048 for 1.7B) and projection is handled internally.
    """

    def __init__(self, code_predictor, num_layers, num_groups):
        super().__init__()
        self.cp_model = code_predictor.model  # the transformer backbone
        self.projection = code_predictor.small_to_mtp_projection
        self.num_layers = num_layers
        self.num_groups = num_groups  # number of groups predicted (num_code_groups - 1)

        # Stack all lm_head weights into a single buffer for indexing
        # Each lm_head: Linear(cp_hidden, cp_vocab, bias=False)
        self.register_buffer(
            "lm_head_weights",
            torch.stack([h.weight for h in code_predictor.lm_head]),  # (num_groups, vocab, hidden)
        )

    def forward(self, inputs_embeds, generation_steps, past_keys, past_values):
        # Apply projection (2048 -> 1024 for 1.7B, Identity for 0.6B)
        inputs_embeds = self.projection(inputs_embeds)

        # Reconstruct DynamicCache from stacked KV tensors
        cache = DynamicCache()
        for i in range(self.num_layers):
            cache.update(past_keys[i], past_values[i], i)

        # Compute cache_position
        past_seq = past_keys.shape[3]
        cache_position = torch.arange(
            past_seq, past_seq + inputs_embeds.shape[1], device=inputs_embeds.device
        )

        outputs = self.cp_model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=cache,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state

        # Select the right lm_head by generation_steps
        # lm_head_weights: (num_groups, vocab, hidden)
        weight = self.lm_head_weights[generation_steps[0]]  # (vocab, hidden)
        logits = F.linear(hidden_states, weight)  # (B, S, vocab)

        # Stack updated KV cache using new API
        past_kv = outputs.past_key_values
        keys_list = []
        values_list = []
        for i in range(self.num_layers):
            k, v = past_kv[i]
            keys_list.append(k)
            values_list.append(v)
        present_keys = torch.stack(keys_list)
        present_values = torch.stack(values_list)

        return logits, present_keys, present_values


def export_code_predictor(model_id: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    patch_causal_mask()

    print(f"Loading model: {model_id}")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    cp = model.talker.code_predictor
    cp_cfg = model.config.talker_config.code_predictor_config
    talker_cfg = model.config.talker_config

    num_layers = cp_cfg.num_hidden_layers
    cp_hidden = cp_cfg.hidden_size
    talker_hidden = talker_cfg.hidden_size
    num_kv_heads = cp_cfg.num_key_value_heads
    head_dim = cp_cfg.head_dim
    vocab_size = cp_cfg.vocab_size
    num_groups = cp_cfg.num_code_groups - 1  # groups 1-15

    print(f"  Code Predictor: cp_hidden={cp_hidden}, talker_hidden={talker_hidden}, "
          f"layers={num_layers}, kv_heads={num_kv_heads}, groups={num_groups}")

    wrapper = CodePredictorWrapper(cp, num_layers, num_groups)
    wrapper.eval()

    # Dummy inputs
    # First call: inputs_embeds has 2 tokens (hidden_state + group0_embed), both in talker hidden space
    S = 2
    past_seq = 0
    dummy_embeds = torch.randn(1, S, talker_hidden)
    dummy_gen_steps = torch.tensor([0], dtype=torch.int64)
    dummy_past_keys = torch.zeros(num_layers, 1, num_kv_heads, past_seq, head_dim)
    dummy_past_values = torch.zeros(num_layers, 1, num_kv_heads, past_seq, head_dim)

    dynamic_axes = {
        "inputs_embeds": {1: "seq_len"},
        "past_keys": {3: "past_seq"},
        "past_values": {3: "past_seq"},
        "logits": {1: "seq_len"},
        "present_keys": {3: "total_seq"},
        "present_values": {3: "total_seq"},
    }

    onnx_path = os.path.join(output_dir, "code_predictor.onnx")

    print("\nExporting code_predictor.onnx ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_embeds, dummy_gen_steps, dummy_past_keys, dummy_past_values),
            onnx_path,
            opset_version=17,
            dynamo=False,
            input_names=["inputs_embeds", "generation_steps", "past_keys", "past_values"],
            output_names=["logits", "present_keys", "present_values"],
            dynamic_axes=dynamic_axes,
        )

    # Consolidate external data
    _consolidate(onnx_path)
    print(f"  Saved: {onnx_path}")

    # Quick validation
    _validate(wrapper, dummy_embeds, dummy_gen_steps, dummy_past_keys, dummy_past_values, onnx_path)

    # Also test with subsequent call (single token, non-empty cache)
    print("  Validating with decode step (S=1, past_seq=2) ...")
    dummy_embeds_1 = torch.randn(1, 1, talker_hidden)
    dummy_gen_steps_1 = torch.tensor([1], dtype=torch.int64)
    dummy_past_keys_1 = torch.randn(num_layers, 1, num_kv_heads, 2, head_dim)
    dummy_past_values_1 = torch.randn(num_layers, 1, num_kv_heads, 2, head_dim)
    _validate(wrapper, dummy_embeds_1, dummy_gen_steps_1, dummy_past_keys_1, dummy_past_values_1, onnx_path)

    print("\nCode predictor export complete.")


def _consolidate(onnx_path: str):
    """Consolidate external data into a single .onnx.data file."""
    data_path = onnx_path + ".data"
    model = onnx.load(onnx_path)
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(data_path),
    )


def _validate(wrapper, embeds, gen_steps, past_keys, past_values, onnx_path):
    import onnxruntime as ort

    with torch.no_grad():
        pt_out = wrapper(embeds, gen_steps, past_keys, past_values)

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {
        "inputs_embeds": embeds.numpy(),
        "generation_steps": gen_steps.numpy(),
        "past_keys": past_keys.numpy(),
        "past_values": past_values.numpy(),
    })

    pt_logits = pt_out[0].numpy()
    ort_logits = ort_out[0]
    max_err = np.max(np.abs(pt_logits - ort_logits))
    print(f"  CP validation: logits max_err={max_err:.6e}, shape={ort_logits.shape}")
    if max_err > 1e-3:
        print(f"  WARNING: max error {max_err:.6e} exceeds 1e-3 threshold")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS code predictor to ONNX")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    )
    parser.add_argument(
        "--output-dir",
        default="./output/qwen3-tts-1.7b-voicedesign",
    )
    args = parser.parse_args()
    export_code_predictor(args.model_id, args.output_dir)


if __name__ == "__main__":
    main()
