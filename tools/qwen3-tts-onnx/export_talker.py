#!/usr/bin/env python3
"""Export Qwen3-TTS talker as talker_prefill.onnx and talker_decode.onnx."""

import argparse
import os

import numpy as np
import onnx
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from mask_patch import patch_causal_mask


class TalkerPrefillWrapper(nn.Module):
    """Wrapper for ONNX export of the talker prefill (full sequence, no past KV)."""

    def __init__(self, talker_model, codec_head, num_layers):
        super().__init__()
        self.talker_model = talker_model
        self.codec_head = codec_head
        self.num_layers = num_layers

    def forward(self, inputs_embeds, attention_mask, position_ids):
        cache_position = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        )

        outputs = self.talker_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states)
        past_kv = outputs.past_key_values

        # Extract per-layer KV using new Cache API: c[i] -> (key, value)
        result = [logits, hidden_states]
        for i in range(self.num_layers):
            k, v = past_kv[i]
            result.append(k)
            result.append(v)
        return tuple(result)


class TalkerDecodeWrapper(nn.Module):
    """Wrapper for ONNX export of single-step talker decode with KV cache."""

    def __init__(self, talker_model, codec_head, num_layers):
        super().__init__()
        self.talker_model = talker_model
        self.codec_head = codec_head
        self.num_layers = num_layers

    def forward(self, inputs_embeds, attention_mask, position_ids, past_keys, past_values):
        # Reconstruct DynamicCache from stacked KV tensors
        # past_keys: (num_layers, B, kv_heads, past_seq, head_dim)
        cache = DynamicCache()
        for i in range(self.num_layers):
            cache.update(past_keys[i], past_values[i], i)

        # cache_position for single decode step
        past_seq = past_keys.shape[3]
        cache_position = torch.arange(
            past_seq, past_seq + inputs_embeds.shape[1], device=inputs_embeds.device
        )

        outputs = self.talker_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.codec_head(hidden_states)
        past_kv = outputs.past_key_values

        # Stack updated KV cache using new API
        keys_list = []
        values_list = []
        for i in range(self.num_layers):
            k, v = past_kv[i]
            keys_list.append(k)
            values_list.append(v)
        present_keys = torch.stack(keys_list)
        present_values = torch.stack(values_list)
        return logits, hidden_states, present_keys, present_values


def export_talker(model_id: str, output_dir: str):
    fp32_dir = os.path.join(output_dir, "fp32")
    os.makedirs(fp32_dir, exist_ok=True)

    # Patch causal mask creation to avoid torch.vmap (incompatible with ONNX export)
    patch_causal_mask()

    print(f"Loading model: {model_id}")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    talker = model.talker
    talker_model = talker.model
    codec_head = talker.codec_head
    cfg = model.config.talker_config

    num_layers = cfg.num_hidden_layers
    hidden_size = cfg.hidden_size
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim

    print(f"  Talker: hidden={hidden_size}, layers={num_layers}, kv_heads={num_kv_heads}, head_dim={head_dim}")

    # ========== Export Prefill ==========
    print("\nExporting talker_prefill.onnx ...")
    prefill_wrapper = TalkerPrefillWrapper(talker_model, codec_head, num_layers)
    prefill_wrapper.eval()

    T = 10
    dummy_embeds = torch.randn(1, T, hidden_size)
    dummy_mask = torch.ones(1, T, dtype=torch.int64)
    dummy_pos = torch.arange(T).unsqueeze(0).unsqueeze(0).expand(3, 1, T)

    prefill_output_names = ["logits", "hidden_states"]
    for i in range(num_layers):
        prefill_output_names.append(f"present_key_{i}")
        prefill_output_names.append(f"present_value_{i}")

    prefill_dynamic = {
        "inputs_embeds": {1: "seq_len"},
        "attention_mask": {1: "seq_len"},
        "position_ids": {2: "seq_len"},
        "logits": {1: "seq_len"},
        "hidden_states": {1: "seq_len"},
    }
    for i in range(num_layers):
        prefill_dynamic[f"present_key_{i}"] = {2: "seq_len"}
        prefill_dynamic[f"present_value_{i}"] = {2: "seq_len"}

    prefill_path = os.path.join(fp32_dir, "talker_prefill.onnx")
    with torch.no_grad():
        torch.onnx.export(
            prefill_wrapper,
            (dummy_embeds, dummy_mask, dummy_pos),
            prefill_path,
            opset_version=17,
            dynamo=False,
            input_names=["inputs_embeds", "attention_mask", "position_ids"],
            output_names=prefill_output_names,
            dynamic_axes=prefill_dynamic,
        )

    _consolidate(prefill_path)
    print(f"  Saved: {prefill_path}")
    _validate_prefill(prefill_wrapper, dummy_embeds, dummy_mask, dummy_pos, prefill_path)

    # ========== Export Decode ==========
    print("\nExporting talker_decode.onnx ...")
    decode_wrapper = TalkerDecodeWrapper(talker_model, codec_head, num_layers)
    decode_wrapper.eval()

    past_seq = T
    dummy_decode_embeds = torch.randn(1, 1, hidden_size)
    dummy_decode_mask = torch.ones(1, past_seq + 1, dtype=torch.int64)
    dummy_decode_pos = torch.tensor([[[past_seq]]]).expand(3, 1, 1)
    dummy_past_keys = torch.randn(num_layers, 1, num_kv_heads, past_seq, head_dim)
    dummy_past_values = torch.randn(num_layers, 1, num_kv_heads, past_seq, head_dim)

    decode_dynamic = {
        "inputs_embeds": {},
        "attention_mask": {1: "total_seq"},
        "position_ids": {},
        "past_keys": {3: "past_seq"},
        "past_values": {3: "past_seq"},
        "logits": {},
        "hidden_states": {},
        "present_keys": {3: "total_seq"},
        "present_values": {3: "total_seq"},
    }

    decode_path = os.path.join(fp32_dir, "talker_decode.onnx")
    with torch.no_grad():
        torch.onnx.export(
            decode_wrapper,
            (dummy_decode_embeds, dummy_decode_mask, dummy_decode_pos,
             dummy_past_keys, dummy_past_values),
            decode_path,
            opset_version=17,
            dynamo=False,
            input_names=["inputs_embeds", "attention_mask", "position_ids",
                         "past_keys", "past_values"],
            output_names=["logits", "hidden_states", "present_keys", "present_values"],
            dynamic_axes=decode_dynamic,
        )

    _consolidate(decode_path)
    print(f"  Saved: {decode_path}")
    _validate_decode(
        decode_wrapper, dummy_decode_embeds, dummy_decode_mask, dummy_decode_pos,
        dummy_past_keys, dummy_past_values, decode_path
    )

    print("\nTalker export complete.")


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


def _validate_prefill(wrapper, embeds, mask, pos, onnx_path):
    """Quick check that ONNX output shapes match PyTorch."""
    import onnxruntime as ort

    with torch.no_grad():
        pt_out = wrapper(embeds, mask, pos)

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {
        "inputs_embeds": embeds.numpy(),
        "attention_mask": mask.numpy(),
        "position_ids": pos.numpy(),
    })

    pt_logits = pt_out[0].numpy()
    ort_logits = ort_out[0]
    max_err = np.max(np.abs(pt_logits - ort_logits))
    print(f"  Prefill validation: logits max_err={max_err:.6e}, shape={ort_logits.shape}")
    if max_err > 1e-3:
        print(f"  WARNING: max error {max_err:.6e} exceeds 1e-3 threshold")


def _validate_decode(wrapper, embeds, mask, pos, past_keys, past_values, onnx_path):
    """Quick check that ONNX output shapes match PyTorch."""
    import onnxruntime as ort

    with torch.no_grad():
        pt_out = wrapper(embeds, mask, pos, past_keys, past_values)

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {
        "inputs_embeds": embeds.numpy(),
        "attention_mask": mask.numpy(),
        "position_ids": pos.numpy(),
        "past_keys": past_keys.numpy(),
        "past_values": past_values.numpy(),
    })

    pt_logits = pt_out[0].numpy()
    ort_logits = ort_out[0]
    max_err = np.max(np.abs(pt_logits - ort_logits))
    print(f"  Decode validation: logits max_err={max_err:.6e}, shape={ort_logits.shape}")
    if max_err > 1e-3:
        print(f"  WARNING: max error {max_err:.6e} exceeds 1e-3 threshold")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS talker to ONNX")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    )
    parser.add_argument(
        "--output-dir",
        default="./output/qwen3-tts-1.7b-voicedesign",
    )
    args = parser.parse_args()
    export_talker(args.model_id, args.output_dir)


if __name__ == "__main__":
    main()
