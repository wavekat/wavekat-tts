"""Patch transformers' create_causal_mask to avoid torch.vmap for ONNX export.

The default create_causal_mask in transformers >=4.57 uses torch.vmap to build
attention masks, which is incompatible with torch.onnx.export (JIT trace mode).
This module replaces it with a simple implementation using standard tensor ops.

Usage: call patch_causal_mask() before any torch.onnx.export that involves
transformer models using create_causal_mask.
"""

import torch


def patch_causal_mask():
    """Replace create_causal_mask with a vmap-free version."""
    import transformers.masking_utils as mu

    def simple_causal_mask(config, input_embeds, attention_mask, cache_position,
                           past_key_values, position_ids=None, **kwargs):
        dtype = input_embeds.dtype
        device = input_embeds.device
        batch_size, query_len = input_embeds.shape[:2]

        if past_key_values is not None:
            kv_len = past_key_values.get_seq_length() + query_len
        else:
            kv_len = query_len

        min_val = torch.finfo(dtype).min

        # Build causal mask using comparison: query positions attend to kv positions <= them
        row_idx = cache_position.view(-1, 1)  # (Q, 1) — absolute positions of queries
        col_idx = torch.arange(kv_len, device=device).view(1, -1)  # (1, K)
        attend = col_idx <= row_idx  # (Q, K)
        causal_mask = torch.where(
            attend,
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.tensor(min_val, dtype=dtype, device=device),
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        # Apply padding mask
        if attention_mask is not None and attention_mask.shape[1] > 0:
            pad_mask = (attention_mask[:, :kv_len] == 0)  # (B, K)
            causal_mask = causal_mask.clone()
            causal_mask.masked_fill_(pad_mask.unsqueeze(1).unsqueeze(1), min_val)

        return causal_mask

    def simple_sliding_window_causal_mask(config, input_embeds, attention_mask,
                                          cache_position, past_key_values,
                                          position_ids=None, **kwargs):
        dtype = input_embeds.dtype
        device = input_embeds.device
        batch_size, query_len = input_embeds.shape[:2]

        if past_key_values is not None:
            kv_len = past_key_values.get_seq_length() + query_len
        else:
            kv_len = query_len

        min_val = torch.finfo(dtype).min
        sliding_window = getattr(config, "sliding_window", None)

        row_idx = cache_position.view(-1, 1)
        col_idx = torch.arange(kv_len, device=device).view(1, -1)

        attend = col_idx <= row_idx
        if sliding_window is not None:
            attend = attend & (col_idx > (row_idx - sliding_window))

        causal_mask = torch.where(
            attend,
            torch.tensor(0.0, dtype=dtype, device=device),
            torch.tensor(min_val, dtype=dtype, device=device),
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        if attention_mask is not None and attention_mask.shape[1] > 0:
            pad_mask = (attention_mask[:, :kv_len] == 0)
            causal_mask = causal_mask.clone()
            causal_mask.masked_fill_(pad_mask.unsqueeze(1).unsqueeze(1), min_val)

        return causal_mask

    # Patch in transformers.masking_utils
    mu.create_causal_mask = simple_causal_mask
    mu.create_sliding_window_causal_mask = simple_sliding_window_causal_mask

    # Patch in the modeling module (imported at module level)
    import qwen_tts.core.models.modeling_qwen3_tts as mod
    if hasattr(mod, 'create_causal_mask'):
        mod.create_causal_mask = simple_causal_mask
    if hasattr(mod, 'create_sliding_window_causal_mask'):
        mod.create_sliding_window_causal_mask = simple_sliding_window_causal_mask

    # Also patch the tokenizer decoder module (vocoder uses transformers)
    try:
        import qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 as tok_mod
        if hasattr(tok_mod, 'create_causal_mask'):
            tok_mod.create_causal_mask = simple_causal_mask
        if hasattr(tok_mod, 'create_sliding_window_causal_mask'):
            tok_mod.create_sliding_window_causal_mask = simple_sliding_window_causal_mask
    except ImportError:
        pass

    print("  Patched create_causal_mask (vmap-free)")
