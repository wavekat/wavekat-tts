#!/usr/bin/env python3
"""Export Qwen3-TTS vocoder (speech tokenizer decoder) as vocoder.onnx.

NOTE: The vocoder's internal transformer bakes sequence length during JIT trace
(cache_position = torch.arange(seq_len) becomes constant). The exported model
works correctly for the same T used during export. For variable-length sequences
at inference time, use the chunked_decode approach: split codes into fixed-size
chunks with overlap, decode each chunk, and concatenate.
"""

import argparse
import os

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    EuclideanCodebook,
)
from mask_patch import patch_causal_mask


def precompute_codebook_embeddings(decoder):
    """Replace EuclideanCodebook's runtime division with precomputed embeddings."""
    count = 0
    for name, module in decoder.named_modules():
        if isinstance(module, EuclideanCodebook):
            with torch.no_grad():
                embedding = module.embedding_sum / module.cluster_usage.clamp(
                    min=module.epsilon
                )[:, None]
                module.register_buffer("_precomputed_embedding", embedding)

                def make_fast_decode(mod):
                    def fast_decode(codes):
                        return F.embedding(codes, mod._precomputed_embedding)
                    return fast_decode

                module.decode = make_fast_decode(module)
                count += 1
    print(f"  Precomputed {count} EuclideanCodebook embeddings")


class VocoderWrapper(nn.Module):
    """Wrapper for ONNX export of the vocoder.

    Input: codes (1, num_quantizers, T) int64
    Output: waveform (1, 1, T*upsample_factor) float32
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, codes):
        wav = self.decoder(codes)
        return wav


def export_vocoder(model_id: str, output_dir: str, chunk_size: int = 300):
    os.makedirs(output_dir, exist_ok=True)
    patch_causal_mask()

    print(f"Loading model: {model_id}")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    speech_tokenizer = model.speech_tokenizer
    decoder = speech_tokenizer.model.decoder

    num_quantizers = decoder.config.num_quantizers
    upsample = decoder.total_upsample
    print(f"  Vocoder: num_quantizers={num_quantizers}, upsample={upsample}")

    precompute_codebook_embeddings(decoder)
    decoder.pre_transformer.config.use_cache = False

    wrapper = VocoderWrapper(decoder)
    wrapper.eval()

    # Export with a fixed chunk size (default 300 frames, matching chunked_decode)
    # The ONNX model will work correctly for exactly this T.
    # For variable lengths, use chunked decode at this fixed chunk size.
    T = chunk_size
    dummy_codes = torch.randint(0, 2048, (1, num_quantizers, T), dtype=torch.int64)

    onnx_path = os.path.join(output_dir, "vocoder.onnx")

    print(f"\nExporting vocoder.onnx (chunk_size={T}) ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_codes,),
            onnx_path,
            opset_version=17,
            dynamo=False,
            input_names=["codes"],
            output_names=["waveform"],
        )

    _consolidate(onnx_path)
    print(f"  Saved: {onnx_path}")

    # Validation with the same T
    _validate(wrapper, dummy_codes, onnx_path)

    # Also export a smaller chunk model for tail segments
    small_T = 50
    small_path = os.path.join(output_dir, "vocoder_small.onnx")
    dummy_small = torch.randint(0, 2048, (1, num_quantizers, small_T), dtype=torch.int64)

    print(f"\nExporting vocoder_small.onnx (chunk_size={small_T}) ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_small,),
            small_path,
            opset_version=17,
            dynamo=False,
            input_names=["codes"],
            output_names=["waveform"],
        )
    _consolidate(small_path)
    print(f"  Saved: {small_path}")
    _validate(wrapper, dummy_small, small_path)

    print(f"\nVocoder export complete.")
    print(f"  Use chunk_size={T} for main decode, {small_T} for tail segments.")
    print(f"  For variable-length: split codes into chunks with 25-frame overlap.")


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


def _validate(wrapper, codes, onnx_path):
    import onnxruntime as ort

    with torch.no_grad():
        pt_out = wrapper(codes)

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {"codes": codes.numpy()})

    pt_wav = pt_out.numpy()
    ort_wav = ort_out[0]
    max_err = np.max(np.abs(pt_wav - ort_wav))
    print(f"  Vocoder validation: waveform max_err={max_err:.6e}, "
          f"shape={ort_wav.shape}, range=[{ort_wav.min():.3f}, {ort_wav.max():.3f}]")
    if max_err > 1e-3:
        print(f"  WARNING: max error {max_err:.6e} exceeds 1e-3 threshold")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS vocoder to ONNX")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--output-dir", default="./output/qwen3-tts-1.7b-voicedesign")
    parser.add_argument("--chunk-size", type=int, default=300,
                        help="Fixed chunk size for vocoder (default: 300)")
    args = parser.parse_args()
    export_vocoder(args.model_id, args.output_dir, args.chunk_size)


if __name__ == "__main__":
    main()
