#!/usr/bin/env python3
"""Export Qwen3-TTS vocoder (speech tokenizer decoder) as vocoder.onnx.

Uses the dynamo-based ONNX exporter (torch.onnx.export with dynamo=True) which
performs symbolic tracing, correctly handling dynamic shapes throughout the model
(conv layers, attention reshapes, causal masks, rotary embeddings, etc.).

The legacy JIT-trace exporter bakes intermediate tensor shapes as constants,
breaking dynamic sequence length support. The dynamo exporter avoids this.
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


def export_vocoder(model_id: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

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

    T = 100
    dummy_codes = torch.randint(0, 2048, (1, num_quantizers, T), dtype=torch.int64)

    onnx_path = os.path.join(output_dir, "vocoder.onnx")

    print(f"\nExporting vocoder.onnx (dynamo, dynamic T, traced with T={T}) ...")

    # Use dynamo-based exporter for proper dynamic shape support
    num_frames = torch.export.Dim("num_frames", min=2, max=4096)
    dynamic_shapes = {"codes": {2: num_frames}}

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_codes,),
            onnx_path,
            dynamo=True,
            input_names=["codes"],
            output_names=["waveform"],
            dynamic_shapes=dynamic_shapes,
        )

    print(f"  Saved: {onnx_path}")

    # Consolidate if external data files were created
    _try_consolidate(onnx_path)

    # Validate at the trace size
    _validate(wrapper, dummy_codes, onnx_path)

    # Validate at different T values to confirm dynamic axes work
    for test_T in [50, 200, 299]:
        test_codes = torch.randint(0, 2048, (1, num_quantizers, test_T), dtype=torch.int64)
        _validate(wrapper, test_codes, onnx_path, label=f"T={test_T}")

    print(f"\nVocoder export complete (dynamic sequence length).")


def _try_consolidate(onnx_path: str):
    """Consolidate external data into a single .onnx.data file if needed."""
    data_path = onnx_path + ".data"
    # Check if there are external data files to consolidate
    onnx_dir = os.path.dirname(onnx_path)
    basename = os.path.basename(onnx_path)

    # The dynamo exporter may create .onnx_data or other external files
    try:
        model = onnx.load(onnx_path)
        onnx.save_model(
            model,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(data_path),
        )
    except Exception as e:
        print(f"  Note: consolidation skipped ({e})")


def _validate(wrapper, codes, onnx_path, label=None):
    import onnxruntime as ort

    with torch.no_grad():
        pt_out = wrapper(codes)

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {"codes": codes.numpy()})

    pt_wav = pt_out.numpy()
    ort_wav = ort_out[0]
    max_err = np.max(np.abs(pt_wav - ort_wav))
    tag = f" ({label})" if label else ""
    print(f"  Vocoder validation{tag}: max_err={max_err:.6e}, "
          f"shape={ort_wav.shape}, range=[{ort_wav.min():.3f}, {ort_wav.max():.3f}]")
    if max_err > 1e-3:
        print(f"  WARNING: max error {max_err:.6e} exceeds 1e-3 threshold")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS vocoder to ONNX")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--output-dir", default="./output/qwen3-tts-1.7b-voicedesign")
    args = parser.parse_args()
    export_vocoder(args.model_id, args.output_dir)


if __name__ == "__main__":
    main()
