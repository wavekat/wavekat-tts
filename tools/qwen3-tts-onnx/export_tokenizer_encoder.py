#!/usr/bin/env python3
"""Export the speech-tokenizer encoder as tokenizer_encoder.onnx.

Voice clone needs the *reference* audio turned into the same discrete codes the
talker normally produces. The Base model uses `model.speech_tokenizer.encode()`
for this — a Mimi-based residual VQ encoder that runs at 12 Hz with 16
quantizer groups.

Pipeline (as in qwen_tts.inference.qwen3_tts_tokenizer.Qwen3TTSTokenizer.encode):

  audio (24 kHz mono float32, 1-D)
    └── unsqueeze ──> (1, T) waveform
                       └── inner encoder.encode(input_values=(1, 1, T))
                            └── codes: (1, num_quantizers, frames)
                                   └── slice [:, :encoder_valid_num_quantizers]

We expose the same signature in ONNX:

  Input :  waveform `(1, FIXED_SAMPLES)` float32, 24 kHz
  Output:  audio_codes `(1, num_valid, frames)` int64

NOTE: The Mimi encoder uses data-dependent conv padding (`.item()` in
`_get_extra_padding_for_conv1d`), which makes `torch.export` / dynamo fail
on dynamic shapes. We therefore use the **legacy JIT tracer** with a fixed
canonical sample length. Host code must zero-pad (or truncate) the reference
waveform to exactly `CANONICAL_SAMPLES` before feeding it to the ONNX session,
and trim trailing code frames based on the original audio length.

The encoder only runs *once per reference clip* (not in the autoregressive
loop), so the fixed-size constraint has no performance impact.
"""

import argparse
import os

import numpy as np
import onnx
import torch
import torch.nn as nn

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration
from mask_patch import patch_causal_mask

# Canonical sample count: 10 s × 24 kHz = 240 000 samples.
# Covers reference clips up to 10 seconds. Shorter clips are zero-padded;
# the host code trims the output codes based on the original sample count.
CANONICAL_SECONDS = 10
CANONICAL_SR = 24000
CANONICAL_SAMPLES = CANONICAL_SECONDS * CANONICAL_SR  # 240_000


class TokenizerEncoderWrapper(nn.Module):
    """Wraps `Qwen3TTSTokenizerV2Encoder` (a `MimiModel` subclass).

    Calls the inner encoder's `.encode()` and slices to the valid quantizer
    count. Input is a fixed-length waveform; output is the code matrix.
    """

    def __init__(self, encoder, num_valid_quantizers: int):
        super().__init__()
        self.encoder = encoder
        self.num_valid = num_valid_quantizers

    def forward(self, waveform):  # (1, CANONICAL_SAMPLES) float32
        # Mimi encoder expects (B, 1, T)
        encoded = self.encoder.encode(
            input_values=waveform.unsqueeze(1),
            return_dict=True,
        )
        # encoded.audio_codes: (B, num_quantizers, frames) int64
        return encoded.audio_codes[:, : self.num_valid]


def export_tokenizer_encoder(model_id: str, output_dir: str):
    # Mimi's encoder_transformer uses create_causal_mask which calls torch.vmap,
    # incompatible with JIT tracing. Patch it the same way as the talker export.
    patch_causal_mask()

    print(f"Loading model: {model_id}")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    speech_tokenizer = model.speech_tokenizer
    if speech_tokenizer is None:
        raise SystemExit(
            f"Model {model_id} does not expose a speech_tokenizer. "
            "Voice clone export requires a Base checkpoint with the 12Hz "
            "tokenizer (e.g. Qwen/Qwen3-TTS-12Hz-0.6B-Base)."
        )

    inner = speech_tokenizer.model
    encoder = inner.encoder
    num_valid = int(inner.encoder_valid_num_quantizers)
    input_sr = int(inner.input_sample_rate)
    encode_downsample = int(inner.encode_downsample_rate)
    print(
        f"  Tokenizer encoder: input_sr={input_sr}, "
        f"downsample={encode_downsample}, num_valid_quantizers={num_valid}"
    )
    assert input_sr == CANONICAL_SR, (
        f"Expected input_sr={CANONICAL_SR}, got {input_sr}. "
        f"Update CANONICAL_SR in this script."
    )

    wrapper = TokenizerEncoderWrapper(encoder, num_valid)
    wrapper.eval()

    dummy_wav = torch.randn(1, CANONICAL_SAMPLES, dtype=torch.float32) * 0.1
    expected_frames = CANONICAL_SAMPLES // encode_downsample
    print(
        f"  Fixed trace size: {CANONICAL_SAMPLES} samples "
        f"({CANONICAL_SECONDS}s @ {CANONICAL_SR} Hz) → {expected_frames} code frames"
    )

    onnx_path = os.path.join(output_dir, "tokenizer_encoder.onnx")
    os.makedirs(output_dir, exist_ok=True)

    pre_export = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()

    # Legacy JIT tracer — no dynamic shapes. The Mimi encoder's conv padding
    # uses .item() which creates data-dependent guards incompatible with dynamo.
    print(
        f"\nExporting tokenizer_encoder.onnx "
        f"(JIT trace, fixed T={CANONICAL_SAMPLES}) ..."
    )
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_wav,),
            onnx_path,
            opset_version=17,
            dynamo=False,
            input_names=["waveform"],
            output_names=["audio_codes"],
        )

    _try_consolidate(onnx_path, pre_export)
    print(f"  Saved: {onnx_path}")

    _validate(wrapper, dummy_wav, onnx_path)

    print(f"\nTokenizer encoder export complete.")
    print(f"  IMPORTANT: host code must zero-pad waveforms to exactly {CANONICAL_SAMPLES}")
    print(f"  samples ({CANONICAL_SECONDS}s @ {CANONICAL_SR} Hz) before inference,")
    print(f"  then trim output codes to ceil(original_samples / {encode_downsample}) frames.")


def _try_consolidate(onnx_path: str, pre_export_files: set | None = None):
    onnx_dir = os.path.dirname(onnx_path)
    data_path = onnx_path + ".data"

    try:
        m = onnx.load(onnx_path)
        onnx.save_model(
            m,
            onnx_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(data_path),
        )
    except Exception as e:
        print(f"  Note: consolidation skipped ({e})")
        return

    if pre_export_files is not None:
        current_files = set(os.listdir(onnx_dir))
        scattered = current_files - pre_export_files - {
            os.path.basename(onnx_path),
            os.path.basename(data_path),
        }
        for f in scattered:
            path = os.path.join(onnx_dir, f)
            if os.path.isfile(path):
                os.remove(path)
        if scattered:
            print(f"  Cleaned up {len(scattered)} scattered external data files")


def _validate(wrapper, waveform, onnx_path, label=None):
    import onnxruntime as ort

    with torch.no_grad():
        pt_codes = wrapper(waveform)

    sess = ort.InferenceSession(onnx_path)
    ort_codes = sess.run(None, {"waveform": waveform.numpy()})[0]

    pt_arr = pt_codes.numpy()
    # Codes are integer — they should match exactly.
    same = bool(np.array_equal(pt_arr, ort_codes))
    diffs = int(np.sum(pt_arr != ort_codes))
    tag = f" ({label})" if label else ""
    print(
        f"  Tokenizer encoder validation{tag}: identical={same}, "
        f"diffs={diffs}, shape={ort_codes.shape}"
    )
    if not same:
        print(f"  WARNING: {diffs} code mismatches between PyTorch and ONNX")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS speech tokenizer encoder to ONNX")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="HuggingFace model ID (must include a 12Hz speech tokenizer)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output/qwen3-tts-0.6b-base",
        help="Output directory (tokenizer_encoder.onnx is written at the root)",
    )
    args = parser.parse_args()
    export_tokenizer_encoder(args.model_id, args.output_dir)


if __name__ == "__main__":
    main()
