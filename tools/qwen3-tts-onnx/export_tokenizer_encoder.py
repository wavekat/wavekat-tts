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

  Input :  waveform `(B, T)` float32, 24 kHz
  Output:  audio_codes `(B, num_valid, frames)` int64

Padding-mask trimming is *not* baked in — host code passes the actual sample
count and trims the trailing tail itself if needed. For voice cloning we
encode whole short clips (no batching), so padding doesn't matter.

The dynamo exporter handles the Mimi encoder's dynamic conv shapes correctly;
the legacy JIT trace bakes them as constants and breaks dynamic T.
"""

import argparse
import os

import numpy as np
import onnx
import torch
import torch.nn as nn

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


class TokenizerEncoderWrapper(nn.Module):
    """Wraps `Qwen3TTSTokenizerV2Encoder` (a `MimiModel` subclass).

    Calls the inner encoder's `.encode()` and slices to the valid quantizer
    count. We keep batch dim and frame dim dynamic; the rest is static.
    """

    def __init__(self, encoder, num_valid_quantizers: int):
        super().__init__()
        self.encoder = encoder
        self.num_valid = num_valid_quantizers

    def forward(self, waveform):  # (B, T) float32
        # Mimi encoder expects (B, 1, T)
        encoded = self.encoder.encode(
            input_values=waveform.unsqueeze(1),
            return_dict=True,
        )
        # encoded.audio_codes: (B, num_quantizers, frames) int64
        return encoded.audio_codes[:, : self.num_valid]


def export_tokenizer_encoder(model_id: str, output_dir: str):
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

    wrapper = TokenizerEncoderWrapper(encoder, num_valid)
    wrapper.eval()

    # Trace with ~3 s of audio at the encoder's input sample rate.
    # 3 s × 24 kHz = 72_000 samples; round to a multiple of the downsample rate.
    target_samples = 3 * input_sr
    target_samples -= target_samples % encode_downsample
    dummy_wav = torch.randn(1, target_samples, dtype=torch.float32) * 0.1
    print(f"  Tracing with waveform shape {tuple(dummy_wav.shape)} (~3 s @ {input_sr} Hz)")

    onnx_path = os.path.join(output_dir, "tokenizer_encoder.onnx")
    os.makedirs(output_dir, exist_ok=True)

    pre_export = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()

    # Use dynamo for clean dynamic-shape support through the conv stack.
    batch_dim = torch.export.Dim("batch", min=1, max=8)
    samples_dim = torch.export.Dim("samples", min=encode_downsample, max=20 * input_sr)
    # Dynamo requires the dynamic dim to be divisible by the encoder downsample.
    # We can't express divisibility at dim level, so we just trust callers.
    dynamic_shapes = {"waveform": {0: batch_dim, 1: samples_dim}}

    print("\nExporting tokenizer_encoder.onnx (dynamo, dynamic batch + samples) ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_wav,),
            onnx_path,
            dynamo=True,
            input_names=["waveform"],
            output_names=["audio_codes"],
            dynamic_shapes=dynamic_shapes,
        )

    _try_consolidate(onnx_path, pre_export)
    print(f"  Saved: {onnx_path}")

    _validate(wrapper, dummy_wav, onnx_path)
    # Validate at a couple of other lengths to confirm dynamic axes hold.
    for sec in (1, 5):
        n = sec * input_sr
        n -= n % encode_downsample
        test_wav = torch.randn(1, n, dtype=torch.float32) * 0.1
        _validate(wrapper, test_wav, onnx_path, label=f"{sec}s")

    print("\nTokenizer encoder export complete.")


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
