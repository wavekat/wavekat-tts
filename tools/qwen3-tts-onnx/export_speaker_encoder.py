#!/usr/bin/env python3
"""Export Qwen3-TTS-Base speaker encoder (ECAPA-TDNN) as speaker_encoder.onnx.

The speaker encoder is the conditioning module on the Base ("voice clone")
checkpoints. It maps an 80-bin mel-spectrogram of the reference clip to a
1024-dim speaker embedding, which is inserted into the talker prefill at the
speaker-slot position (after `codec_think_eos`, before the transition).

Mel input: (B, T_mel, mel_dim=128), float32
  Computed in host code (not part of ONNX) with the standard HiFi-GAN
  parameters used by the reference pipeline:
    n_fft=1024, hop_size=256, win_size=1024, num_mels=128, fmin=0, fmax=12000,
    sample_rate=24000, center=False, log on top of dynamic-range compression.

Output: (B, enc_dim=1024), float32

The encoder lives on `model.speaker_encoder` and only exists when
`config.tts_model_type == "base"`. VoiceDesign / CustomVoice checkpoints don't
have it; running this script against them is a no-op + clear error.
"""

import argparse
import os

import numpy as np
import onnx
import torch
import torch.nn as nn

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


class SpeakerEncoderWrapper(nn.Module):
    """Pass-through wrapper around the ECAPA-TDNN speaker encoder.

    The reference pipeline calls `speaker_encoder(mels)[0]` to drop the batch
    dim. We keep the batch dim in ONNX so callers can decide what to do.
    """

    def __init__(self, speaker_encoder):
        super().__init__()
        self.speaker_encoder = speaker_encoder

    def forward(self, mels):  # (B, T_mel, mel_dim)
        return self.speaker_encoder(mels)  # (B, enc_dim)


def export_speaker_encoder(model_id: str, output_dir: str):
    print(f"Loading model: {model_id}")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    if model.speaker_encoder is None:
        raise SystemExit(
            f"Model {model_id} (tts_model_type={model.config.tts_model_type}) "
            "has no speaker encoder. Voice clone export only applies to Base "
            "checkpoints (e.g. Qwen/Qwen3-TTS-12Hz-0.6B-Base)."
        )

    spk_cfg = model.config.speaker_encoder_config
    print(
        f"  Speaker encoder: mel_dim={spk_cfg.mel_dim}, "
        f"enc_dim={spk_cfg.enc_dim}, sample_rate={spk_cfg.sample_rate}"
    )

    wrapper = SpeakerEncoderWrapper(model.speaker_encoder)
    wrapper.eval()

    # Trace with a representative mel length: ~3 s at 24 kHz / hop 256
    # ≈ 281 frames. Use 300 for headroom.
    T_mel = 300
    dummy_mels = torch.randn(1, T_mel, spk_cfg.mel_dim, dtype=torch.float32)

    onnx_path = os.path.join(output_dir, "speaker_encoder.onnx")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nExporting speaker_encoder.onnx (trace, dynamic mel length, T={T_mel}) ...")

    pre_export = set(os.listdir(output_dir)) if os.path.exists(output_dir) else set()

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_mels,),
            onnx_path,
            opset_version=17,
            dynamo=False,
            input_names=["mels"],
            output_names=["speaker_embedding"],
            dynamic_axes={
                "mels": {0: "batch", 1: "mel_frames"},
                "speaker_embedding": {0: "batch"},
            },
        )

    _try_consolidate(onnx_path, pre_export)
    print(f"  Saved: {onnx_path}")

    _validate(wrapper, dummy_mels, onnx_path)
    # Spot-check dynamic axis at a couple of other lengths.
    for test_T in [100, 500]:
        test_mels = torch.randn(1, test_T, spk_cfg.mel_dim, dtype=torch.float32)
        _validate(wrapper, test_mels, onnx_path, label=f"T_mel={test_T}")

    print("\nSpeaker encoder export complete.")


def _try_consolidate(onnx_path: str, pre_export_files: set | None = None):
    """Consolidate external data into a single .onnx.data file if needed."""
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


def _validate(wrapper, mels, onnx_path, label=None):
    import onnxruntime as ort

    with torch.no_grad():
        pt_out = wrapper(mels)

    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {"mels": mels.numpy()})[0]

    pt_arr = pt_out.numpy()
    max_err = float(np.max(np.abs(pt_arr - ort_out)))
    tag = f" ({label})" if label else ""
    print(
        f"  Speaker encoder validation{tag}: max_err={max_err:.6e}, "
        f"shape={ort_out.shape}"
    )
    if max_err > 1e-4:
        print(f"  WARNING: max error {max_err:.6e} exceeds 1e-4 threshold")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS speaker encoder to ONNX")
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="HuggingFace model ID (must be a Base checkpoint)",
    )
    parser.add_argument(
        "--output-dir",
        default="./output/qwen3-tts-0.6b-base",
        help="Output directory (speaker_encoder.onnx is written at the root, alongside fp32/, int4/, etc.)",
    )
    args = parser.parse_args()
    export_speaker_encoder(args.model_id, args.output_dir)


if __name__ == "__main__":
    main()
