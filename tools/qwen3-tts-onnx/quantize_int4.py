#!/usr/bin/env python3
"""Quantize ONNX models to INT4 (weight-only, RTN).

Applies INT4 weight-only quantization to the exported ONNX models using
onnxruntime's MatMulNBits quantizer. This replaces float32 MatMul weight
matrices with 4-bit packed representations, reducing model size ~6-8x.

The vocoder is optionally skipped since audio quality can degrade with
aggressive quantization of the decoder network.

Examples:
  # Quantize all models (talker + code predictor + vocoder)
  python quantize_int4.py

  # Quantize only transformer models (skip vocoder)
  python quantize_int4.py --skip-vocoder

  # Custom model directory
  python quantize_int4.py --model-dir ./output/my-model

  # Custom block size
  python quantize_int4.py --block-size 64
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import onnx
from onnxruntime.quantization import quant_utils
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer,
    RTNWeightOnlyQuantConfig,
)


MODELS_TO_QUANTIZE = [
    "talker_prefill.onnx",
    "talker_decode.onnx",
    "code_predictor.onnx",
    "vocoder.onnx",
]


def quantize_model(input_path: str, output_path: str, block_size: int, is_symmetric: bool):
    """Quantize a single ONNX model to INT4."""
    basename = os.path.basename(input_path)
    print(f"\n{'='*60}")
    print(f"Quantizing: {basename}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Config: block_size={block_size}, symmetric={is_symmetric}")

    t0 = time.time()

    # Load model with shape inference
    print(f"  Loading model...")
    model = quant_utils.load_model_with_shape_infer(Path(input_path))

    # Configure RTN (Round-To-Nearest) quantization — no calibration data needed
    algo_config = RTNWeightOnlyQuantConfig()

    quantizer = MatMulNBitsQuantizer(
        model,
        bits=4,
        block_size=block_size,
        is_symmetric=is_symmetric,
        accuracy_level=4,  # int8 input, int32 accumulator — fastest
        algo_config=algo_config,
    )

    print(f"  Quantizing weights...")
    quantizer.process()

    # Save with external data consolidated into a single file
    print(f"  Saving...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_path = output_path + ".data"
    quantizer.model.save_model_to_file(
        output_path,
        use_external_data_format=True,
    )

    # Consolidate external data into a single file
    try:
        m = onnx.load(output_path)
        onnx.save_model(
            m,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(data_path),
        )
        # Clean up any scattered data files
        _cleanup_scattered_data(output_path)
    except Exception as e:
        print(f"  Note: consolidation skipped ({e})")

    elapsed = time.time() - t0

    # Report sizes
    input_size = _total_size(input_path)
    output_size = _total_size(output_path)
    ratio = input_size / output_size if output_size > 0 else 0

    print(f"  Done in {elapsed:.1f}s")
    print(f"  Size: {_fmt_size(input_size)} -> {_fmt_size(output_size)} ({ratio:.1f}x reduction)")


def _total_size(onnx_path: str) -> int:
    """Total size of .onnx + .onnx.data files."""
    total = os.path.getsize(onnx_path) if os.path.exists(onnx_path) else 0
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        total += os.path.getsize(data_path)
    return total


def _fmt_size(nbytes: int) -> str:
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3:.2f} GB"
    elif nbytes >= 1024**2:
        return f"{nbytes / 1024**2:.1f} MB"
    else:
        return f"{nbytes / 1024:.1f} KB"


def _cleanup_scattered_data(onnx_path: str):
    """Remove scattered external data files left by the quantizer."""
    onnx_dir = os.path.dirname(onnx_path)
    basename = os.path.basename(onnx_path)
    data_file = basename + ".data"
    # Common scattered patterns from quantizer
    for f in os.listdir(onnx_dir):
        if f.startswith(basename) and f != basename and f != data_file:
            path = os.path.join(onnx_dir, f)
            if os.path.isfile(path):
                os.remove(path)


def validate_int4(input_dir: str, output_dir: str, model_name: str):
    """Quick validation: load INT4 model and run with dummy input."""
    import numpy as np
    import onnxruntime as ort

    fp32_path = os.path.join(input_dir, model_name)
    int4_path = os.path.join(output_dir, model_name)

    if not os.path.exists(int4_path):
        return

    print(f"\n  Validating {model_name}...")

    fp32_sess = ort.InferenceSession(fp32_path)
    int4_sess = ort.InferenceSession(int4_path)

    # Create dummy inputs matching the model's expected shapes
    inputs = {}
    for inp in fp32_sess.get_inputs():
        shape = []
        for dim in inp.shape:
            if isinstance(dim, str):
                shape.append(4)  # small size for dynamic dims
            else:
                shape.append(dim)

        if inp.type == "tensor(int64)":
            if "codes" in inp.name:
                inputs[inp.name] = np.random.randint(0, 2048, shape).astype(np.int64)
            else:
                inputs[inp.name] = np.ones(shape, dtype=np.int64)
        else:
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

    try:
        fp32_out = fp32_sess.run(None, inputs)
        int4_out = int4_sess.run(None, inputs)

        for i, (fp32_o, int4_o) in enumerate(zip(fp32_out, int4_out)):
            if fp32_o.shape != int4_o.shape:
                print(f"    Output {i}: SHAPE MISMATCH {fp32_o.shape} vs {int4_o.shape}")
                continue
            max_err = np.max(np.abs(fp32_o.astype(np.float64) - int4_o.astype(np.float64)))
            cos_sim = _cosine_similarity(fp32_o.flatten(), int4_o.flatten())
            out_name = fp32_sess.get_outputs()[i].name
            print(f"    {out_name}: max_err={max_err:.4e}, cos_sim={cos_sim:.6f}")
    except Exception as e:
        print(f"    Validation error: {e}")


def _cosine_similarity(a, b):
    import numpy as np
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main():
    parser = argparse.ArgumentParser(
        description="Quantize ONNX models to INT4 (weight-only, RTN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python quantize_int4.py
  python quantize_int4.py --skip-vocoder
  python quantize_int4.py --block-size 64
  python quantize_int4.py --model-dir ./output/my-model
""",
    )
    parser.add_argument(
        "--model-dir",
        default="./output/qwen3-tts-1.7b-voicedesign",
        help="Root model directory containing fp32/ subfolder (default: ./output/qwen3-tts-1.7b-voicedesign)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Quantization block size, power of 2, >= 16 (default: 128)",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        default=True,
        help="Use symmetric quantization (default: True)",
    )
    parser.add_argument(
        "--skip-vocoder",
        action="store_true",
        help="Skip vocoder quantization (keeps FP32 vocoder for better audio quality)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip post-quantization validation",
    )
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    input_dir = os.path.join(model_dir, "fp32")
    output_dir = os.path.join(model_dir, "int4")

    print(f"INT4 Weight-Only Quantization (RTN)")
    print(f"  Model dir: {model_dir}")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Block size: {args.block_size}")
    print(f"  Symmetric: {args.symmetric}")
    print(f"  Skip vocoder: {args.skip_vocoder}")

    os.makedirs(output_dir, exist_ok=True)

    # Quantize each model
    models = list(MODELS_TO_QUANTIZE)
    if args.skip_vocoder:
        models.remove("vocoder.onnx")
        # Copy FP32 vocoder instead
        for f in ["vocoder.onnx", "vocoder.onnx.data"]:
            src = os.path.join(input_dir, f)
            dst = os.path.join(output_dir, f)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"  Copied FP32: {f}")

    for model_name in models:
        input_path = os.path.join(input_dir, model_name)
        if not os.path.exists(input_path):
            print(f"\n  Skipping {model_name} (not found)")
            continue

        output_path = os.path.join(output_dir, model_name)
        quantize_model(
            input_path=input_path,
            output_path=output_path,
            block_size=args.block_size,
            is_symmetric=args.symmetric,
        )

        if not args.skip_validation:
            validate_int4(input_dir, output_dir, model_name)

    # Summary
    print(f"\n{'='*60}")
    print(f"Quantization complete!")
    print(f"\nOutput directory: {output_dir}")
    total_input = 0
    total_output = 0
    for model_name in MODELS_TO_QUANTIZE:
        input_path = os.path.join(input_dir, model_name)
        output_path = os.path.join(output_dir, model_name)
        if os.path.exists(output_path):
            in_sz = _total_size(input_path)
            out_sz = _total_size(output_path)
            total_input += in_sz
            total_output += out_sz
            ratio = in_sz / out_sz if out_sz > 0 else 0
            print(f"  {model_name:30s}  {_fmt_size(in_sz):>10s} -> {_fmt_size(out_sz):>10s}  ({ratio:.1f}x)")

    if total_output > 0:
        ratio = total_input / total_output
        print(f"  {'TOTAL':30s}  {_fmt_size(total_input):>10s} -> {_fmt_size(total_output):>10s}  ({ratio:.1f}x)")

    print(f"\nUse with generate_onnx.py:")
    print(f"  python generate_onnx.py --model-dir {model_dir} --variant int4 --text 'Hello world'")


if __name__ == "__main__":
    main()
