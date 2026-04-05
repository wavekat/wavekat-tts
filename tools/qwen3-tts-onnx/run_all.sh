#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_ID="${1:-Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"
OUTPUT_DIR="${2:-./output/qwen3-tts-1.7b-voicedesign}"

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Model:  $MODEL_ID"
echo "Output: $OUTPUT_DIR"
echo ""

echo "=== Step 1/6: Export embeddings ==="
python export_embeddings.py --model-id "$MODEL_ID" --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Step 2/6: Export talker (prefill + decode) ==="
python export_talker.py --model-id "$MODEL_ID" --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Step 3/6: Export code predictor ==="
python export_code_predictor.py --model-id "$MODEL_ID" --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Step 4/6: Export vocoder ==="
python export_vocoder.py --model-id "$MODEL_ID" --output-dir "$OUTPUT_DIR"

echo ""
echo "=== Step 5/6: Validate ==="
python validate.py --model-id "$MODEL_ID" --onnx-dir "$OUTPUT_DIR"

echo ""
echo "=== Step 6/6: Quantize INT4 ==="
python quantize_int4.py --model-dir "$OUTPUT_DIR"

echo ""
echo "Done! Output directory: $OUTPUT_DIR"
echo "  FP32 models: $OUTPUT_DIR/fp32/"
echo "  INT4 models: $OUTPUT_DIR/int4/"
