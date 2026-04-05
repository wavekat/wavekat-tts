#!/usr/bin/env python3
"""Generate WAV files from text using Qwen3-TTS.

Supports both PyTorch-only and ONNX vocoder modes.

Examples:
  # Basic generation (PyTorch vocoder)
  python generate.py --text "Give every small business the voice of a big one." \
    --instruct "Speak in a warm and friendly female voice"

  # With voice instruction
  python generate.py --text "AI phone answering for small businesses." \
    --instruct "Speak in a professional male voice"

  # Chinese text
  python generate.py --text "让每一家小企业，都拥有大企业的声音。" --lang chinese

  # Use ONNX vocoder (requires exported vocoder.onnx)
  python generate.py --text "Give every small business the voice of a big one." \
    --onnx-vocoder ./output/qwen3-tts-1.7b-voicedesign/fp32

  # Custom output path
  python generate.py --text "Hello" -o demo.wav
"""

import argparse
import time

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer

from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


def generate(
    model_id: str,
    text: str,
    instruct: str | None,
    language: str,
    output_path: str,
    onnx_vocoder_dir: str | None,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
):
    print(f"Loading model: {model_id}")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Build input_ids with chat template
    chat_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(chat_text, add_special_tokens=False)
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.int64)

    # Build instruct_ids if provided
    instruct_ids = None
    if instruct:
        instruct_text = f"<|im_start|>user\n{instruct}<|im_end|>\n"
        instruct_tokens = tokenizer.encode(instruct_text, add_special_tokens=False)
        instruct_ids = [torch.tensor([instruct_tokens], dtype=torch.int64)]

    # Determine speaker (None for VoiceDesign models)
    talker_cfg = model.config.talker_config
    if talker_cfg.spk_id:
        speaker = list(talker_cfg.spk_id.keys())[0]
    else:
        speaker = None

    print(f"  Text: '{text}'")
    if instruct:
        print(f"  Instruct: '{instruct}'")
    print(f"  Language: {language}, Speaker: {speaker}")
    print(f"  Sampling: temp={temperature}, rep_penalty={repetition_penalty}")

    t0 = time.time()
    with torch.no_grad():
        codes, _ = model.generate(
            input_ids=[input_ids_tensor],
            instruct_ids=instruct_ids,
            languages=[language],
            speakers=[speaker],
            non_streaming_mode=True,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=temperature,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
    gen_time = time.time() - t0

    codes_arr = codes[0].cpu().numpy()  # (num_frames, 16)
    num_frames = codes_arr.shape[0]
    print(f"  Generated {num_frames} frames in {gen_time:.1f}s")

    # Decode to audio
    codes_input = codes_arr.T[np.newaxis, :, :].astype(np.int64)  # (1, 16, T)

    if onnx_vocoder_dir:
        import os
        import onnxruntime as ort
        vocoder_path = os.path.join(onnx_vocoder_dir, "vocoder.onnx")
        print(f"  Decoding with ONNX vocoder: {vocoder_path}")
        t0 = time.time()
        sess = ort.InferenceSession(vocoder_path)
        wav = sess.run(None, {"codes": codes_input})[0].flatten()
        dec_time = time.time() - t0
    else:
        print(f"  Decoding with PyTorch vocoder")
        t0 = time.time()
        decoder = model.speech_tokenizer.model.decoder
        with torch.no_grad():
            wav = decoder(torch.tensor(codes_input)).cpu().numpy().flatten()
        dec_time = time.time() - t0

    duration = len(wav) / 24000
    print(f"  Vocoder: {dec_time:.1f}s, audio: {duration:.1f}s")

    sf.write(output_path, wav, 24000)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate WAV from text using Qwen3-TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python generate.py --text "Give every small business the voice of a big one." --instruct "Speak in a warm and friendly female voice"
  python generate.py --text "AI phone answering for small businesses." --instruct "Speak in a professional male voice"
  python generate.py --text "让每一家小企业，都拥有大企业的声音。" --lang chinese
  python generate.py --text "Hello" --onnx-vocoder ./output/qwen3-tts-1.7b-voicedesign/fp32
""",
    )
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--instruct", default=None,
                        help="Voice design instruction (e.g. 'Speak in a deep male voice')")
    parser.add_argument("--lang", default="english",
                        help="Language (default: english)")
    parser.add_argument("--model-id", default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                        help="HuggingFace model ID")
    parser.add_argument("-o", "--output", default="output.wav",
                        help="Output WAV path (default: output.wav)")
    parser.add_argument("--onnx-vocoder", default=None,
                        help="Path to ONNX export dir (uses vocoder.onnx for decoding)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max codec frames to generate (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature (default: 0.9)")
    parser.add_argument("--repetition-penalty", type=float, default=1.05,
                        help="Repetition penalty (default: 1.05)")
    args = parser.parse_args()

    generate(
        model_id=args.model_id,
        text=args.text,
        instruct=args.instruct,
        language=args.lang,
        output_path=args.output,
        onnx_vocoder_dir=args.onnx_vocoder,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == "__main__":
    main()
