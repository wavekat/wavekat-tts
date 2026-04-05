MODEL_ID ?= Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
OUTPUT_DIR ?= ./output/qwen3-tts-1.7b-voicedesign
PYTHON ?= .venv/bin/python

FP32_DIR = $(OUTPUT_DIR)/fp32
INT4_DIR = $(OUTPUT_DIR)/int4

.PHONY: all venv export embeddings talker code-predictor vocoder validate quantize generate generate-int4 generate-zh generate-zh-int4 clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

venv: ## Create virtualenv and install dependencies
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

all: venv export validate quantize ## Export FP32, validate, then quantize INT4

export: embeddings talker code-predictor vocoder ## Export all FP32 ONNX models

embeddings: ## Export embeddings, config, and tokenizer
	$(PYTHON) export_embeddings.py --model-id $(MODEL_ID) --output-dir $(OUTPUT_DIR)

talker: ## Export talker prefill + decode ONNX models
	$(PYTHON) export_talker.py --model-id $(MODEL_ID) --output-dir $(OUTPUT_DIR)

code-predictor: ## Export code predictor ONNX model
	$(PYTHON) export_code_predictor.py --model-id $(MODEL_ID) --output-dir $(OUTPUT_DIR)

vocoder: ## Export vocoder ONNX model
	$(PYTHON) export_vocoder.py --model-id $(MODEL_ID) --output-dir $(OUTPUT_DIR)

validate: ## Validate ONNX exports against PyTorch
	$(PYTHON) validate.py --model-id $(MODEL_ID) --onnx-dir $(OUTPUT_DIR)

validate-int4: ## Validate INT4 exports against PyTorch
	$(PYTHON) validate.py --model-id $(MODEL_ID) --onnx-dir $(OUTPUT_DIR) --variant int4

quantize: ## Quantize FP32 models to INT4
	$(PYTHON) quantize_int4.py --model-dir $(OUTPUT_DIR)

generate: ## Generate sample audio (FP32)
	$(PYTHON) generate_onnx.py --model-dir $(OUTPUT_DIR) \
		--text "Give every small business the voice of a big one." \
		--instruct "Speak in a warm and friendly female voice" \
		-o sample_fp32.wav

generate-int4: ## Generate sample audio (INT4)
	$(PYTHON) generate_onnx.py --model-dir $(OUTPUT_DIR) --variant int4 \
		--text "Give every small business the voice of a big one." \
		--instruct "Speak in a warm and friendly female voice" \
		-o sample_int4.wav

generate-zh: ## Generate Chinese sample audio (FP32)
	$(PYTHON) generate_onnx.py --model-dir $(OUTPUT_DIR) \
		--text "让每一家小企业，都拥有大企业的声音。" \
		--lang chinese \
		--instruct "Speak slowly in a calm and professional female voice" \
		-o sample_zh_fp32.wav

generate-zh-int4: ## Generate Chinese sample audio (INT4)
	$(PYTHON) generate_onnx.py --model-dir $(OUTPUT_DIR) --variant int4 \
		--text "让每一家小企业，都拥有大企业的声音。" \
		--lang chinese \
		--instruct "Speak slowly in a calm and professional female voice" \
		-o sample_zh_int4.wav

clean: ## Remove output directory
	rm -rf $(OUTPUT_DIR)
