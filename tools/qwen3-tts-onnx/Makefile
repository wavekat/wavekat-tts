MODEL_ID ?= Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
OUTPUT_DIR ?= ./output/qwen3-tts-1.7b-voicedesign
PYTHON ?= python

FP32_DIR = $(OUTPUT_DIR)/fp32
INT4_DIR = $(OUTPUT_DIR)/int4

.PHONY: all export embeddings talker code-predictor vocoder validate quantize clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

all: export validate quantize ## Export FP32, validate, then quantize INT4

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
	$(PYTHON) generate_onnx.py --model-dir $(OUTPUT_DIR) --text "Hello, how are you today?"

generate-int4: ## Generate sample audio (INT4)
	$(PYTHON) generate_onnx.py --model-dir $(OUTPUT_DIR) --variant int4 --text "Hello, how are you today?"

clean: ## Remove output directory
	rm -rf $(OUTPUT_DIR)
