.PHONY: help check test test-qwen3 test-all fmt clippy doc bench-rtf bench-rtf-cuda bench-rtf-trt bench-csv bench-csv-cuda bench-csv-trt

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

check: fmt clippy test ## Run fmt, clippy, and test (no features)

fmt: ## Check formatting
	cargo fmt --all -- --check

clippy: ## Run clippy on all targets
	cargo clippy --all-targets -- -D warnings
	cargo clippy --all-targets --features qwen3-tts -- -D warnings

test: ## Run tests (no features)
	cargo test

test-qwen3: ## Run tests with qwen3-tts feature
	cargo test --features qwen3-tts

test-qwen3-cuda: ## Run tests with qwen3-tts + CUDA
	cargo test --features "qwen3-tts,cuda"

test-all: ## Run tests with all features
	cargo test --all-features

doc: ## Build and open docs
	cargo doc --all-features --no-deps --open

bench-rtf: ## RTF benchmark on CPU (int4)
	cargo run --release --example bench_rtf --features qwen3-tts

bench-rtf-cuda: ## RTF benchmark on CUDA (int4) — for Azure T4
	cargo run --release --example bench_rtf --features "qwen3-tts,cuda" -- --provider cuda

bench-rtf-trt: ## RTF benchmark on TensorRT (int4) — for Azure T4
	cargo run --release --example bench_rtf --features "qwen3-tts,tensorrt" -- --provider tensorrt

bench-csv: ## RTF benchmark on CPU (int4), save CSV to bench/results/
	@mkdir -p bench/results
	cargo run --release --example bench_rtf --features qwen3-tts -- --csv > bench/results/cpu-int4.csv

bench-csv-cuda: ## RTF benchmark on CUDA T4 (int4), save CSV to bench/results/
	@mkdir -p bench/results
	cargo run --release --example bench_rtf --features "qwen3-tts,cuda" -- \
		--provider cuda --hardware t4 --csv > bench/results/cuda-t4-int4.csv

bench-csv-trt: ## RTF benchmark on TensorRT T4 (int4), save CSV to bench/results/
	@mkdir -p bench/results
	cargo run --release --example bench_rtf --features "qwen3-tts,tensorrt" -- \
		--provider tensorrt --hardware t4 --csv > bench/results/trt-t4-int4.csv
