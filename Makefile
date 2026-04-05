.PHONY: help check test test-qwen3 test-all fmt clippy doc

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

test-all: ## Run tests with all features
	cargo test --all-features

doc: ## Build and open docs
	cargo doc --all-features --no-deps --open
