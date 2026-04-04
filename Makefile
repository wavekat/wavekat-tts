.PHONY: check test test-qwen3 test-all fmt clippy doc

check: fmt clippy test

fmt:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets -- -D warnings
	cargo clippy --all-targets --features qwen3-tts -- -D warnings

test:
	cargo test

test-qwen3:
	cargo test --features qwen3-tts

test-all:
	cargo test --all-features

doc:
	cargo doc --all-features --no-deps --open
