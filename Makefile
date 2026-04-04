.PHONY: check test test-edge test-all fmt clippy doc

check: fmt clippy test

fmt:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets -- -D warnings
	cargo clippy --all-targets --features edge-tts -- -D warnings

test:
	cargo test

test-edge:
	cargo test --features edge-tts

test-all:
	cargo test --all-features

doc:
	cargo doc --all-features --no-deps --open

# Generate turn detection dataset (requires: pip install edge-tts && brew install ffmpeg)
dataset:
	cargo run --example batch_generate --features edge-tts
