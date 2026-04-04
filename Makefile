.PHONY: check test test-kokoro test-all fmt clippy doc

check: fmt clippy test

fmt:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets -- -D warnings
	cargo clippy --all-targets --features kokoro -- -D warnings

test:
	cargo test

test-kokoro:
	cargo test --features kokoro

test-all:
	cargo test --all-features

doc:
	cargo doc --all-features --no-deps --open
