# GitHub Actions Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| [CI](ci.yml) | Push to `main`, PRs, manual | Runs `cargo fmt`, `clippy`, tests, and doc build |
| [Release-plz](release-plz.yml) | Push to `main`, manual | Automates crate releases to crates.io and opens release PRs |
| [Update benchmark table](update-bench.yml) | Push to `main` (when `bench/results/**.csv` changes), manual | Regenerates the benchmark table in README.md |
| [Export ONNX](export-onnx.yml) | Manual | Exports Qwen3-TTS to ONNX (FP32 + INT4), validates, and uploads to HuggingFace. Supports `voicedesign` (1.7B) and `clone` (0.6B Base) variants via input selector. |
