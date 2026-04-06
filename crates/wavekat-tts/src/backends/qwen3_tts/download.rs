//! Download model files from HuggingFace Hub.

use std::path::PathBuf;

use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};

use crate::TtsError;

const REPO_ID: &str = "wavekat/Qwen3-TTS-1.7B-VoiceDesign-ONNX";
const REVISION: &str = "2026-04-06";

/// ONNX model files for INT4 precision.
const ONNX_FILES_INT4: &[&str] = &[
    "int4/talker_prefill.onnx",
    "int4/talker_prefill.onnx.data",
    "int4/talker_decode.onnx",
    "int4/talker_decode.onnx.data",
    "int4/code_predictor.onnx",
    "int4/code_predictor.onnx.data",
    "int4/vocoder.onnx",
    "int4/vocoder.onnx.data",
];

/// ONNX model files for FP32 precision.
const ONNX_FILES_FP32: &[&str] = &[
    "fp32/talker_prefill.onnx",
    "fp32/talker_prefill.onnx.data",
    "fp32/talker_decode.onnx",
    "fp32/talker_decode.onnx.data",
    "fp32/code_predictor.onnx",
    "fp32/code_predictor.onnx.data",
    "fp32/vocoder.onnx",
    "fp32/vocoder.onnx.data",
];

/// Shared files required for all precision variants (embeddings + tokenizer + config).
///
/// `embeddings/small_to_mtp_projection_{weight,bias}.npy` are intentionally
/// excluded — that projection is baked into `code_predictor.onnx`.
const SHARED_FILES: &[&str] = &[
    "config.json",
    // Embedding tables
    "embeddings/text_embedding.npy",
    "embeddings/text_projection_fc1_weight.npy",
    "embeddings/text_projection_fc1_bias.npy",
    "embeddings/text_projection_fc2_weight.npy",
    "embeddings/text_projection_fc2_bias.npy",
    "embeddings/talker_codec_embedding.npy",
    "embeddings/cp_codec_embedding_0.npy",
    "embeddings/cp_codec_embedding_1.npy",
    "embeddings/cp_codec_embedding_2.npy",
    "embeddings/cp_codec_embedding_3.npy",
    "embeddings/cp_codec_embedding_4.npy",
    "embeddings/cp_codec_embedding_5.npy",
    "embeddings/cp_codec_embedding_6.npy",
    "embeddings/cp_codec_embedding_7.npy",
    "embeddings/cp_codec_embedding_8.npy",
    "embeddings/cp_codec_embedding_9.npy",
    "embeddings/cp_codec_embedding_10.npy",
    "embeddings/cp_codec_embedding_11.npy",
    "embeddings/cp_codec_embedding_12.npy",
    "embeddings/cp_codec_embedding_13.npy",
    "embeddings/cp_codec_embedding_14.npy",
    // Tokenizer
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
];

/// Resolve the local HF Hub snapshot directory for the Qwen3-TTS model,
/// downloading any missing files as needed.
///
/// Set `WAVEKAT_MODEL_DIR` to skip HF Hub and load from a local directory
/// that mirrors the repo layout (`int4/` or `fp32/`, `embeddings/`, `tokenizer/`).
///
/// Authentication: set `HF_TOKEN` if the repo requires it.  hf-hub 0.5 does
/// not read `HF_TOKEN` from the environment natively; this function bridges
/// the gap by passing it to `ApiBuilder::with_token`.
///
/// Cache location: `$HF_HOME/hub/` (default `~/.cache/huggingface/hub/`).
pub fn ensure_model_dir(precision: super::ModelPrecision) -> Result<PathBuf, TtsError> {
    if let Ok(dir) = std::env::var("WAVEKAT_MODEL_DIR") {
        return Ok(PathBuf::from(dir));
    }

    // from_env() reads HF_HOME / HF_ENDPOINT.
    // Bridge HF_TOKEN which hf-hub doesn't read from the environment natively.
    let mut builder = ApiBuilder::from_env();
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            builder = builder.with_token(Some(token));
        }
    }
    let api = builder
        .build()
        .map_err(|e| TtsError::Model(format!("failed to initialize HF Hub client: {e}")))?;

    let repo = api.repo(Repo::with_revision(
        REPO_ID.to_string(),
        RepoType::Model,
        REVISION.to_string(),
    ));

    let onnx_files = match precision {
        super::ModelPrecision::Int4 => ONNX_FILES_INT4,
        super::ModelPrecision::Fp32 => ONNX_FILES_FP32,
    };
    let total = 1 + onnx_files.len() + SHARED_FILES[1..].len(); // config + onnx + shared (excl. config)

    eprintln!("Ensuring Qwen3-TTS 1.7B ({}) model ({total} files from {REPO_ID})...", precision.subdir());

    // config.json first — its parent is the snapshot root.
    eprintln!("[1/{total}] {}", SHARED_FILES[0]);
    let config_path = repo
        .get(SHARED_FILES[0])
        .map_err(|e| TtsError::Model(format!("failed to download {}: {e}", SHARED_FILES[0])))?;

    let model_dir = config_path
        .parent()
        .ok_or_else(|| TtsError::Model("unexpected cache path for config.json".into()))?
        .to_path_buf();

    for (i, filename) in onnx_files.iter().chain(SHARED_FILES[1..].iter()).enumerate() {
        eprintln!("[{}/{total}] {filename}", i + 2);
        repo.get(filename)
            .map_err(|e| TtsError::Model(format!("failed to download {filename}: {e}")))?;
    }

    eprintln!("Files ready. Loading model ...");
    Ok(model_dir)
}
