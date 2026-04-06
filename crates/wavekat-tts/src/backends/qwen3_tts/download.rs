//! Auto-download model files to a local cache directory.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::TtsError;

/// Pinned commit — guarantees immutable file URLs.
const REVISION: &str = "6a297d9641354ef0c16e63d329a93a6239bca0a2";

const BASE_URL: &str = "https://huggingface.co/elbruno/Qwen3-TTS-12Hz-0.6B-Base-ONNX/resolve";

/// (remote_path, local_filename) — remote paths map to the HF repo layout,
/// local filenames are flattened into the cache directory.
const MODEL_FILES: &[(&str, &str)] = &[
    // ONNX sessions + external weight files
    ("talker_prefill.onnx", "talker_prefill.onnx"),
    ("talker_prefill.onnx.data", "talker_prefill.onnx.data"),
    ("talker_decode.onnx", "talker_decode.onnx"),
    ("talker_decode.onnx.data", "talker_decode.onnx.data"),
    ("code_predictor.onnx", "code_predictor.onnx"),
    ("vocoder.onnx", "vocoder.onnx"),
    ("vocoder.onnx.data", "vocoder.onnx.data"),
    // Embedding tables (in embeddings/ on HF, flattened locally)
    ("embeddings/text_embedding.npy", "text_embedding.npy"),
    (
        "embeddings/text_projection_fc1_weight.npy",
        "text_projection_fc1_weight.npy",
    ),
    (
        "embeddings/text_projection_fc1_bias.npy",
        "text_projection_fc1_bias.npy",
    ),
    (
        "embeddings/text_projection_fc2_weight.npy",
        "text_projection_fc2_weight.npy",
    ),
    (
        "embeddings/text_projection_fc2_bias.npy",
        "text_projection_fc2_bias.npy",
    ),
    (
        "embeddings/talker_codec_embedding.npy",
        "talker_codec_embedding.npy",
    ),
    (
        "embeddings/cp_codec_embedding_0.npy",
        "cp_codec_embedding_0.npy",
    ),
    (
        "embeddings/cp_codec_embedding_1.npy",
        "cp_codec_embedding_1.npy",
    ),
    (
        "embeddings/cp_codec_embedding_2.npy",
        "cp_codec_embedding_2.npy",
    ),
    (
        "embeddings/cp_codec_embedding_3.npy",
        "cp_codec_embedding_3.npy",
    ),
    (
        "embeddings/cp_codec_embedding_4.npy",
        "cp_codec_embedding_4.npy",
    ),
    (
        "embeddings/cp_codec_embedding_5.npy",
        "cp_codec_embedding_5.npy",
    ),
    (
        "embeddings/cp_codec_embedding_6.npy",
        "cp_codec_embedding_6.npy",
    ),
    (
        "embeddings/cp_codec_embedding_7.npy",
        "cp_codec_embedding_7.npy",
    ),
    (
        "embeddings/cp_codec_embedding_8.npy",
        "cp_codec_embedding_8.npy",
    ),
    (
        "embeddings/cp_codec_embedding_9.npy",
        "cp_codec_embedding_9.npy",
    ),
    (
        "embeddings/cp_codec_embedding_10.npy",
        "cp_codec_embedding_10.npy",
    ),
    (
        "embeddings/cp_codec_embedding_11.npy",
        "cp_codec_embedding_11.npy",
    ),
    (
        "embeddings/cp_codec_embedding_12.npy",
        "cp_codec_embedding_12.npy",
    ),
    (
        "embeddings/cp_codec_embedding_13.npy",
        "cp_codec_embedding_13.npy",
    ),
    (
        "embeddings/cp_codec_embedding_14.npy",
        "cp_codec_embedding_14.npy",
    ),
    // Tokenizer (in tokenizer/ on HF, flattened locally)
    ("tokenizer/vocab.json", "vocab.json"),
    ("tokenizer/merges.txt", "merges.txt"),
];

/// Resolve the default model cache directory, downloading any missing files.
///
/// Resolution order:
/// 1. `$WAVEKAT_MODEL_DIR` if set
/// 2. `$XDG_CACHE_HOME/wavekat/qwen3-tts-0.6b/`
/// 3. `$HOME/.cache/wavekat/qwen3-tts-0.6b/`
pub fn ensure_model_dir() -> Result<PathBuf, TtsError> {
    let dir = default_cache_dir()?;
    ensure_files(&dir)?;
    Ok(dir)
}

fn default_cache_dir() -> Result<PathBuf, TtsError> {
    if let Ok(dir) = std::env::var("WAVEKAT_MODEL_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let base = if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(xdg)
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".cache")
    } else {
        return Err(TtsError::Model(
            "cannot determine cache directory: set WAVEKAT_MODEL_DIR or HOME".into(),
        ));
    };
    Ok(base.join("wavekat").join("qwen3-tts-0.6b"))
}

fn ensure_files(dir: &Path) -> Result<(), TtsError> {
    let missing: Vec<(&str, &str)> = MODEL_FILES
        .iter()
        .filter(|(_, local)| !dir.join(local).exists())
        .copied()
        .collect();

    if missing.is_empty() {
        return Ok(());
    }

    fs::create_dir_all(dir).map_err(|e| {
        TtsError::Model(format!("failed to create cache dir {}: {e}", dir.display()))
    })?;

    let total = missing.len();
    eprintln!(
        "Downloading Qwen3-TTS model ({total} files) to {} ...",
        dir.display()
    );

    for (i, (remote, local)) in missing.iter().enumerate() {
        let url = format!("{BASE_URL}/{REVISION}/{remote}");
        let dest = dir.join(local);
        eprintln!("[{}/{}] {}", i + 1, total, local);
        download_file(&url, &dest)?;
    }

    eprintln!("Download complete.");
    Ok(())
}

fn download_file(url: &str, dest: &Path) -> Result<(), TtsError> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| TtsError::Model(format!("download failed for {url}: {e}")))?;

    let content_length: Option<u64> = response
        .header("Content-Length")
        .and_then(|s| s.parse().ok());

    let mut reader = response.into_reader();

    // Write to a temp file first, then rename to avoid partial files on interrupt.
    let tmp = dest.with_extension("tmp");
    let mut file = fs::File::create(&tmp)
        .map_err(|e| TtsError::Model(format!("failed to create {}: {e}", tmp.display())))?;

    let mut buf = [0u8; 256 * 1024];
    let mut downloaded: u64 = 0;
    let mut last_report: u64 = 0;

    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| TtsError::Model(format!("download read error: {e}")))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| TtsError::Model(format!("write error: {e}")))?;
        downloaded += n as u64;

        // Report progress every 50 MB for large files.
        if downloaded - last_report >= 50_000_000 {
            if let Some(total) = content_length {
                let mb = downloaded / 1_000_000;
                let total_mb = total / 1_000_000;
                eprint!("\r  {mb}/{total_mb} MB");
            }
            last_report = downloaded;
        }
    }

    // Clear progress line if we printed any.
    if last_report > 0 {
        eprintln!();
    }

    drop(file);
    fs::rename(&tmp, dest).map_err(|e| {
        TtsError::Model(format!(
            "failed to rename {} → {}: {e}",
            tmp.display(),
            dest.display()
        ))
    })?;

    Ok(())
}
