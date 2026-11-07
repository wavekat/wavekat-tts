//! Locate Kokoro model files: a local directory or the HF Hub cache.

use std::path::PathBuf;

use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::{Repo, RepoType};

use crate::TtsError;

pub(super) const REPO_ID: &str = "onnx-community/Kokoro-82M-v1.0-ONNX";
/// Pinned so a change upstream can't silently change the voice.
const REVISION: &str = "1939ad2a8e416c0acfeecc08a694d14ef25f2231";

/// Where model files come from.
pub(super) enum ModelSource {
    /// A directory that mirrors the HF repo layout
    /// (`tokenizer.json`, `onnx/…`, `voices/…`).
    Local(PathBuf),
    /// Downloaded on demand into the HF Hub cache.
    Hub(Box<ApiRepo>),
}

impl ModelSource {
    /// Resolve the source for `config`.
    ///
    /// Order: `config.model_dir`, then `WAVEKAT_KOKORO_MODEL_DIR` when it
    /// already holds `tokenizer.json`, then HF Hub (using
    /// `WAVEKAT_KOKORO_MODEL_DIR` as the cache root if it is set but empty).
    pub(super) fn resolve(config: &super::KokoroConfig) -> Result<Self, TtsError> {
        if let Some(dir) = &config.model_dir {
            return Ok(Self::Local(dir.clone()));
        }

        let cache_dir = match std::env::var("WAVEKAT_KOKORO_MODEL_DIR") {
            Ok(dir) if !dir.is_empty() => {
                let path = PathBuf::from(dir);
                if path.join("tokenizer.json").exists() {
                    return Ok(Self::Local(path));
                }
                Some(path)
            }
            _ => None,
        };

        // from_env() reads HF_HOME / HF_ENDPOINT; HF_TOKEN is bridged by hand
        // because hf-hub 0.5 doesn't read it.
        let mut builder = ApiBuilder::from_env().with_progress(false);
        if let Some(dir) = cache_dir {
            builder = builder.with_cache_dir(dir);
        }
        if let Ok(token) = std::env::var("HF_TOKEN") {
            if !token.is_empty() {
                builder = builder.with_token(Some(token));
            }
        }
        let api = builder
            .build()
            .map_err(|e| TtsError::Model(format!("failed to initialize HF Hub client: {e}")))?;
        Ok(Self::Hub(Box::new(api.repo(Repo::with_revision(
            REPO_ID.to_string(),
            RepoType::Model,
            REVISION.to_string(),
        )))))
    }

    /// Path to `name` (relative to the repo root), downloading it if needed.
    pub(super) fn file(&self, name: &str) -> Result<PathBuf, TtsError> {
        match self {
            Self::Local(dir) => {
                let path = dir.join(name);
                if path.exists() {
                    Ok(path)
                } else {
                    Err(TtsError::Model(format!(
                        "missing {name} in {}",
                        dir.display()
                    )))
                }
            }
            Self::Hub(repo) => repo
                .get(name)
                .map_err(|e| TtsError::Model(format!("failed to download {name}: {e}"))),
        }
    }
}
