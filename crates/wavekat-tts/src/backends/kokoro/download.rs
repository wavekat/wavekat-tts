//! Locate Kokoro model files: a local directory or the HF Hub cache.

use std::path::PathBuf;

use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use hf_hub::api::Progress;
use hf_hub::{Cache, Repo, RepoType};

use super::{DownloadCallback, DownloadProgress};
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
    Hub(Box<Hub>),
}

pub(super) struct Hub {
    api: ApiRepo,
    cache: Cache,
    on_download: Option<DownloadCallback>,
}

fn repo() -> Repo {
    Repo::with_revision(REPO_ID.to_string(), RepoType::Model, REVISION.to_string())
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
        let cache = match cache_dir {
            Some(dir) => {
                builder = builder.with_cache_dir(dir.clone());
                Cache::new(dir)
            }
            None => Cache::from_env(),
        };
        if let Ok(token) = std::env::var("HF_TOKEN") {
            if !token.is_empty() {
                builder = builder.with_token(Some(token));
            }
        }
        let api = builder
            .build()
            .map_err(|e| TtsError::Model(format!("failed to initialize HF Hub client: {e}")))?;
        Ok(Self::Hub(Box::new(Hub {
            api: api.repo(repo()),
            cache,
            on_download: config.on_download.clone(),
        })))
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
            Self::Hub(hub) => hub.file(name),
        }
    }
}

impl Hub {
    fn file(&self, name: &str) -> Result<PathBuf, TtsError> {
        let result = match &self.on_download {
            None => self.api.get(name),
            Some(callback) => match self.cache.repo(repo()).get(name) {
                Some(path) => Ok(path),
                None => self
                    .api
                    .download_with_progress(name, Reporter::new(name, callback)),
            },
        };
        result.map_err(|e| TtsError::Model(format!("failed to download {name}: {e}")))
    }
}

/// Bridges hf-hub's [`Progress`] to a [`DownloadCallback`].
struct Reporter<'a> {
    file: &'a str,
    callback: &'a DownloadCallback,
    downloaded: u64,
    total: u64,
}

impl<'a> Reporter<'a> {
    fn new(file: &'a str, callback: &'a DownloadCallback) -> Self {
        Self {
            file,
            callback,
            downloaded: 0,
            total: 0,
        }
    }

    fn report(&self) {
        self.callback.call(DownloadProgress {
            file: self.file,
            downloaded: self.downloaded,
            total: self.total,
        });
    }
}

impl Progress for Reporter<'_> {
    fn init(&mut self, size: usize, _filename: &str) {
        self.total = size as u64;
        self.downloaded = 0;
        self.report();
    }

    fn update(&mut self, size: usize) {
        self.downloaded = (self.downloaded + size as u64).min(self.total);
        self.report();
    }

    fn finish(&mut self) {
        self.downloaded = self.total;
        self.report();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn reporter_counts_bytes_up_to_total() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        let callback = DownloadCallback(Arc::new(move |p: DownloadProgress<'_>| {
            log.lock()
                .unwrap()
                .push((p.file.to_string(), p.downloaded, p.total));
        }));
        let mut r = Reporter::new("voices/af_heart.bin", &callback);
        r.init(100, "voices/af_heart.bin");
        r.update(40);
        r.update(80); // overshoot is clamped
        r.finish();
        let seen = seen.lock().unwrap();
        let bytes: Vec<_> = seen.iter().map(|s| (s.1, s.2)).collect();
        assert_eq!(bytes, [(0, 100), (40, 100), (100, 100), (100, 100)]);
        assert!(seen.iter().all(|s| s.0 == "voices/af_heart.bin"));
    }
}
