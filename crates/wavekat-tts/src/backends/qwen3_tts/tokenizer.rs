use std::path::Path;

use crate::TtsError;

// Special text token IDs (Qwen3 BPE vocab)
pub const IM_START: u32 = 151644;
pub const ASSISTANT: u32 = 77091;
pub const NEWLINE: u32 = 198;
pub const TTS_BOS: u32 = 151672;
pub const TTS_EOS: u32 = 151673;
pub const TTS_PAD: u32 = 151671;

/// Language → codec language ID mapping.
pub fn language_id(lang: &str) -> Option<i64> {
    match lang {
        "en" | "english" => Some(2050),
        "ko" | "korean" => Some(2064),
        "zh" | "chinese" => Some(2055),
        "ja" | "japanese" => Some(2058),
        "de" | "german" => Some(2053),
        "es" | "spanish" => Some(2054),
        "fr" | "french" => Some(2061),
        "ru" | "russian" => Some(2069),
        "it" | "italian" => Some(2070),
        "pt" | "portuguese" => Some(2071),
        _ => None,
    }
}

/// Wraps a GPT-2 byte-level BPE tokenizer built from vocab.json + merges.txt.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Build tokenizer from `tokenizer/vocab.json` + `tokenizer/merges.txt` in `model_dir`.
    pub fn new(model_dir: &Path) -> Result<Self, TtsError> {
        let vocab_path = model_dir.join("tokenizer").join("vocab.json");
        let merges_path = model_dir.join("tokenizer").join("merges.txt");

        let bpe = tokenizers::models::bpe::BPE::from_file(
            &vocab_path.to_string_lossy(),
            &merges_path.to_string_lossy(),
        )
        .build()
        .map_err(|e| {
            TtsError::Model(format!(
                "failed to build BPE from {} + {}: {e}",
                vocab_path.display(),
                merges_path.display()
            ))
        })?;

        let mut inner = tokenizers::Tokenizer::new(bpe);

        // GPT-2 byte-level pre-tokenizer (same as Qwen2Tokenizer).
        inner.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, true, true),
        ));

        Ok(Self { inner })
    }

    /// Encode text into BPE token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TtsError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| TtsError::Synthesis(format!("tokenization failed: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }
}
