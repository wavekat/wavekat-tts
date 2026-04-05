use std::path::Path;

use crate::TtsError;

// Special text token IDs (Qwen3 BPE vocab)
const IM_START: u32 = 151644;
const ASSISTANT: u32 = 77091;
const NEWLINE: u32 = 198;
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

/// Wraps the HuggingFace BPE tokenizer.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    /// Load tokenizer from `vocab.json` + `merges.txt` in `model_dir`.
    pub fn new(model_dir: &Path) -> Result<Self, TtsError> {
        let vocab_path = model_dir.join("vocab.json");
        let merges_path = model_dir.join("merges.txt");

        let inner = tokenizers::Tokenizer::from_file(&vocab_path).map_err(|e| {
            TtsError::Model(format!(
                "failed to load tokenizer from {}: {e}",
                vocab_path.display()
            ))
        })?;

        // Verify merges file exists (tokenizer loads it via vocab.json config,
        // but we check explicitly for a clear error message).
        if !merges_path.exists() {
            return Err(TtsError::Model(format!(
                "merges.txt not found at {}",
                merges_path.display()
            )));
        }

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

/// One position in the prefill embedding sequence.
///
/// Each position contributes a text embedding and a codec embedding that get
/// summed together to form the final `inputs_embeds` vector.
#[derive(Debug, Clone)]
pub struct PrefillPosition {
    /// Text token ID → looked up in text_embedding + projected.
    pub text_token: u32,
    /// Codec token ID → looked up in talker_codec_embedding.
    pub codec_token: i64,
}

/// Build the full prefill token sequence for non-streaming synthesis.
///
/// Layout:
/// ```text
/// [im_start, assistant, \n]                   — role prefix (text only, codec=pad)
/// [tts_pad+think, tts_pad+think_bos,          — codec control prefix
///  tts_pad+lang_id, tts_pad+think_eos]
/// [tts_bos+pad]                               — start marker
/// [text_tok_0+pad, text_tok_1+pad, ...]       — text tokens
/// [tts_eos+pad]                               — end marker
/// [tts_pad+bos]                               — final bos
/// ```
pub fn build_prefill_sequence(text_tokens: &[u32], lang_id: i64) -> Vec<PrefillPosition> {
    // Codec control token IDs
    const CODEC_PAD: i64 = 2148;
    const CODEC_BOS: i64 = 2149;
    const CODEC_THINK: i64 = 2154;
    const CODEC_THINK_BOS: i64 = 2156;
    const CODEC_THINK_EOS: i64 = 2157;

    let mut seq = Vec::new();

    // Role prefix: <|im_start|> assistant \n
    for &text_tok in &[IM_START, ASSISTANT, NEWLINE] {
        seq.push(PrefillPosition {
            text_token: text_tok,
            codec_token: CODEC_PAD,
        });
    }

    // Codec control: think, think_bos, language, think_eos
    for &codec_tok in &[CODEC_THINK, CODEC_THINK_BOS, lang_id, CODEC_THINK_EOS] {
        seq.push(PrefillPosition {
            text_token: TTS_PAD,
            codec_token: codec_tok,
        });
    }

    // Start marker
    seq.push(PrefillPosition {
        text_token: TTS_BOS,
        codec_token: CODEC_PAD,
    });

    // Text tokens (each paired with codec pad)
    for &text_tok in text_tokens {
        seq.push(PrefillPosition {
            text_token: text_tok,
            codec_token: CODEC_PAD,
        });
    }

    // End marker
    seq.push(PrefillPosition {
        text_token: TTS_EOS,
        codec_token: CODEC_PAD,
    });

    // Final bos
    seq.push(PrefillPosition {
        text_token: TTS_PAD,
        codec_token: CODEC_BOS,
    });

    seq
}
