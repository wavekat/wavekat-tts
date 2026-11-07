//! Text → Kokoro token ids.
//!
//! misaki-rs turns English text into Kokoro's phoneme alphabet. It is built
//! without espeak-ng, so words missing from its lexicon go through
//! [`SpellFallback`]: split into known pieces where possible, otherwise
//! spelled letter by letter. That keeps names like `wavekat-voice` or `PR42`
//! audible instead of silently dropped.

use std::collections::HashMap;
use std::path::Path;

use misaki_rs::fallback::FallbackError;
use misaki_rs::{Fallback, Language, Lexicon, G2P};

use crate::TtsError;

/// Kokoro's context limit, excluding the two padding tokens.
pub(super) const MAX_TOKENS: usize = 510;

/// Phoneme → token id table from the model repo's `tokenizer.json`.
pub(super) struct Vocab {
    ids: HashMap<char, i64>,
}

impl Vocab {
    pub(super) fn load(path: &Path) -> Result<Self, TtsError> {
        let text = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| TtsError::Model(format!("bad tokenizer.json: {e}")))?;
        let vocab = json
            .pointer("/model/vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| TtsError::Model("tokenizer.json has no model.vocab".into()))?;
        let mut ids = HashMap::with_capacity(vocab.len());
        for (k, v) in vocab {
            let mut chars = k.chars();
            if let (Some(c), None, Some(id)) = (chars.next(), chars.next(), v.as_i64()) {
                ids.insert(c, id);
            }
        }
        Ok(Self { ids })
    }

    /// Token ids for a phoneme string. Characters outside the vocab are
    /// dropped, as the upstream tokenizer's normalizer does.
    pub(super) fn encode(&self, phonemes: &str) -> Vec<i64> {
        phonemes
            .chars()
            .filter_map(|c| self.ids.get(&c).copied())
            .collect()
    }
}

/// English G2P with a no-espeak fallback.
pub(super) struct Phonemizer {
    us: G2P,
    gb: G2P,
}

impl Phonemizer {
    pub(super) fn new() -> Self {
        let make = |lang| {
            let mut g = G2P::with_fallback(lang, Some(Box::new(SpellFallback::new(lang))));
            // Unknown pieces that still slip through become silence, not "❓".
            g.unk = String::new();
            g
        };
        Self {
            us: make(Language::EnglishUS),
            gb: make(Language::EnglishGB),
        }
    }

    /// Phonemes for `text`, split into chunks that each fit Kokoro's
    /// context window.
    pub(super) fn chunks(
        &self,
        text: &str,
        british: bool,
        vocab: &Vocab,
    ) -> Result<Vec<Vec<i64>>, TtsError> {
        let g2p = if british { &self.gb } else { &self.us };
        let mut out: Vec<Vec<i64>> = Vec::new();
        let mut current: Vec<i64> = Vec::new();
        let space = vocab.encode(" ");

        for sentence in split_sentences(text) {
            let (phonemes, _) = g2p
                .g2p(sentence)
                .map_err(|e| TtsError::Synthesis(format!("phonemize failed: {e}")))?;
            let ids = vocab.encode(phonemes.trim());
            if ids.is_empty() {
                continue;
            }
            for piece in split_long(&ids, &space) {
                let extra = if current.is_empty() { 0 } else { space.len() };
                if current.len() + extra + piece.len() > MAX_TOKENS {
                    out.push(std::mem::take(&mut current));
                }
                if !current.is_empty() {
                    current.extend_from_slice(&space);
                }
                current.extend_from_slice(piece);
            }
        }
        if !current.is_empty() {
            out.push(current);
        }
        Ok(out)
    }
}

/// Split on sentence-ending punctuation followed by whitespace.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let bytes: Vec<(usize, char)> = text.char_indices().collect();
    for (i, &(pos, c)) in bytes.iter().enumerate() {
        if matches!(c, '.' | '!' | '?' | '\n') {
            let next_is_space = bytes.get(i + 1).is_none_or(|&(_, n)| n.is_whitespace());
            if next_is_space {
                let end = pos + c.len_utf8();
                let s = text[start..end].trim();
                if !s.is_empty() {
                    out.push(s);
                }
                start = end;
            }
        }
    }
    let rest = text[start..].trim();
    if !rest.is_empty() {
        out.push(rest);
    }
    out
}

/// Break an over-long sentence at spaces so every piece fits.
fn split_long<'a>(ids: &'a [i64], space: &[i64]) -> Vec<&'a [i64]> {
    if ids.len() <= MAX_TOKENS {
        return vec![ids];
    }
    let space_id = space.first().copied();
    let mut out = Vec::new();
    let mut rest = ids;
    while rest.len() > MAX_TOKENS {
        let window = &rest[..MAX_TOKENS];
        let cut = space_id
            .and_then(|s| window.iter().rposition(|&t| t == s))
            .filter(|&p| p > 0)
            .unwrap_or(MAX_TOKENS);
        out.push(&rest[..cut]);
        rest = &rest[cut..];
        if rest.first().copied() == space_id {
            rest = &rest[1..];
        }
    }
    if !rest.is_empty() {
        out.push(rest);
    }
    out
}

/// Out-of-vocabulary fallback that needs no espeak-ng.
///
/// 1. Split the word at `-`, `_`, `.`, `/`, camel case and letter/digit
///    boundaries; look each piece up.
/// 2. A lowercase piece that is still unknown is tried as two known words
///    (`wavekat` → `wave` + `kat`), keeping pieces of 3+ letters.
/// 3. Anything left is spelled out: letters by name, digits as numbers.
pub(super) struct SpellFallback {
    lexicon: Lexicon,
    british: bool,
}

impl SpellFallback {
    pub(super) fn new(lang: Language) -> Self {
        Self {
            lexicon: Lexicon::new(lang),
            british: lang == Language::EnglishGB,
        }
    }

    fn known(&self, word: &str) -> Option<String> {
        if word.chars().count() < 2 {
            return None;
        }
        self.lexicon
            .get_word(word, "NN", None, None)
            .map(|(p, _)| p)
            .filter(|p| !p.is_empty())
    }

    fn piece(&self, piece: &str) -> String {
        if let Some(p) = self.known(piece) {
            return p;
        }
        if piece.chars().all(|c| c.is_ascii_digit()) {
            return piece
                .chars()
                .map(|c| digit(c).to_string())
                .collect::<Vec<_>>()
                .join(" ");
        }
        let lower = piece.to_lowercase();
        if lower != piece || piece.chars().count() >= 6 {
            // ALLCAPS (an acronym) is spelled; long lowercase words try a
            // two-word split first.
            if lower == piece {
                if let Some(p) = self.two_words(&lower) {
                    return p;
                }
            } else if piece.chars().skip(1).all(|c| c.is_lowercase()) {
                if let Some(p) = self.known(&lower).or_else(|| self.two_words(&lower)) {
                    return p;
                }
            }
        }
        self.spell(piece)
    }

    fn two_words(&self, word: &str) -> Option<String> {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() < 6 {
            return None;
        }
        // Prefer the split whose first word is longest.
        for cut in (3..=chars.len() - 3).rev() {
            let (a, b): (String, String) =
                (chars[..cut].iter().collect(), chars[cut..].iter().collect());
            if let (Some(pa), Some(pb)) =
                (self.known_or_soundalike(&a), self.known_or_soundalike(&b))
            {
                return Some(format!("{pa} {pb}"));
            }
        }
        None
    }

    /// A known word, or one spelled with `k` where English usually has `c`
    /// (`kat` → `cat`), which is common in made-up product names.
    fn known_or_soundalike(&self, word: &str) -> Option<String> {
        self.known(word).or_else(|| {
            let alt = word.replace('k', "c");
            (alt != word).then(|| self.known(&alt)).flatten()
        })
    }

    fn spell(&self, piece: &str) -> String {
        piece
            .chars()
            .filter_map(|c| {
                if c.is_ascii_digit() {
                    Some(digit(c).to_string())
                } else if c.is_ascii_alphabetic() {
                    Some(letter(c.to_ascii_lowercase(), self.british).to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Fallback for SpellFallback {
    fn phonemize(&self, word: &str) -> Result<String, FallbackError> {
        let out: Vec<String> = split_word(word)
            .into_iter()
            .map(|p| self.piece(&p))
            .filter(|p| !p.is_empty())
            .collect();
        if out.is_empty() {
            Err(FallbackError::NoPhonemes {
                word: word.to_string(),
            })
        } else {
            Ok(out.join(" "))
        }
    }
}

/// Split a token into pieces at separators, camel case and digit runs.
fn split_word(word: &str) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut cur = String::new();
    let mut prev: Option<char> = None;
    for c in word.chars() {
        if !c.is_alphanumeric() {
            if !cur.is_empty() {
                pieces.push(std::mem::take(&mut cur));
            }
            prev = None;
            continue;
        }
        if let Some(p) = prev {
            let boundary = (p.is_lowercase() && c.is_uppercase())
                || (p.is_alphabetic() && c.is_ascii_digit())
                || (p.is_ascii_digit() && c.is_alphabetic());
            if boundary && !cur.is_empty() {
                pieces.push(std::mem::take(&mut cur));
            }
        }
        cur.push(c);
        prev = Some(c);
    }
    if !cur.is_empty() {
        pieces.push(cur);
    }
    pieces
}

/// Letter names in misaki's phoneme alphabet.
fn letter(c: char, british: bool) -> &'static str {
    match c {
        'a' => "ˈA",
        'b' => "bˈi",
        'c' => "sˈi",
        'd' => "dˈi",
        'e' => "ˈi",
        'f' => "ˈɛf",
        'g' => "ʤˈi",
        'h' => "ˈAʧ",
        'i' => "ˈI",
        'j' => "ʤˈA",
        'k' => "kˈA",
        'l' => "ˈɛl",
        'm' => "ˈɛm",
        'n' => "ˈɛn",
        'o' => "ˈO",
        'p' => "pˈi",
        'q' => "kjˈu",
        'r' if british => "ˈɑː",
        'r' => "ˈɑɹ",
        's' => "ˈɛs",
        't' => "tˈi",
        'u' => "jˈu",
        'v' => "vˈi",
        'w' => "dˈʌbᵊljˌu",
        'x' => "ˈɛks",
        'y' => "wˈI",
        'z' if british => "zˈɛd",
        'z' => "zˈi",
        _ => "",
    }
}

fn digit(c: char) -> &'static str {
    match c {
        '0' => "zˈɪɹO",
        '1' => "wˈʌn",
        '2' => "tˈu",
        '3' => "θɹˈi",
        '4' => "fˈɔɹ",
        '5' => "fˈIv",
        '6' => "sˈɪks",
        '7' => "sˈɛvən",
        '8' => "ˈAt",
        '9' => "nˈIn",
        _ => "",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentences_split_on_end_punctuation() {
        assert_eq!(
            split_sentences("Hi there. Version 1.2 is out! Ok?"),
            vec!["Hi there.", "Version 1.2 is out!", "Ok?"]
        );
        assert_eq!(split_sentences("no end"), vec!["no end"]);
        assert!(split_sentences("   ").is_empty());
    }

    #[test]
    fn words_split_at_case_digits_and_separators() {
        assert_eq!(split_word("wavekat-voice"), vec!["wavekat", "voice"]);
        assert_eq!(split_word("PR42"), vec!["PR", "42"]);
        assert_eq!(split_word("myRepoName"), vec!["my", "Repo", "Name"]);
        assert_eq!(
            split_word("feat/kokoro_backend"),
            vec!["feat", "kokoro", "backend"]
        );
    }

    #[test]
    fn long_runs_split_at_spaces_within_limit() {
        let space = vec![16];
        let mut ids = Vec::new();
        for _ in 0..200 {
            ids.extend_from_slice(&[50, 51, 16]);
        }
        let parts = split_long(&ids, &space);
        assert!(parts.len() >= 2);
        assert!(parts.iter().all(|p| p.len() <= MAX_TOKENS));
        assert!(parts.iter().all(|p| p.first() != Some(&16)));
    }

    #[test]
    fn fallback_speaks_unknown_words() {
        let fb = SpellFallback::new(Language::EnglishUS);
        // A made-up word is spelled, never empty.
        let out = fb.phonemize("qzx").unwrap();
        assert_eq!(out, "kjˈu zˈi ˈɛks");
        // A compound of two known words is read as words.
        let out = fb.phonemize("wavekat-voice").unwrap();
        assert!(!out.is_empty());
        assert!(!out.contains('❓'));
        // `k` for `c` in a made-up name still splits into words.
        let out = fb.phonemize("wavekat").unwrap();
        assert_ne!(out, fb.spell("wavekat"));
        // Digits are read as numbers.
        assert_eq!(fb.phonemize("42").unwrap(), "fˈɔɹ tˈu");
    }
}
