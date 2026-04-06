use rand::Rng;

/// Sampling configuration for token generation.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

/// Sample a token index from logits using top-p (nucleus) sampling.
///
/// `mask` is called with a token index and returns `true` if the token should
/// be suppressed (set to -inf before softmax).
pub fn sample(
    logits: &[f32],
    config: &SamplerConfig,
    past_tokens: &[i64],
    mask: impl Fn(usize) -> bool,
) -> usize {
    let mut scores: Vec<f32> = logits.to_vec();

    // 1. Mask forbidden tokens
    for (i, s) in scores.iter_mut().enumerate() {
        if mask(i) {
            *s = f32::NEG_INFINITY;
        }
    }

    // 2. Apply repetition penalty
    if config.repetition_penalty != 1.0 {
        for &tok in past_tokens {
            let idx = tok as usize;
            if idx < scores.len() {
                if scores[idx] > 0.0 {
                    scores[idx] /= config.repetition_penalty;
                } else {
                    scores[idx] *= config.repetition_penalty;
                }
            }
        }
    }

    // 3. Apply temperature
    if config.temperature != 1.0 && config.temperature > 0.0 {
        for s in &mut scores {
            *s /= config.temperature;
        }
    }

    // 4. Softmax
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }

    // 5. Top-p filtering
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0;
    let mut cutoff = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumulative += p;
        if cumulative >= config.top_p {
            cutoff = i + 1;
            break;
        }
    }
    let candidates = &indexed[..cutoff];

    // Renormalize
    let cand_sum: f32 = candidates.iter().map(|&(_, p)| p).sum();

    // 6. Sample
    let mut rng = rand::rng();
    let r: f32 = rng.random::<f32>() * cand_sum;
    let mut accum = 0.0;
    for &(idx, p) in candidates {
        accum += p;
        if accum >= r {
            return idx;
        }
    }

    // Fallback: return the highest-probability token
    candidates[0].0
}

/// Logit mask for the Talker LM (group 0).
///
/// Suppresses control tokens (2048..3072) except codec EOS (2150).
pub fn talker_mask(token: usize) -> bool {
    (2048..3072).contains(&token) && token != 2150
}

/// No masking for the Code Predictor (groups 1-15).
pub fn no_mask(_token: usize) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn talker_mask_allows_codebook_tokens() {
        for i in 0..2048 {
            assert!(!talker_mask(i), "token {i} should not be masked");
        }
    }

    #[test]
    fn talker_mask_blocks_control_tokens_except_eos() {
        assert!(!talker_mask(2150), "EOS should not be masked");
        assert!(talker_mask(2148), "PAD should be masked");
        assert!(talker_mask(2149), "BOS should be masked");
        assert!(talker_mask(2154), "THINK should be masked");
    }

    #[test]
    fn sample_returns_valid_index() {
        let logits = vec![0.0; 100];
        let config = SamplerConfig {
            temperature: 1.0,
            top_p: 0.9,
            repetition_penalty: 1.0,
        };
        let idx = sample(&logits, &config, &[], no_mask);
        assert!(idx < 100);
    }

    #[test]
    fn sample_respects_mask() {
        // All logits equal, but mask everything except token 5
        let logits = vec![0.0; 10];
        let config = SamplerConfig {
            temperature: 1.0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        };
        let idx = sample(&logits, &config, &[], |i| i != 5);
        assert_eq!(idx, 5);
    }
}
