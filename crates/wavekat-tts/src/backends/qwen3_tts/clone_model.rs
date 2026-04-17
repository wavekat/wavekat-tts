//! Voice-clone ONNX pipeline for Qwen3-TTS 0.6B Base.
//!
//! Chains 6 ONNX models: tokenizer_encoder → speaker_encoder → talker (prefill
//! + decode loop) → code_predictor → vocoder.
//!
//! Reference audio codes are prepended to the generated codes before vocoding,
//! then the leading reference portion is trimmed proportionally.

use std::path::Path;
use std::sync::Mutex;

use ndarray::{concatenate, s, Array1, Array2, Array3, Array5, Axis};
use ort::session::Session;
use ort::value::TensorRef;
use wavekat_core::AudioFrame;

use crate::TtsError;

use super::mel::MelSpectrogram;
use super::model::{
    apply_execution_provider, load_npy1, load_npy2, prepare_onnx_dir, text_project,
};
use super::sampler::{self, SamplerConfig};
use super::tokenizer::{self, ASSISTANT, IM_START, NEWLINE, TTS_BOS, TTS_EOS, TTS_PAD};

// Codec control token IDs (shared across all Qwen3-TTS variants)
const CODEC_PAD: i64 = 2148;
const CODEC_BOS: i64 = 2149;
const CODEC_EOS: i64 = 2150;
const CODEC_THINK: i64 = 2154;
const CODEC_THINK_BOS: i64 = 2156;
const CODEC_THINK_EOS: i64 = 2157;

// Model dimensions — Qwen3-TTS-12Hz-0.6B-Base
const HIDDEN_DIM: usize = 1024;
const NUM_LAYERS: usize = 28;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const TALKER_VOCAB_SIZE: usize = 3072;
const CP_NUM_LAYERS: usize = 5;
const CP_NUM_KV_HEADS: usize = 8;
const NUM_CP_GROUPS: usize = 15; // codebook groups 1-15
const SAMPLE_RATE: u32 = 24000;
const MAX_NEW_TOKENS: usize = 8192;

// Tokenizer encoder constants
const TOKENIZER_CANONICAL_SAMPLES: usize = 240_000; // 10s @ 24kHz
const TOKENIZER_DOWNSAMPLE: usize = 1920; // 24000 Hz / 12.5 Hz

/// Sampling defaults from config.json.
const TALKER_SAMPLER: SamplerConfig = SamplerConfig {
    temperature: 0.9,
    top_k: 50,
    repetition_penalty: 1.05,
};

const CP_SAMPLER: SamplerConfig = SamplerConfig {
    temperature: 0.9,
    top_k: 50,
    repetition_penalty: 1.0,
};

/// Talker output: (logits, hidden_state, kv_keys, kv_values).
type TalkerOutput = (Vec<f32>, Array3<f32>, Array5<f32>, Array5<f32>);

/// All ONNX sessions and embedding tables for the 0.6B Base voice clone pipeline.
pub struct CloneModel {
    talker_prefill: Mutex<Session>,
    talker_decode: Mutex<Session>,
    code_predictor: Mutex<Session>,
    vocoder: Mutex<Session>,
    speaker_encoder: Mutex<Session>,
    tokenizer_encoder: Mutex<Session>,

    // Embedding tables (immutable after construction)
    text_embedding: Array2<f32>,
    text_proj_fc1_weight: Array2<f32>,
    text_proj_fc1_bias: Array1<f32>,
    text_proj_fc2_weight: Array2<f32>,
    text_proj_fc2_bias: Array1<f32>,
    talker_codec_embedding: Array2<f32>,
    cp_codec_embeddings: Vec<Array2<f32>>,

    // Precomputed
    tts_pad_embed: Array1<f32>,
    mel: MelSpectrogram,
}

impl CloneModel {
    /// Load all 6 ONNX sessions and embedding tables from `model_dir`.
    pub fn load(model_dir: &Path, config: &super::ModelConfig) -> Result<Self, TtsError> {
        let onnx_dir = prepare_onnx_dir(&model_dir.join(config.precision.subdir()))?;

        let load_session = |name: &str, dir: &Path| -> Result<Session, TtsError> {
            let path = dir.join(name);
            let builder = Session::builder()
                .map_err(|e| TtsError::Model(format!("session builder error: {e}")))?;
            apply_execution_provider(builder, config.execution_provider)?
                .commit_from_file(&path)
                .map_err(|e| TtsError::Model(format!("failed to load {name}: {e}")))
        };

        eprint!("Loading talker prefill     ... ");
        let talker_prefill = load_session("talker_prefill.onnx", &onnx_dir)?;
        eprintln!("done");

        eprint!("Loading talker decode      ... ");
        let talker_decode = load_session("talker_decode.onnx", &onnx_dir)?;
        eprintln!("done");

        eprint!("Loading code predictor     ... ");
        let code_predictor = load_session("code_predictor.onnx", &onnx_dir)?;
        eprintln!("done");

        eprint!("Loading vocoder            ... ");
        let vocoder = load_session("vocoder.onnx", &onnx_dir)?;
        eprintln!("done");

        // Speaker encoder and tokenizer encoder are FP32-only, stored at model root
        eprint!("Loading speaker encoder    ... ");
        let speaker_encoder = load_session("speaker_encoder.onnx", model_dir)?;
        eprintln!("done");

        eprint!("Loading tokenizer encoder  ... ");
        let tokenizer_encoder = load_session("tokenizer_encoder.onnx", model_dir)?;
        eprintln!("done");

        eprint!("Loading embeddings         ... ");
        let text_embedding = load_npy2(model_dir, "embeddings/text_embedding.npy")?;
        let text_proj_fc1_weight =
            load_npy2(model_dir, "embeddings/text_projection_fc1_weight.npy")?;
        let text_proj_fc1_bias = load_npy1(model_dir, "embeddings/text_projection_fc1_bias.npy")?;
        let text_proj_fc2_weight =
            load_npy2(model_dir, "embeddings/text_projection_fc2_weight.npy")?;
        let text_proj_fc2_bias = load_npy1(model_dir, "embeddings/text_projection_fc2_bias.npy")?;
        let talker_codec_embedding = load_npy2(model_dir, "embeddings/talker_codec_embedding.npy")?;

        let mut cp_codec_embeddings = Vec::with_capacity(NUM_CP_GROUPS);
        for i in 0..NUM_CP_GROUPS {
            cp_codec_embeddings.push(load_npy2(
                model_dir,
                &format!("embeddings/cp_codec_embedding_{i}.npy"),
            )?);
        }
        eprintln!("done");

        let tts_pad_raw = text_embedding.row(TTS_PAD as usize).to_owned();
        let tts_pad_embed = text_project(
            &tts_pad_raw,
            &text_proj_fc1_weight,
            &text_proj_fc1_bias,
            &text_proj_fc2_weight,
            &text_proj_fc2_bias,
        );

        eprintln!("Clone model ready.");

        Ok(Self {
            talker_prefill: Mutex::new(talker_prefill),
            talker_decode: Mutex::new(talker_decode),
            code_predictor: Mutex::new(code_predictor),
            vocoder: Mutex::new(vocoder),
            speaker_encoder: Mutex::new(speaker_encoder),
            tokenizer_encoder: Mutex::new(tokenizer_encoder),
            text_embedding,
            text_proj_fc1_weight,
            text_proj_fc1_bias,
            text_proj_fc2_weight,
            text_proj_fc2_bias,
            talker_codec_embedding,
            cp_codec_embeddings,
            tts_pad_embed,
            mel: MelSpectrogram::new(),
        })
    }

    /// Run the full voice-clone pipeline.
    ///
    /// `pcm_24k`     — reference audio resampled to 24 kHz mono
    /// `ref_tokens`  — tokenized reference transcript
    /// `text_tokens` — tokenized target text
    /// `language`    — language code (e.g. "en")
    pub fn synthesize(
        &self,
        pcm_24k: &[f32],
        ref_tokens: &[u32],
        text_tokens: &[u32],
        language: &str,
    ) -> Result<AudioFrame<'static>, TtsError> {
        let lang_id = tokenizer::language_id(language)
            .ok_or_else(|| TtsError::UnsupportedLanguage(language.to_string()))?;

        // 1. Speaker embedding from mel spectrogram
        let speaker_embed = self.encode_speaker(pcm_24k)?;

        // 2. Reference codes from tokenizer encoder
        let ref_codes = self.encode_ref_codes(pcm_24k)?;
        let ref_frames = ref_codes.nrows();

        // 3. Build ICL prefill embeddings
        let prefill_embeds =
            self.build_icl_prefill(ref_tokens, text_tokens, lang_id, &speaker_embed, &ref_codes)?;
        let prefill_len = prefill_embeds.shape()[1];

        // 4. Run talker prefill
        let (logits, hidden_states, mut past_keys, mut past_values) =
            self.run_talker_prefill(&prefill_embeds, prefill_len)?;

        // 5. Autoregressive decode loop
        let mut all_codes: Vec<[i64; 16]> = Vec::new();
        let mut talker_past_tokens: Vec<i64> = Vec::new();
        let mut current_logits = logits;

        let mut current_hidden = hidden_states
            .slice(s![0, prefill_len - 1.., ..])
            .to_owned()
            .into_shape_with_order((1, 1, HIDDEN_DIM))
            .map_err(|e| TtsError::Synthesis(format!("reshape hidden: {e}")))?;

        for step in 0..MAX_NEW_TOKENS {
            let group0 = sampler::sample(
                &current_logits,
                &TALKER_SAMPLER,
                &talker_past_tokens,
                |tok| sampler::talker_mask(tok) || (step < 2 && tok == CODEC_EOS as usize),
            ) as i64;

            if group0 == CODEC_EOS {
                break;
            }
            talker_past_tokens.push(group0);

            // Code predictor for groups 1-15
            let mut codes = [0i64; 16];
            codes[0] = group0;
            self.run_code_predictor(&current_hidden, &mut codes)?;
            all_codes.push(codes);

            // Next talker input: sum all 16 group embeddings + tts_pad
            let mut next_embed = self.talker_codec_embedding.row(group0 as usize).to_owned();
            for g in 0..NUM_CP_GROUPS {
                let cp_embed = self.cp_codec_embeddings[g].row(codes[g + 1] as usize);
                next_embed += &cp_embed;
            }
            next_embed += &self.tts_pad_embed;

            let next_embed = next_embed
                .into_shape_with_order((1, 1, HIDDEN_DIM))
                .map_err(|e| TtsError::Synthesis(format!("reshape next_embed: {e}")))?;

            let total_seq = prefill_len + step + 1;
            let position = (prefill_len + step) as i64;

            let (new_logits, new_hidden, new_keys, new_values) =
                self.run_talker_decode(&next_embed, total_seq, position, &past_keys, &past_values)?;

            current_logits = new_logits;
            current_hidden = new_hidden;
            past_keys = new_keys;
            past_values = new_values;
        }

        if all_codes.is_empty() {
            return Err(TtsError::Synthesis("model produced no audio tokens".into()));
        }

        // 6. Vocoder: prepend ref codes, decode, trim reference portion
        self.run_vocoder_clone(&all_codes, &ref_codes, ref_frames)
    }

    // ------------------------------------------------------------------
    // Reference-audio preprocessing
    // ------------------------------------------------------------------

    /// Compute mel spectrogram → run speaker_encoder.onnx → (1024,) speaker embed.
    fn encode_speaker(&self, pcm_24k: &[f32]) -> Result<Array1<f32>, TtsError> {
        let mel = self.mel.compute(pcm_24k); // (T_mel, 128)
        let n_frames = mel.nrows();
        let mel_3d = mel
            .into_shape_with_order((1, n_frames, 128))
            .map_err(|e| TtsError::Synthesis(format!("reshape mel: {e}")))?;

        let t_mel = TensorRef::from_array_view(&mel_3d)
            .map_err(|e| TtsError::Synthesis(format!("tensor mel: {e}")))?;

        let mut session = self.speaker_encoder.lock().unwrap();
        let outputs = session
            .run(ort::inputs!["mels" => t_mel])
            .map_err(|e| TtsError::Synthesis(format!("speaker encoder failed: {e}")))?;

        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::Synthesis(format!("extract speaker embed: {e}")))?;

        Ok(Array1::from(data.to_vec()))
    }

    /// Zero-pad audio to canonical 10 s → run tokenizer_encoder.onnx → (T_ref, 16) codes.
    fn encode_ref_codes(&self, pcm_24k: &[f32]) -> Result<Array2<i64>, TtsError> {
        let n = pcm_24k.len().min(TOKENIZER_CANONICAL_SAMPLES);
        let mut padded = vec![0.0f32; TOKENIZER_CANONICAL_SAMPLES];
        padded[..n].copy_from_slice(&pcm_24k[..n]);

        let waveform = Array2::from_shape_vec((1, TOKENIZER_CANONICAL_SAMPLES), padded)
            .map_err(|e| TtsError::Synthesis(format!("reshape waveform: {e}")))?;

        let t_wav = TensorRef::from_array_view(&waveform)
            .map_err(|e| TtsError::Synthesis(format!("tensor waveform: {e}")))?;

        let mut session = self.tokenizer_encoder.lock().unwrap();
        let outputs = session
            .run(ort::inputs!["waveform" => t_wav])
            .map_err(|e| TtsError::Synthesis(format!("tokenizer encoder failed: {e}")))?;

        // Output shape: (1, 16, 125)
        let (_, codes_data) = outputs[0]
            .try_extract_tensor::<i64>()
            .map_err(|e| TtsError::Synthesis(format!("extract ref codes: {e}")))?;

        let actual_frames = (n as f64 / TOKENIZER_DOWNSAMPLE as f64).ceil() as usize;

        // Reshape (1, 16, 125) → take only actual_frames, transpose to (T_ref, 16)
        let full = Array3::from_shape_vec((1, 16, 125), codes_data.to_vec())
            .map_err(|e| TtsError::Synthesis(format!("reshape codes: {e}")))?;

        let trimmed = full.slice(s![0, .., ..actual_frames]).t().to_owned(); // (actual_frames, 16)

        Ok(trimmed)
    }

    // ------------------------------------------------------------------
    // ICL prefill construction
    // ------------------------------------------------------------------

    /// Project a text token through the embedding table + SiLU MLP.
    fn text_project_token(&self, token: u32) -> Array1<f32> {
        let raw = self.text_embedding.row(token as usize).to_owned();
        text_project(
            &raw,
            &self.text_proj_fc1_weight,
            &self.text_proj_fc1_bias,
            &self.text_proj_fc2_weight,
            &self.text_proj_fc2_bias,
        )
    }

    /// Build ICL prefill embeddings for the voice-clone pipeline.
    ///
    /// Layout:
    /// ```text
    /// [im_start, assistant, \n]                        — role prefix (text_proj only)
    /// [tts_pad + codec(think)]
    /// [tts_pad + codec(think_bos)]                     — codec think prefix
    /// [tts_pad + codec(lang_id)]
    /// [tts_pad + codec(think_eos)]
    /// [tts_pad + speaker_embed]                        — speaker slot
    /// [tts_bos + codec_pad]                            — transition
    /// ICL block:
    ///   text side:  [text_proj(ref_tokens ++ text_tokens), tts_eos] + codec_pad  (T1)
    ///   codec side: [codec_bos, Σ_g codec_embed_g[ref_code]] + tts_pad           (T2)
    /// ```
    fn build_icl_prefill(
        &self,
        ref_tokens: &[u32],
        text_tokens: &[u32],
        lang_id: i64,
        speaker_embed: &Array1<f32>,
        ref_codes: &Array2<i64>, // (T_ref, 16)
    ) -> Result<Array3<f32>, TtsError> {
        let codec_pad_embed = self
            .talker_codec_embedding
            .row(CODEC_PAD as usize)
            .to_owned();
        let codec_bos_embed = self
            .talker_codec_embedding
            .row(CODEC_BOS as usize)
            .to_owned();
        let tts_bos_embed = self.text_project_token(TTS_BOS);
        let tts_eos_embed = self.text_project_token(TTS_EOS);

        let codec_prefix = [CODEC_THINK, CODEC_THINK_BOS, lang_id, CODEC_THINK_EOS];
        let ref_frames = ref_codes.nrows();

        // ICL text side: text_proj(ref_tokens ++ text_tokens) | tts_eos
        let combined_text_len = ref_tokens.len() + text_tokens.len();
        let t1 = combined_text_len + 1; // +1 for tts_eos

        // ICL codec side: codec_bos | Σ_g codec_embed per ref frame
        let t2 = 1 + ref_frames;

        // Total: 3 role + 4 codec_prefix + 1 speaker + 1 transition + T1 + T2
        let seq_len = 3 + codec_prefix.len() + 1 + 1 + t1 + t2;
        let mut embeds = Array3::<f32>::zeros((1, seq_len, HIDDEN_DIM));
        let mut pos = 0;

        // Part A: Role prefix (3 tokens, text_proj only)
        for &tok in &[IM_START, ASSISTANT, NEWLINE] {
            let embed = self.text_project_token(tok);
            embeds.slice_mut(s![0, pos, ..]).assign(&embed);
            pos += 1;
        }

        // Part B: Codec think prefix — tts_pad + codec_embed(token)
        for &codec_tok in &codec_prefix {
            let mut embed = self.tts_pad_embed.clone();
            embed += &self.talker_codec_embedding.row(codec_tok as usize);
            embeds.slice_mut(s![0, pos, ..]).assign(&embed);
            pos += 1;
        }

        // Part C: Speaker slot — tts_pad + speaker_embed
        {
            let embed = &self.tts_pad_embed + speaker_embed;
            embeds.slice_mut(s![0, pos, ..]).assign(&embed);
            pos += 1;
        }

        // Part D: Transition — tts_bos + codec_pad
        {
            let embed = &tts_bos_embed + &codec_pad_embed;
            embeds.slice_mut(s![0, pos, ..]).assign(&embed);
            pos += 1;
        }

        // Part E: ICL block
        // Text side: text_proj(ref_tokens ++ text_tokens) | tts_eos, all + codec_pad
        for &tok in ref_tokens.iter().chain(text_tokens.iter()) {
            let embed = self.text_project_token(tok) + &codec_pad_embed;
            embeds.slice_mut(s![0, pos, ..]).assign(&embed);
            pos += 1;
        }
        {
            let embed = &tts_eos_embed + &codec_pad_embed;
            embeds.slice_mut(s![0, pos, ..]).assign(&embed);
            pos += 1;
        }

        // Codec side: codec_bos + tts_pad, then Σ codec_embed_g[ref_code] + tts_pad per frame
        {
            let embed = &codec_bos_embed + &self.tts_pad_embed;
            embeds.slice_mut(s![0, pos, ..]).assign(&embed);
            pos += 1;
        }
        for f in 0..ref_frames {
            // Group 0: talker codec embedding
            let mut embed = self
                .talker_codec_embedding
                .row(ref_codes[[f, 0]] as usize)
                .to_owned();
            // Groups 1-15: CP codec embeddings
            for g in 0..NUM_CP_GROUPS {
                embed += &self.cp_codec_embeddings[g].row(ref_codes[[f, g + 1]] as usize);
            }
            embed += &self.tts_pad_embed;
            embeds.slice_mut(s![0, pos, ..]).assign(&embed);
            pos += 1;
        }

        debug_assert_eq!(pos, seq_len);
        Ok(embeds)
    }

    // ------------------------------------------------------------------
    // Talker prefill / decode (same structure as model.rs, 1024-dim)
    // ------------------------------------------------------------------

    fn run_talker_prefill(
        &self,
        inputs_embeds: &Array3<f32>,
        seq_len: usize,
    ) -> Result<TalkerOutput, TtsError> {
        let attention_mask = Array2::<i64>::ones((1, seq_len));

        let positions: Vec<i64> = (0..seq_len as i64).collect();
        let pos_2d = Array1::from(positions)
            .into_shape_with_order((1, seq_len))
            .map_err(|e| TtsError::Synthesis(format!("reshape pos: {e}")))?;
        let position_ids = ndarray::stack(Axis(0), &[pos_2d.view(), pos_2d.view(), pos_2d.view()])
            .map_err(|e| TtsError::Synthesis(format!("stack pos: {e}")))?;

        let t_embeds = TensorRef::from_array_view(inputs_embeds)
            .map_err(|e| TtsError::Synthesis(format!("tensor inputs_embeds: {e}")))?;
        let t_mask = TensorRef::from_array_view(&attention_mask)
            .map_err(|e| TtsError::Synthesis(format!("tensor mask: {e}")))?;
        let t_pos = TensorRef::from_array_view(&position_ids)
            .map_err(|e| TtsError::Synthesis(format!("tensor pos: {e}")))?;

        let mut session = self.talker_prefill.lock().unwrap();
        let outputs = session
            .run(ort::inputs![
                "inputs_embeds" => t_embeds,
                "attention_mask" => t_mask,
                "position_ids" => t_pos,
            ])
            .map_err(|e| TtsError::Synthesis(format!("talker prefill failed: {e}")))?;

        let (_, logits_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::Synthesis(format!("extract logits: {e}")))?;
        let logits: Vec<f32> = logits_data[logits_data.len() - TALKER_VOCAB_SIZE..].to_vec();

        let (_, hidden_data) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::Synthesis(format!("extract hidden: {e}")))?;
        let hidden = Array3::from_shape_vec((1, seq_len, HIDDEN_DIM), hidden_data.to_vec())
            .map_err(|e| TtsError::Synthesis(format!("reshape hidden: {e}")))?;

        let mut key_layers = Vec::with_capacity(NUM_LAYERS);
        let mut value_layers = Vec::with_capacity(NUM_LAYERS);
        for layer in 0..NUM_LAYERS {
            let key_idx = 2 + layer * 2;
            let val_idx = 2 + layer * 2 + 1;

            let (_, key_data) = outputs[key_idx]
                .try_extract_tensor::<f32>()
                .map_err(|e| TtsError::Synthesis(format!("extract key layer {layer}: {e}")))?;
            let (_, val_data) = outputs[val_idx]
                .try_extract_tensor::<f32>()
                .map_err(|e| TtsError::Synthesis(format!("extract val layer {layer}: {e}")))?;

            let key_arr = ndarray::ArrayD::from_shape_vec(
                vec![1, NUM_KV_HEADS, seq_len, HEAD_DIM],
                key_data.to_vec(),
            )
            .map_err(|e| TtsError::Synthesis(format!("reshape key {layer}: {e}")))?
            .insert_axis(Axis(0));
            let val_arr = ndarray::ArrayD::from_shape_vec(
                vec![1, NUM_KV_HEADS, seq_len, HEAD_DIM],
                val_data.to_vec(),
            )
            .map_err(|e| TtsError::Synthesis(format!("reshape val {layer}: {e}")))?
            .insert_axis(Axis(0));

            key_layers.push(key_arr);
            value_layers.push(val_arr);
        }

        let past_keys = concatenate(
            Axis(0),
            &key_layers.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .map_err(|e| TtsError::Synthesis(format!("stack keys: {e}")))?
        .into_shape_with_order((NUM_LAYERS, 1, NUM_KV_HEADS, seq_len, HEAD_DIM))
        .map_err(|e| TtsError::Synthesis(format!("reshape stacked keys: {e}")))?;

        let past_values = concatenate(
            Axis(0),
            &value_layers.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .map_err(|e| TtsError::Synthesis(format!("stack values: {e}")))?
        .into_shape_with_order((NUM_LAYERS, 1, NUM_KV_HEADS, seq_len, HEAD_DIM))
        .map_err(|e| TtsError::Synthesis(format!("reshape stacked values: {e}")))?;

        Ok((logits, hidden, past_keys, past_values))
    }

    fn run_talker_decode(
        &self,
        inputs_embeds: &Array3<f32>,
        total_seq: usize,
        position: i64,
        past_keys: &Array5<f32>,
        past_values: &Array5<f32>,
    ) -> Result<TalkerOutput, TtsError> {
        let attention_mask = Array2::<i64>::ones((1, total_seq));
        let position_ids = Array3::<i64>::from_elem((3, 1, 1), position);

        let t_embeds = TensorRef::from_array_view(inputs_embeds)
            .map_err(|e| TtsError::Synthesis(format!("tensor embeds: {e}")))?;
        let t_mask = TensorRef::from_array_view(&attention_mask)
            .map_err(|e| TtsError::Synthesis(format!("tensor mask: {e}")))?;
        let t_pos = TensorRef::from_array_view(&position_ids)
            .map_err(|e| TtsError::Synthesis(format!("tensor pos: {e}")))?;
        let t_keys = TensorRef::from_array_view(past_keys)
            .map_err(|e| TtsError::Synthesis(format!("tensor past_keys: {e}")))?;
        let t_values = TensorRef::from_array_view(past_values)
            .map_err(|e| TtsError::Synthesis(format!("tensor past_values: {e}")))?;

        let mut session = self.talker_decode.lock().unwrap();
        let outputs = session
            .run(ort::inputs![
                "inputs_embeds" => t_embeds,
                "attention_mask" => t_mask,
                "position_ids" => t_pos,
                "past_keys" => t_keys,
                "past_values" => t_values,
            ])
            .map_err(|e| TtsError::Synthesis(format!("talker decode failed: {e}")))?;

        let (_, logits_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::Synthesis(format!("extract decode logits: {e}")))?;
        let logits = logits_data.to_vec();

        let (_, hidden_data) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::Synthesis(format!("extract decode hidden: {e}")))?;
        let hidden = Array3::from_shape_vec((1, 1, HIDDEN_DIM), hidden_data.to_vec())
            .map_err(|e| TtsError::Synthesis(format!("reshape decode hidden: {e}")))?;

        let (_, keys_data) = outputs[2]
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::Synthesis(format!("extract decode keys: {e}")))?;
        let new_keys = Array5::from_shape_vec(
            (NUM_LAYERS, 1, NUM_KV_HEADS, total_seq, HEAD_DIM),
            keys_data.to_vec(),
        )
        .map_err(|e| TtsError::Synthesis(format!("reshape decode keys: {e}")))?;

        let (_, values_data) = outputs[3]
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::Synthesis(format!("extract decode values: {e}")))?;
        let new_values = Array5::from_shape_vec(
            (NUM_LAYERS, 1, NUM_KV_HEADS, total_seq, HEAD_DIM),
            values_data.to_vec(),
        )
        .map_err(|e| TtsError::Synthesis(format!("reshape decode values: {e}")))?;

        Ok((logits, hidden, new_keys, new_values))
    }

    // ------------------------------------------------------------------
    // Code predictor (groups 1-15)
    // ------------------------------------------------------------------

    fn run_code_predictor(
        &self,
        hidden_state: &Array3<f32>,
        codes: &mut [i64; 16],
    ) -> Result<(), TtsError> {
        let group0_embed = self
            .talker_codec_embedding
            .row(codes[0] as usize)
            .to_owned()
            .into_shape_with_order((1, 1, HIDDEN_DIM))
            .map_err(|e| TtsError::Synthesis(format!("reshape group0 embed: {e}")))?;

        let first_input = concatenate(Axis(1), &[hidden_state.view(), group0_embed.view()])
            .map_err(|e| TtsError::Synthesis(format!("concat cp input: {e}")))?;

        let mut cp_past_keys =
            Array5::<f32>::zeros((CP_NUM_LAYERS, 1, CP_NUM_KV_HEADS, 0, HEAD_DIM));
        let mut cp_past_values =
            Array5::<f32>::zeros((CP_NUM_LAYERS, 1, CP_NUM_KV_HEADS, 0, HEAD_DIM));
        let mut cp_input = first_input;

        let mut session = self.code_predictor.lock().unwrap();

        for group_idx in 0..NUM_CP_GROUPS {
            let generation_steps = Array1::<i64>::from_elem(1, group_idx as i64);

            let t_input = TensorRef::from_array_view(&cp_input)
                .map_err(|e| TtsError::Synthesis(format!("tensor cp input: {e}")))?;
            let t_steps = TensorRef::from_array_view(&generation_steps)
                .map_err(|e| TtsError::Synthesis(format!("tensor gen steps: {e}")))?;
            let t_keys = TensorRef::from_array_view(&cp_past_keys)
                .map_err(|e| TtsError::Synthesis(format!("tensor cp keys: {e}")))?;
            let t_values = TensorRef::from_array_view(&cp_past_values)
                .map_err(|e| TtsError::Synthesis(format!("tensor cp values: {e}")))?;

            let outputs = session
                .run(ort::inputs![
                    "inputs_embeds" => t_input,
                    "generation_steps" => t_steps,
                    "past_keys" => t_keys,
                    "past_values" => t_values,
                ])
                .map_err(|e| {
                    TtsError::Synthesis(format!("code predictor group {group_idx} failed: {e}"))
                })?;

            let (_, logits_data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| TtsError::Synthesis(format!("extract cp logits: {e}")))?;
            let cp_vocab_size = 2048;
            let last_logits = &logits_data[logits_data.len() - cp_vocab_size..];

            let token = sampler::sample(last_logits, &CP_SAMPLER, &[], sampler::no_mask) as i64;
            codes[group_idx + 1] = token;

            let seq_so_far = if group_idx == 0 { 2 } else { group_idx + 2 };

            let (_, keys_data) = outputs[1]
                .try_extract_tensor::<f32>()
                .map_err(|e| TtsError::Synthesis(format!("extract cp keys: {e}")))?;
            let (_, values_data) = outputs[2]
                .try_extract_tensor::<f32>()
                .map_err(|e| TtsError::Synthesis(format!("extract cp values: {e}")))?;

            cp_past_keys = Array5::from_shape_vec(
                (CP_NUM_LAYERS, 1, CP_NUM_KV_HEADS, seq_so_far, HEAD_DIM),
                keys_data.to_vec(),
            )
            .map_err(|e| TtsError::Synthesis(format!("reshape cp keys: {e}")))?;
            cp_past_values = Array5::from_shape_vec(
                (CP_NUM_LAYERS, 1, CP_NUM_KV_HEADS, seq_so_far, HEAD_DIM),
                values_data.to_vec(),
            )
            .map_err(|e| TtsError::Synthesis(format!("reshape cp values: {e}")))?;

            if group_idx < NUM_CP_GROUPS - 1 {
                let next_embed = self.cp_codec_embeddings[group_idx]
                    .row(token as usize)
                    .to_owned()
                    .into_shape_with_order((1, 1, HIDDEN_DIM))
                    .map_err(|e| TtsError::Synthesis(format!("reshape cp embed: {e}")))?;
                cp_input = next_embed;
            }
        }

        Ok(())
    }

    // ------------------------------------------------------------------
    // Vocoder with reference-code prepend + proportional trim
    // ------------------------------------------------------------------

    fn run_vocoder_clone(
        &self,
        gen_codes: &[[i64; 16]],
        ref_codes: &Array2<i64>, // (T_ref, 16)
        ref_frames: usize,
    ) -> Result<AudioFrame<'static>, TtsError> {
        let gen_frames = gen_codes.len();
        let total_frames = ref_frames + gen_frames;

        // Build (1, 16, total_frames) code tensor: ref_codes | gen_codes
        let mut codes = Array3::<i64>::zeros((1, 16, total_frames));

        // Fill ref codes (stored as T_ref × 16)
        for f in 0..ref_frames {
            for g in 0..16 {
                codes[[0, g, f]] = ref_codes[[f, g]];
            }
        }
        // Fill generated codes
        for (t, frame_codes) in gen_codes.iter().enumerate() {
            for (g, &code) in frame_codes.iter().enumerate() {
                codes[[0, g, ref_frames + t]] = code;
            }
        }

        let t_codes = TensorRef::from_array_view(&codes)
            .map_err(|e| TtsError::Synthesis(format!("tensor codes: {e}")))?;

        let mut session = self.vocoder.lock().unwrap();
        let outputs = session
            .run(ort::inputs!["codes" => t_codes])
            .map_err(|e| TtsError::Synthesis(format!("vocoder failed: {e}")))?;

        let (_, waveform) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::Synthesis(format!("extract waveform: {e}")))?;

        // Trim leading reference portion proportionally
        let cut = ref_frames as f64 / total_frames.max(1) as f64 * waveform.len() as f64;
        let trimmed = waveform[cut as usize..].to_vec();

        Ok(AudioFrame::from_vec(trimmed, SAMPLE_RATE))
    }
}
