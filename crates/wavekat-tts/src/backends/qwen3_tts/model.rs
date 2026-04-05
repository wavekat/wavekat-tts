use std::path::Path;
use std::sync::Mutex;

use ndarray::{concatenate, s, Array1, Array2, Array3, Array5, Axis};
use ort::session::Session;
use ort::value::TensorRef;
use wavekat_core::AudioFrame;

use crate::TtsError;

use super::sampler::{self, SamplerConfig};
use super::tokenizer::{self, ASSISTANT, IM_START, NEWLINE, TTS_BOS, TTS_EOS, TTS_PAD};

// Codec control token IDs
const CODEC_PAD: i64 = 2148;
const CODEC_BOS: i64 = 2149;
const CODEC_THINK: i64 = 2154;
const CODEC_THINK_BOS: i64 = 2156;
const CODEC_THINK_EOS: i64 = 2157;

/// Talker output: (logits, hidden_state, kv_keys, kv_values).
type TalkerOutput = (Vec<f32>, Array3<f32>, Array5<f32>, Array5<f32>);

// Model dimensions (0.6B-12Hz)
const HIDDEN_DIM: usize = 1024;
const NUM_LAYERS: usize = 28;
const NUM_KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const CP_NUM_LAYERS: usize = 5;
const NUM_CP_GROUPS: usize = 15; // codebook groups 1-15
const SAMPLE_RATE: u32 = 24000;
const CODEC_EOS: i64 = 2150;
const MAX_NEW_TOKENS: usize = 2048;

/// Sampling defaults matching the reference implementation.
const TALKER_SAMPLER: SamplerConfig = SamplerConfig {
    temperature: 0.7,
    top_p: 0.8,
    repetition_penalty: 1.05,
};

const CP_SAMPLER: SamplerConfig = SamplerConfig {
    temperature: 0.2,
    top_p: 0.5,
    repetition_penalty: 1.0,
};

/// All ONNX sessions and embedding tables needed for inference.
///
/// Sessions are wrapped in `Mutex` because `Session::run` requires `&mut self`,
/// while `TtsBackend::synthesize` takes `&self`.
pub struct Model {
    talker_prefill: Mutex<Session>,
    talker_decode: Mutex<Session>,
    code_predictor: Mutex<Session>,
    vocoder: Mutex<Session>,

    // Embedding tables (immutable after construction)
    text_embedding: Array2<f32>,           // (vocab, 2048)
    text_proj_fc1_weight: Array2<f32>,     // (1024, 2048)
    text_proj_fc1_bias: Array1<f32>,       // (1024,)
    text_proj_fc2_weight: Array2<f32>,     // (1024, 1024)
    text_proj_fc2_bias: Array1<f32>,       // (1024,)
    talker_codec_embedding: Array2<f32>,   // (3072, 1024)
    cp_codec_embeddings: Vec<Array2<f32>>, // 15 × (2048, 1024)

    // Precomputed
    tts_pad_embed: Array1<f32>, // (1024,) projected tts_pad text embedding
}

impl Model {
    /// Load all ONNX sessions and embedding tables from `model_dir`.
    pub fn load(model_dir: &Path) -> Result<Self, TtsError> {
        let load_session = |name: &str| -> Result<Session, TtsError> {
            let path = model_dir.join(name);
            Session::builder()
                .map_err(|e| TtsError::Model(format!("session builder error: {e}")))?
                .commit_from_file(&path)
                .map_err(|e| TtsError::Model(format!("failed to load {name}: {e}")))
        };

        let talker_prefill = load_session("talker_prefill.onnx")?;
        let talker_decode = load_session("talker_decode.onnx")?;
        let code_predictor = load_session("code_predictor.onnx")?;
        let vocoder = load_session("vocoder.onnx")?;

        let text_embedding = load_npy2(model_dir, "text_embedding.npy")?;
        let text_proj_fc1_weight = load_npy2(model_dir, "text_projection_fc1_weight.npy")?;
        let text_proj_fc1_bias = load_npy1(model_dir, "text_projection_fc1_bias.npy")?;
        let text_proj_fc2_weight = load_npy2(model_dir, "text_projection_fc2_weight.npy")?;
        let text_proj_fc2_bias = load_npy1(model_dir, "text_projection_fc2_bias.npy")?;
        let talker_codec_embedding = load_npy2(model_dir, "talker_codec_embedding.npy")?;

        let mut cp_codec_embeddings = Vec::with_capacity(NUM_CP_GROUPS);
        for i in 0..NUM_CP_GROUPS {
            cp_codec_embeddings.push(load_npy2(
                model_dir,
                &format!("cp_codec_embedding_{i}.npy"),
            )?);
        }

        let tts_pad_raw = text_embedding.row(TTS_PAD as usize).to_owned();
        let tts_pad_embed = text_project(
            &tts_pad_raw,
            &text_proj_fc1_weight,
            &text_proj_fc1_bias,
            &text_proj_fc2_weight,
            &text_proj_fc2_bias,
        );

        Ok(Self {
            talker_prefill: Mutex::new(talker_prefill),
            talker_decode: Mutex::new(talker_decode),
            code_predictor: Mutex::new(code_predictor),
            vocoder: Mutex::new(vocoder),
            text_embedding,
            text_proj_fc1_weight,
            text_proj_fc1_bias,
            text_proj_fc2_weight,
            text_proj_fc2_bias,
            talker_codec_embedding,
            cp_codec_embeddings,
            tts_pad_embed,
        })
    }

    /// Run the full synthesis pipeline: prefill → decode → code predict → vocoder.
    pub fn synthesize(
        &self,
        text_tokens: &[u32],
        language: &str,
    ) -> Result<AudioFrame<'static>, TtsError> {
        let lang_id = tokenizer::language_id(language)
            .ok_or_else(|| TtsError::UnsupportedLanguage(language.to_string()))?;

        let (prefill_embeds, trailing) = self.build_prefill_embeds(text_tokens, lang_id)?;
        let prefill_len = prefill_embeds.shape()[1];

        // Run talker prefill
        let (logits, hidden_states, mut past_keys, mut past_values) =
            self.run_talker_prefill(&prefill_embeds, prefill_len)?;

        // Decode loop
        let mut all_codes: Vec<[i64; 16]> = Vec::new();
        let mut talker_past_tokens: Vec<i64> = Vec::new();
        let mut current_logits = logits;

        // Hidden state for code predictor: starts from prefill last position
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
                sampler::talker_mask,
            ) as i64;

            if group0 == CODEC_EOS {
                break;
            }
            talker_past_tokens.push(group0);

            // Run code predictor for groups 1-15
            let mut codes = [0i64; 16];
            codes[0] = group0;
            self.run_code_predictor(&current_hidden, &mut codes)?;
            all_codes.push(codes);

            // Build next talker input: sum of all 16 group embeddings + trailing text
            let mut next_embed = self.talker_codec_embedding.row(group0 as usize).to_owned();
            for g in 0..NUM_CP_GROUPS {
                let cp_embed = self.cp_codec_embeddings[g].row(codes[g + 1] as usize);
                next_embed += &cp_embed;
            }
            if step < trailing.len() {
                next_embed += &trailing[step];
            } else {
                next_embed += &self.tts_pad_embed;
            }

            let next_embed = next_embed
                .into_shape_with_order((1, 1, HIDDEN_DIM))
                .map_err(|e| TtsError::Synthesis(format!("reshape next_embed: {e}")))?;

            // Run talker decode
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

        self.run_vocoder(&all_codes)
    }

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

    /// Build prefill embeddings and trailing text hidden states.
    ///
    /// Prefill layout (matching the C# reference):
    /// ```text
    /// [im_start, assistant, \n]         — role prefix (text proj only, no codec)
    /// [think, think_bos, lang, think_eos, speaker(=pad)] — tts_pad_embed + codec_embed
    /// [tts_bos_embed + codec_embed(pad)]  — transition marker
    /// [text_proj(first_text) + codec_embed(bos)] — first text token enters here
    /// ```
    ///
    /// Trailing text hidden: `[text_proj(tok) for tok in text[1:]] + [text_proj(TTS_EOS)]`
    /// These are consumed one per decode step; after exhaustion, tts_pad_embed is used.
    fn build_prefill_embeds(
        &self,
        text_tokens: &[u32],
        lang_id: i64,
    ) -> Result<(Array3<f32>, Vec<Array1<f32>>), TtsError> {
        let tts_bos_embed = self.text_project_token(TTS_BOS);

        let first_text = if text_tokens.is_empty() {
            TTS_PAD
        } else {
            text_tokens[0]
        };

        // Codec prefix: [think, think_bos, lang_id, think_eos, speaker_id(=pad)]
        let codec_prefix = [
            CODEC_THINK,
            CODEC_THINK_BOS,
            lang_id,
            CODEC_THINK_EOS,
            CODEC_PAD,
        ];

        // 3 role + 5 codec prefix + 1 transition + 1 first text = 10
        let seq_len = 3 + codec_prefix.len() + 2;
        let mut embeds = Array3::<f32>::zeros((1, seq_len, HIDDEN_DIM));
        let mut pos = 0;

        // 1. Role prefix: text_project only, no codec component
        for &tok in &[IM_START, ASSISTANT, NEWLINE] {
            let embed = self.text_project_token(tok);
            embeds.slice_mut(s![0, pos, ..]).assign(&embed);
            pos += 1;
        }

        // 2. Codec prefix: tts_pad_embed + codec_embed(token)
        for &codec_tok in &codec_prefix {
            let mut embed = self.tts_pad_embed.clone();
            embed += &self.talker_codec_embedding.row(codec_tok as usize);
            embeds.slice_mut(s![0, pos, ..]).assign(&embed);
            pos += 1;
        }

        // 3. Transition: tts_bos_embed + codec_embed(pad)
        {
            let mut embed = tts_bos_embed;
            embed += &self.talker_codec_embedding.row(CODEC_PAD as usize);
            embeds.slice_mut(s![0, pos, ..]).assign(&embed);
            pos += 1;
        }

        // 4. First text token + codec_embed(bos)
        {
            let mut embed = self.text_project_token(first_text);
            embed += &self.talker_codec_embedding.row(CODEC_BOS as usize);
            embeds.slice_mut(s![0, pos, ..]).assign(&embed);
        }

        // Trailing text hidden: remaining text tokens + TTS_EOS
        let mut trailing = Vec::new();
        if text_tokens.len() > 1 {
            for &tok in &text_tokens[1..] {
                trailing.push(self.text_project_token(tok));
            }
        }
        trailing.push(self.text_project_token(TTS_EOS));

        Ok((embeds, trailing))
    }

    /// Run talker_prefill.onnx.
    ///
    /// Returns (logits, hidden_states, past_keys, past_values).
    fn run_talker_prefill(
        &self,
        inputs_embeds: &Array3<f32>,
        seq_len: usize,
    ) -> Result<TalkerOutput, TtsError> {
        let attention_mask = Array2::<i64>::ones((1, seq_len));

        // M-RoPE position IDs: (3, 1, T) — all axes identical for TTS
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

        // Logits: last position (1, 1, 3072) → flat Vec
        let (_, logits_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::Synthesis(format!("extract logits: {e}")))?;
        let logits: Vec<f32> = logits_data.to_vec();

        // Hidden states: (1, T, 1024)
        let (_, hidden_data) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::Synthesis(format!("extract hidden: {e}")))?;
        let hidden = Array3::from_shape_vec((1, seq_len, HIDDEN_DIM), hidden_data.to_vec())
            .map_err(|e| TtsError::Synthesis(format!("reshape hidden: {e}")))?;

        // Stack per-layer KV caches: present_key_0, present_value_0, ...
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

            // (1, 8, T, 128) → insert axis 0 → (1, 1, 8, T, 128)
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

    /// Run talker_decode.onnx for a single step.
    fn run_talker_decode(
        &self,
        inputs_embeds: &Array3<f32>, // (1, 1, 1024)
        total_seq: usize,
        position: i64,
        past_keys: &Array5<f32>,   // (28, 1, 8, past_seq, 128)
        past_values: &Array5<f32>, // (28, 1, 8, past_seq, 128)
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

    /// Run the code predictor to fill codebook groups 1-15.
    fn run_code_predictor(
        &self,
        hidden_state: &Array3<f32>, // (1, 1, 1024)
        codes: &mut [i64; 16],
    ) -> Result<(), TtsError> {
        let group0_embed = self
            .talker_codec_embedding
            .row(codes[0] as usize)
            .to_owned()
            .into_shape_with_order((1, 1, HIDDEN_DIM))
            .map_err(|e| TtsError::Synthesis(format!("reshape group0 embed: {e}")))?;

        // First call: concat(hidden_state, group0_embed) → (1, 2, 1024)
        let first_input = concatenate(Axis(1), &[hidden_state.view(), group0_embed.view()])
            .map_err(|e| TtsError::Synthesis(format!("concat cp input: {e}")))?;

        // Empty KV cache: (5, 1, 8, 0, 128)
        let mut cp_past_keys = Array5::<f32>::zeros((CP_NUM_LAYERS, 1, NUM_KV_HEADS, 0, HEAD_DIM));
        let mut cp_past_values =
            Array5::<f32>::zeros((CP_NUM_LAYERS, 1, NUM_KV_HEADS, 0, HEAD_DIM));
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

            // Sample from last position logits
            let (_, logits_data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| TtsError::Synthesis(format!("extract cp logits: {e}")))?;
            let vocab_size = 2048;
            let last_logits = &logits_data[logits_data.len() - vocab_size..];

            let token = sampler::sample(last_logits, &CP_SAMPLER, &[], sampler::no_mask) as i64;
            codes[group_idx + 1] = token;

            // Update KV cache
            let seq_so_far = if group_idx == 0 { 2 } else { group_idx + 2 };

            let (_, keys_data) = outputs[1]
                .try_extract_tensor::<f32>()
                .map_err(|e| TtsError::Synthesis(format!("extract cp keys: {e}")))?;
            let (_, values_data) = outputs[2]
                .try_extract_tensor::<f32>()
                .map_err(|e| TtsError::Synthesis(format!("extract cp values: {e}")))?;

            cp_past_keys = Array5::from_shape_vec(
                (CP_NUM_LAYERS, 1, NUM_KV_HEADS, seq_so_far, HEAD_DIM),
                keys_data.to_vec(),
            )
            .map_err(|e| TtsError::Synthesis(format!("reshape cp keys: {e}")))?;
            cp_past_values = Array5::from_shape_vec(
                (CP_NUM_LAYERS, 1, NUM_KV_HEADS, seq_so_far, HEAD_DIM),
                values_data.to_vec(),
            )
            .map_err(|e| TtsError::Synthesis(format!("reshape cp values: {e}")))?;

            // Prepare next input (if not the last group)
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

    /// Run vocoder on the collected code matrix → AudioFrame.
    fn run_vocoder(&self, all_codes: &[[i64; 16]]) -> Result<AudioFrame<'static>, TtsError> {
        let num_steps = all_codes.len();

        // Build (1, 16, T) i64 tensor
        let mut codes = Array3::<i64>::zeros((1, 16, num_steps));
        for (t, frame_codes) in all_codes.iter().enumerate() {
            for (g, &code) in frame_codes.iter().enumerate() {
                codes[[0, g, t]] = code;
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

        // Trim leading silence produced by the think phase.
        let start = waveform.iter().position(|&s| s.abs() > 0.01).unwrap_or(0);
        let trimmed = &waveform[start..];

        Ok(AudioFrame::new(trimmed, SAMPLE_RATE).into_owned())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// SiLU-gated MLP text projection: 2048 → 1024.
fn text_project(
    input: &Array1<f32>,
    fc1_weight: &Array2<f32>,
    fc1_bias: &Array1<f32>,
    fc2_weight: &Array2<f32>,
    fc2_bias: &Array1<f32>,
) -> Array1<f32> {
    let hidden = fc1_weight.dot(input) + fc1_bias;
    let activated = hidden.mapv(|x| x * (1.0 / (1.0 + (-x).exp())));
    fc2_weight.dot(&activated) + fc2_bias
}

/// Load a 2D .npy file into Array2<f32>.
fn load_npy2(dir: &Path, name: &str) -> Result<Array2<f32>, TtsError> {
    let path = dir.join(name);
    let bytes = std::fs::read(&path)
        .map_err(|e| TtsError::Model(format!("failed to read {}: {e}", path.display())))?;
    let reader = npyz::NpyFile::new(&bytes[..])
        .map_err(|e| TtsError::Model(format!("failed to parse {name}: {e}")))?;
    let shape = reader.shape().to_vec();
    let data: Vec<f32> = reader
        .into_vec()
        .map_err(|e| TtsError::Model(format!("failed to read data from {name}: {e}")))?;
    Array2::from_shape_vec((shape[0] as usize, shape[1] as usize), data)
        .map_err(|e| TtsError::Model(format!("shape mismatch in {name}: {e}")))
}

/// Load a 1D .npy file into Array1<f32>.
fn load_npy1(dir: &Path, name: &str) -> Result<Array1<f32>, TtsError> {
    let path = dir.join(name);
    let bytes = std::fs::read(&path)
        .map_err(|e| TtsError::Model(format!("failed to read {}: {e}", path.display())))?;
    let reader = npyz::NpyFile::new(&bytes[..])
        .map_err(|e| TtsError::Model(format!("failed to parse {name}: {e}")))?;
    let data: Vec<f32> = reader
        .into_vec()
        .map_err(|e| TtsError::Model(format!("failed to read data from {name}: {e}")))?;
    Ok(Array1::from(data))
}
