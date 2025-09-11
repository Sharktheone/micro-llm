use crate::load::Loadable;
use crate::nn::ndarray::multinomial::multinomial;
use crate::nn::ndarray::silu::silu;
use crate::nn::{Embedding, LinearNoBias, RmsNorm, softmax1};
use micro_backend::{Backend, DType, SupportsDType, Tensor1, Tensor2, Tensor3};
use micro_backend::{RefTensor2, RefTensor3, Tensor};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num_traits::{AsPrimitive, Float, FloatConst, FromPrimitive, One, Zero};
use std::f32::consts::PI;
use std::fmt::{Debug, Display};
use std::slice;
use crate::nn::ndarray::softmax::softmax3;

pub struct LlamaCache<'a, B: Backend + SupportsDType<T>, T: DType> {
    cos: Tensor2<'a, B, T>,
    sin: Tensor2<'a, B, T>,
    kvs: Vec<Option<(Tensor3<'a, B, T>, Tensor3<'a, B, T>)>>,
}

impl<'a, B: Backend + SupportsDType<T>, T: DType> LlamaCache<'a, B, T> {
    pub fn new(config: &LlamaConfig) -> Self {
        let kvs = vec![None; config.num_blocks];

        let rope_scaling = &config.rope_scaling;

        let orig_max_pos_emb = T::from_usize(rope_scaling.original_max_pos_emb).unwrap();

        let low_freq_wavelen =
            rope_scaling.original_max_pos_emb as f32 / rope_scaling.low_freq_factor;
        let high_freq_wavelen =
            rope_scaling.original_max_pos_emb as f32 / rope_scaling.high_freq_factor;

        let theta = calculate_default_inv_freq(config)
            .into_iter()
            .map(|freq| {
                let wavelen = 2. * PI / freq;
                if wavelen < high_freq_wavelen {
                    freq
                } else if wavelen > low_freq_wavelen {
                    freq / rope_scaling.factor
                } else {
                    let smooth = (rope_scaling.original_max_pos_emb as f32 / wavelen
                        - rope_scaling.low_freq_factor)
                        / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                    (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                }
            })
            .map(|freq| T::from_f32(freq).unwrap())
            .collect::<Vec<_>>();

        let len = theta.len();

        let theta = Tensor1::from_vec(theta, len);

        let mut idx_theta = Tensor1::from_iter(0..config.max_pos_emb)
            .mapv(|i| T::from_usize(i).unwrap())
            .into_shape_with_order((config.max_pos_emb, 1))
            .unwrap();

        let theta_view = theta.to_shape((1, theta.len())).unwrap();

        let idx_theta = idx_theta.dot(&theta_view);

        let cos = idx_theta.cos();
        let sin = idx_theta.sin();

        LlamaCache { cos, sin, kvs }
    }

    pub fn reset(&mut self) {
        for kv in &mut self.kvs {
            *kv = None;
        }
    }
}

fn calculate_default_inv_freq(config: &LlamaConfig) -> Vec<f32> {
    let head_dim = config.hidden_size / config.num_heads;

    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / config.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

pub struct LlamaConfig {
    num_blocks: usize,
    num_heads: usize,
    num_kv_heads: usize,
    hidden_size: usize,
    max_pos_emb: usize,
    eps: f32,
    rope_theta: f32,
    rope_scaling: RopeScaling,
}

pub struct RopeScaling {
    factor: f32,
    high_freq_factor: f32,
    low_freq_factor: f32,
    original_max_pos_emb: usize,
}

pub struct LlamaModel<'a, B: Backend + SupportsDType<T>, T: DType> {
    embedding: Embedding<'a, B, T>,
    blocks: Vec<LlamaBlock<'a, B, T>>,
    ln_f: RmsNorm<'a, B, T>,
    lm_head: LinearNoBias<'a, B, T>,
}

impl<'a, B: Backend + SupportsDType<T>, T: DType> LlamaModel<'a, B, T> {
    pub fn forward(
        &self,
        x: &[usize],
        index_pos: usize,
        cache: &mut LlamaCache<B, T>,
    ) -> anyhow::Result<Tensor1<B, T>> {
        let seq_len = x.len();

        let mut x = self.embedding.forward(x);

        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }

        let x = self.ln_f.forward(&x)?;
        let x = x.index_axis(0, seq_len - 1);
        let x = x.to_owned();
        let x = x.insert_axis(0);

        let x = self.lm_head.forward(&x);

        Ok(x.index_axis(0, 0).to_owned())
    }
}

impl<'a, B: Backend + SupportsDType<T>, T: DType> LlamaModel<'a, B, T> {
    pub fn from_safe_tensors(
        model: &safetensors::SafeTensors<'a>,
        prefix: &str,
        config: &LlamaConfig,
    ) -> anyhow::Result<Self> {
        let embedding =
            Embedding::from_safe_tensors(model, &format!("{}model.embed_tokens.", prefix))?;
        let mut blocks = Vec::with_capacity(config.num_blocks);

        for i in 0..config.num_blocks {
            let block = LlamaBlock::from_safe_tensors(
                model,
                &format!("{}model.layers.{i}.", prefix),
                config,
            )?;
            blocks.push(block);
        }

        let ln_f =
            RmsNorm::from_safe_tensors(model, &format!("{}model.norm.", prefix), config.eps)?;
        let lm_head =
            LinearNoBias::from_safe_tensors(model, &format!("{}model.embed_tokens.", prefix))?;

        Ok(LlamaModel {
            embedding,
            blocks,
            ln_f,
            lm_head,
        })
    }
}

impl<'a, B: Backend + SupportsDType<T>, T: DType> LlamaModel<'_, B, T> {
    pub fn next_token(
        &self,
        x: &[usize],
        is_first: bool,
        cache: &mut LlamaCache<'_, B, T>,
    ) -> anyhow::Result<usize> {
        let (ctx, index_pos) = if is_first {
            (x, 0)
        } else {
            let last = x.last().unwrap();

            (slice::from_ref(last), x.len() - 1)
        };

        let logits = self.forward(ctx, index_pos, cache)?;

        let probs = softmax1(logits.view())?;

        let token = multinomial(probs, 1)[0];

        Ok(token)
    }
}

pub struct LlamaBlock<'a, B: Backend + SupportsDType<T>, T: DType> {
    attn: LlamaAttention<'a, B, T>,
    mlp: LlamaMlp<'a, B, T>,
    ln_1: RmsNorm<'a, B, T>,
    ln_2: RmsNorm<'a, B, T>,
}

impl<'a, B: Backend + SupportsDType<T>, T: DType> LlamaBlock<'a, B, T> {
    pub fn forward(
        &self,
        x: &RefTensor2<B, T>,
        index_pos: usize,
        block_idx: usize,
        cache: &mut LlamaCache<'a, B, T>,
    ) -> anyhow::Result<Tensor2<B, T>> {
        let residual = x;

        let x = self.ln_1.forward(x)?;

        let x = self.attn.forward(&x, index_pos, block_idx, cache)? + residual;

        let residual = &x;

        let x = self.ln_2.forward(&x)?;

        let x = self.mlp.forward(&x) + residual;

        Ok(x)
    }
}

impl<'a, B: Backend + SupportsDType<T>, T: DType> LlamaBlock<'a, B, T> {
    pub fn from_safe_tensors(
        model: &safetensors::SafeTensors<'a>,
        prefix: &str,
        config: &LlamaConfig,
    ) -> anyhow::Result<Self> {
        let attn =
            LlamaAttention::from_safe_tensors(model, &format!("{}self_attn.", prefix), config)?;
        let mlp = LlamaMlp::from_safe_tensors(model, &format!("{}mlp.", prefix))?;
        let ln_1 =
            RmsNorm::from_safe_tensors(model, &format!("{}input_layernorm.", prefix), config.eps)?;
        let ln_2 = RmsNorm::from_safe_tensors(
            model,
            &format!("{}post_attention_layernorm.", prefix),
            config.eps,
        )?;

        Ok(LlamaBlock {
            attn,
            mlp,
            ln_1,
            ln_2,
        })
    }
}

pub struct LlamaAttention<'a, B: Backend + SupportsDType<T>, T: DType> {
    q_proj: LinearNoBias<'a, B, T>,
    k_proj: LinearNoBias<'a, B, T>,
    v_proj: LinearNoBias<'a, B, T>,
    o_proj: LinearNoBias<'a, B, T>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_pos_emb: usize,
}

impl<'a, B: Backend + SupportsDType<T>, T: DType> LlamaAttention<'_, B, T> {
    pub fn forward(
        &self,
        x: &RefTensor2<B, T>,
        index_pos: usize,
        block_idx: usize,
        cache: &mut LlamaCache<T>,
    ) -> anyhow::Result<Tensor2<B, T>> {
        let (seq_len, hidden_size) = x.dim();

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let mut q = q.into_shape_with_order((seq_len, self.num_heads, self.head_dim))?;
        q.swap_axes(0, 1);

        let mut k = k.into_shape_with_order((seq_len, self.num_kv_heads, self.head_dim))?;
        k.swap_axes(0, 1);

        let mut v = v.into_shape_with_order((seq_len, self.num_kv_heads, self.head_dim))?;
        v.swap_axes(0, 1);

        let q = Self::apply_rotary_emb(&q, index_pos, cache);
        let mut k = Self::apply_rotary_emb(&k, index_pos, cache);

        if let Some((cached_k, cached_v)) = &cache.kvs[block_idx] {
            k = concatenate(Axis(1), &[cached_k.view(), k.view()])?;
            v = concatenate(Axis(1), &[cached_v.view(), v.view()])?;

            let k_seq_len = k.dim().0;
            if k_seq_len > self.max_pos_emb {
                todo!()
            }

            let v_seq_len = v.dim().0;
            if v_seq_len > self.max_pos_emb {
                todo!()
            }

            cache.kvs[block_idx] = Some((k.clone(), v.clone()));
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let fac = T::from_usize(self.head_dim).unwrap().sqrt();

        let mut k_trans = k.view();
        k_trans.swap_axes(1, 2);

        let att = Self::casual_dot(q.view(), k_trans) / fac;

        let att = if seq_len == 1 { att } else { tril(att.view()) };

        let dim = att.ndim() - 1;

        let att = softmax3(att, dim)?;

        let mut y = Self::casual_dot(att.view(), v.view());

        y.swap_axes(0, 1);

        let y = y.into_shape_clone((seq_len, hidden_size))?;

        let y = self.o_proj.forward(&y);

        Ok(y)
    }
    // fn casual_dot(q: ArrayView3<T>, k: ArrayView3<T>) -> ndarray::Array3<T> {
    //     let mut output = Array3::<T>::zeros((q.shape()[0], q.shape()[1], k.shape()[2]));
    //
    //     for ((q, k), mut o) in q
    //         .axis_iter(Axis(0))
    //         .zip(k.axis_iter(Axis(0)))
    //         .zip(output.axis_iter_mut(Axis(0)))
    //     {
    //         let x = q.dot(&k);
    //
    //         o.assign(&x);
    //     }
    //
    //     output
    // }

    pub fn apply_rotary_emb(x: &RefTensor3<B, T>, index_pos: usize, cache: &LlamaCache<B, T>) -> Tensor3<'_, B, T> {
        let (n_head, seq_len, hidden_size) = x.dim();
        let half = hidden_size / 2;

        let cos = cache.cos.slice(s![index_pos..index_pos + seq_len, ..]);
        let sin = cache.sin.slice(s![index_pos..index_pos + seq_len, ..]);

        let x = x.as_standard_layout();
        let cos = cos.as_standard_layout();
        let sin = sin.as_standard_layout();

        let x_view = x.as_slice().unwrap();
        let cos = cos.as_slice().unwrap();
        let sin = sin.as_slice().unwrap();

        let mut output = vec![T::zero(); x_view.len()];

        let s = x.shape();
        let h = s[0];
        let t = s[1];
        let d = s[2];

        x_view
            .chunks(t * d)
            .zip(output.chunks_mut(t * d))
            .enumerate()
            .for_each(|(bh_i, (src, dst))| {
                for i_t in 0..t {
                    for i_d in 0..d / 2 {
                        let i1 = i_t * d + i_d;
                        let i2 = i1 + d / 2;
                        let i_cs = i_t * (d / 2) + i_d;

                        dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                        dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                    }
                }
            });

        Tensor3::from_shape_vec((n_head, seq_len, hidden_size), output).unwrap()
    }

    pub fn repeat_kv(&self, x: Tensor3<B, T>) -> anyhow::Result<Tensor3<B, T>> {
        let n_rep = self.num_heads / self.num_kv_heads;

        Ok(if n_rep == 1 {
            x
        } else {
            let (n_kv_head, seq_len, head_dim) = x.dim();

            let a = concatenate(Axis(0), &vec![x.view(); n_rep])?;

            a.into_shape_with_order((n_kv_head * n_rep, seq_len, head_dim))?
        })
    }
}

impl<'a, B: Backend + SupportsDType<T>, T: DType> LlamaAttention<'a, B, T> {
    pub fn from_safe_tensors(
        model: &safetensors::SafeTensors<'a>,
        prefix: &str,
        conf: &LlamaConfig,
    ) -> anyhow::Result<Self> {
        let q_proj = LinearNoBias::from_safe_tensors(model, &format!("{}q_proj.", prefix))?;
        let k_proj = LinearNoBias::from_safe_tensors(model, &format!("{}k_proj.", prefix))?;
        let v_proj = LinearNoBias::from_safe_tensors(model, &format!("{}v_proj.", prefix))?;
        let o_proj = LinearNoBias::from_safe_tensors(model, &format!("{}o_proj.", prefix))?;

        let head_dim = conf.hidden_size / conf.num_heads;

        Ok(LlamaAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: conf.num_heads,
            num_kv_heads: conf.num_kv_heads,
            head_dim,
            max_pos_emb: conf.max_pos_emb,
        })
    }
}

pub struct LlamaMlp<'a, B: Backend + SupportsDType<T>, T: DType> {
    c_fc1: LinearNoBias<'a, B, T>,
    c_fc2: LinearNoBias<'a, B, T>,
    c_proj: LinearNoBias<'a, B, T>,
}

impl<'a, B: Backend + SupportsDType<T>, T: DType> LlamaMlp<'a, B, T> {
    fn forward(&self, input: &ndarray::Array2<T>) -> ndarray::Array2<T> {
        let residual = self.c_fc1.forward(input);
        let residual = silu(residual);

        let x = self.c_fc2.forward(&input);

        let x = residual * x;

        let x = self.c_proj.forward(&x);

        x
    }
}

impl<'a, B: Backend + SupportsDType<T>, T: DType> LlamaMlp<'a, B, T> {
    pub fn from_safe_tensors(
        model: &safetensors::SafeTensors<'a>,
        prefix: &str,
    ) -> anyhow::Result<Self> {
        let c_fc1 = LinearNoBias::from_safe_tensors(model, &format!("{}gate_proj.", prefix))?;
        let c_fc2 = LinearNoBias::from_safe_tensors(model, &format!("{}up_proj.", prefix))?;
        let c_proj = LinearNoBias::from_safe_tensors(model, &format!("{}down_proj.", prefix))?;

        Ok(LlamaMlp {
            c_fc1,
            c_fc2,
            c_proj,
        })
    }
}

fn tril<'a, B: Backend + SupportsDType<T>, T: DType>(
    input: &RefTensor3<B, T>,
) -> Tensor3<'a, B, T> {
    let mut output = ndarray::Array3::<T>::zeros(input.raw_dim());

    for ((i, j), (_, o)) in input.indexed_iter().zip(output.indexed_iter_mut()) {
        if i.1 < i.2 {
            *o = T::neg_infinity();
        } else {
            *o = *j;
        }
    }

    output
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::bf16_wrapper::Bf16Wrapper;
    use half::bf16;

    const MODEL_LOCATION: &str =
        "/run/media/shark/datagrave/ai-modles/huggingface/llama-3.2-1B/model.safetensors";

    const TOKENIZER_LOCATION: &str =
        "/run/media/shark/datagrave/ai-modles/huggingface/llama-3.2-1B/tokenizer.json";

    const BOS_TOKEN: u32 = 128000;
    const EOS_TOKEN: u32 = 128001;

    #[test]
    fn test() {
        let bytes = std::fs::read(MODEL_LOCATION).unwrap();

        let model = safetensors::SafeTensors::deserialize(&bytes).unwrap();

        let config = LlamaConfig {
            num_blocks: 16,
            num_heads: 32,
            num_kv_heads: 8,
            max_pos_emb: 131027,
            eps: 1e-5,
            hidden_size: 2048,
            rope_theta: 500000.,
            rope_scaling: RopeScaling {
                factor: 32.,
                high_freq_factor: 4.,
                low_freq_factor: 1.,
                original_max_pos_emb: 8192,
            },
        };

        let model = LlamaModel::<Bf16Wrapper>::from_safe_tensors(&model, "", &config).unwrap();

        let mut cache = LlamaCache::new(&config);

        let tokenizer = micro_tokenizer::hf::load_hf_tokenizer(TOKENIZER_LOCATION).unwrap();

        let mut input = tokenizer.encode("My name is Luna");

        input.insert(0, BOS_TOKEN);

        let mut input = input.iter().map(|x| *x as usize).collect::<Vec<usize>>();

        for _ in 0..10 {
            let token = model.next_token(&input, true, &mut cache).unwrap();

            if token == EOS_TOKEN as usize {
                break;
            }

            if let Some(decoded) = tokenizer.decode_token(token as u32) {
                print!("{decoded}");
            }

            input.push(token);
        }
    }
}
