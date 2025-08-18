use std::slice;
use ndarray::{s, Array2, Array3, LinalgScalar, Axis, concatenate, ArrayView3, ScalarOperand, Array1};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num_traits::{Float, FromPrimitive, Zero};
use crate::load::Loadable;
use crate::nn::{multinomial, silu, softmax1, softmax3, Embedding, Linear, RmsNorm};


pub struct LlamaCache<T> {
    cos: Array2<T>,
    sin: Array2<T>,
    kvs: Vec<Option<(Array3<T>, Array3<T>)>>,
}

pub struct LlamaConfig<T> {
    num_blocks: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_pos_emb: usize,
    eps: T,
}

pub struct LlamaModel<'a, T> {
    embedding: Embedding<'a, T>,
    blocks: Vec<LlamaBlock<'a, T>>,
    ln_f: RmsNorm<'a, T>,
    lm_head: Linear<'a, T>,
}

impl<T: LinalgScalar + Clone + Float + Zero + FromPrimitive + ScalarOperand> LlamaModel<'_, T> {
    pub fn forward(
        &self,
        x: &[usize],
        index_pos: usize,
        cache: &mut LlamaCache<T>,
    ) -> anyhow::Result<Array1<T>> {
        let seq_len = x.len();

        let mut x = self.embedding.forward(x);

        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }

        let x = self.ln_f.forward(&x)?;
        let x = x.index_axis(Axis(0), seq_len - 1);
        let x = x.to_owned();
        let x = x.insert_axis(Axis(0));

        let x = self.lm_head.forward(&x);


        Ok(x.index_axis(Axis(0), 0).to_owned())
    }
}

impl<'a, T: Loadable> LlamaModel<'a, T> {
    pub fn from_safe_tensors(model: &safetensors::SafeTensors<'a>, prefix: &str, config: &LlamaConfig<T>) -> anyhow::Result<Self> {
        let embedding = Embedding::from_safe_tensors(model, &format!("{}model.embed_tokens.", prefix))?;
        let mut blocks = Vec::with_capacity(config.num_blocks);

        for i in 0..config.num_blocks {
            let block = LlamaBlock::from_safe_tensors(model, &format!("{}model.layers.{i}.", prefix), config)?;
            blocks.push(block);
        }

        let ln_f = RmsNorm::from_safe_tensors(model, &format!("{}model.norm.", prefix), config.eps)?;
        let lm_head = Linear::from_safe_tensors(model, &format!("{}model.output_layernorm.", prefix))?;

        Ok(LlamaModel { embedding, blocks, ln_f, lm_head })

    }
}


impl<T: LinalgScalar + Clone + Float + Zero + FromPrimitive + ScalarOperand + Default + SampleUniform + for<'a> std::ops::AddAssign<&'a T>> LlamaModel<'_, T> {
    pub fn next_token(
        &self,
        x: &[usize],
        is_first: bool,
        cache: &mut LlamaCache<T>,
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

pub struct LlamaBlock<'a, T> {
    attn: LlamaAttention<'a, T>,
    mlp: LlamaMlp<'a, T>,
    ln_1: RmsNorm<'a, T>,
    ln_2: RmsNorm<'a, T>,
}

impl<T: LinalgScalar + Clone + Float + Zero + FromPrimitive + ScalarOperand> LlamaBlock<'_, T> {
    pub fn forward(
        &self,
        x: &Array2<T>,
        index_pos: usize,
        block_idx: usize,
        cache: &mut LlamaCache<T>,
    ) -> anyhow::Result<Array2<T>> {
        let residual = x;

        let x = self.ln_1.forward(x)?;

        let x = self.attn.forward(&x, index_pos, block_idx, cache)? + residual;

        let residual = &x;

        let x = self.ln_2.forward(&x)?;

        let x = self.mlp.forward(&x) + residual;

        Ok(x)
    }
}

impl<'a, T: Loadable> LlamaBlock<'a, T> {
    pub fn from_safe_tensors(model: &safetensors::SafeTensors<'a>, prefix: &str, config: &LlamaConfig<T>) -> anyhow::Result<Self> {
        let attn = LlamaAttention::from_safe_tensors(model, &format!("{}self_attn.", prefix), config)?;
        let mlp = LlamaMlp::from_safe_tensors(model, &format!("{}mlp.", prefix))?;
        let ln_1 = RmsNorm::from_safe_tensors(model, &format!("{}input_layernorm.", prefix), config.eps)?;
        let ln_2 = RmsNorm::from_safe_tensors(model, &format!("{}post_attention_layernorm.", prefix), config.eps)?;

        Ok(LlamaBlock { attn, mlp, ln_1, ln_2 })
    }
}


pub struct LlamaAttention<'a, T> {
    q_proj: Linear<'a, T>,
    k_proj: Linear<'a, T>,
    v_proj: Linear<'a, T>,
    o_proj: Linear<'a, T>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_pos_emb: usize,
}


impl<T: LinalgScalar + Clone + Float + Zero + FromPrimitive + ScalarOperand> LlamaAttention<'_, T> {
    pub fn forward(&self, x: &Array2<T>, index_pos: usize, block_idx: usize, cache: &mut LlamaCache<T>) -> anyhow::Result<Array2<T>> {
        let (seq_len, hidden_size) = x.dim();

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let mut q = q.into_shape_with_order(
            (seq_len, self.num_heads, self.head_dim)
        )?;
        q.swap_axes(0, 1);

        let mut k = k.into_shape_with_order(
            (seq_len, self.num_kv_heads, self.head_dim)
        )?;
        k.swap_axes(0, 1);

        let mut v = v.into_shape_with_order(
            (seq_len, self.num_kv_heads, self.head_dim)
        )?;
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

        let att = Self::casual_dot(q.view(), k.t()) / fac;

        let att = if seq_len == 1 {
            att
        } else {
            tril(att.view())
        };

        let dim = att.ndim() - 1;

        let att = softmax3(att, dim)?;

        let mut y = Self::casual_dot(att.view(), v.view());

        y.swap_axes(0, 1);

        let y = y.into_shape_with_order((seq_len, hidden_size))?;

        let y = self.o_proj.forward(&y);

        Ok(y)
    }
    fn casual_dot(
        q: ArrayView3<T>,
        k: ArrayView3<T>,
    ) -> ndarray::Array3<T> {
        let mut output = Array3::<T>::zeros((q.shape()[0], q.shape()[1], k.shape()[2]));

        for ((q, k), mut o) in q
            .axis_iter(Axis(0))
            .zip(k.axis_iter(Axis(0)))
            .zip(output.axis_iter_mut(Axis(0)))
        {
            let x = q.dot(&k);

            o.assign(&x);
        }

        output
    }

    pub fn apply_rotary_emb(
        x: &Array3<T>,
        index_pos: usize,
        cache: &LlamaCache<T>,
    ) -> Array3<T> {
        let (n_head, seq_len, hidden_size) = x.dim();
        let half = hidden_size / 2;

        let cos = cache.cos.slice(s![index_pos..index_pos + seq_len, ..]);
        let sin = cache.sin.slice(s![index_pos..index_pos + seq_len, ..]);

        let cos_b_axis = cos
            .insert_axis(Axis(0));

        let cos_b = cos_b_axis
            .broadcast((n_head, seq_len, cos.dim().1))
            .expect("broadcast cos failed");


        let sin_b_axis  = sin
            .insert_axis(Axis(0));

        let sin_b = sin_b_axis
            .broadcast((n_head, seq_len, sin.dim().1))
            .expect("broadcast sin failed");

        let x_even = x.slice(s![.., .., 0..; 2]); // shape: (n_head, seq_len, half)
        let x_odd  = x.slice(s![.., .., 1..; 2]); // shape: (n_head, seq_len, half)

        let out_even = &x_even * &cos_b - &x_odd * &sin_b;
        let out_odd  = &x_even * &sin_b + &x_odd * &cos_b;

        let mut y = x.to_owned();
        y.slice_mut(s![.., .., 0..; 2]).assign(&out_even);
        y.slice_mut(s![.., .., 1..; 2]).assign(&out_odd);

        y
    }

    pub fn repeat_kv(&self, x: Array3<T>) -> anyhow::Result<Array3<T>> {
        let n_rep = self.num_heads / self.num_kv_heads;

        Ok(if n_rep == 1 {
            x
        } else {
            let (n_kv_head, seq_len, head_dim) = x.dim();

            concatenate(Axis(2), &vec![x.view(); n_rep])?
                .into_shape_with_order((n_kv_head  * n_rep, seq_len, head_dim))?
        })
    }
}

impl<'a, T: Loadable> LlamaAttention<'a, T> {
    pub fn from_safe_tensors(model: &safetensors::SafeTensors<'a>, prefix: &str, conf: &LlamaConfig<T>) -> anyhow::Result<Self> {
        let q_proj = Linear::from_safe_tensors(model, &format!("{}q_proj.", prefix))?;
        let k_proj = Linear::from_safe_tensors(model, &format!("{}k_proj.", prefix))?;
        let v_proj = Linear::from_safe_tensors(model, &format!("{}v_proj.", prefix))?;
        let o_proj = Linear::from_safe_tensors(model, &format!("{}o_proj.", prefix))?;



        Ok(LlamaAttention {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: conf.num_heads,
            num_kv_heads: conf.num_kv_heads,
            head_dim: conf.head_dim,
            max_pos_emb: conf.max_pos_emb,
        })
    }
}

pub struct LlamaMlp<'a, T> {
    c_fc1: Linear<'a, T>,
    c_fc2: Linear<'a, T>,
    c_proj: Linear<'a, T>,
}


impl<T: LinalgScalar + Clone + Float + Zero + FromPrimitive> LlamaMlp<'_, T> {
    fn forward(&self, x: &ndarray::Array2<T>) -> ndarray::Array2<T> {
        let x = self.c_fc1.forward(x);
        let x = silu(x);
        let x = self.c_fc2.forward(&x);

        let x = self.c_proj.forward(&x);

        x

    }
}

impl<'a, T: Loadable> LlamaMlp<'a, T> {
    pub fn from_safe_tensors(model: &safetensors::SafeTensors<'a>, prefix: &str) -> anyhow::Result<Self> {
        let c_fc1 = Linear::from_safe_tensors(model, &format!("{}gate_proj.", prefix))?;
        let c_fc2 = Linear::from_safe_tensors(model, &format!("{}up_proj.", prefix))?;
        let c_proj = Linear::from_safe_tensors(model, &format!("{}down_proj.", prefix))?;

        Ok(LlamaMlp { c_fc1, c_fc2, c_proj })
    }
}

fn tril<T: LinalgScalar + Float + ScalarOperand>(
    input: ArrayView3<T>,
) -> ndarray::Array3<T> {
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
