use crate::nn::{Embedding, Linear, RmsNorm};
pub struct LlamaCache<T> {
    cos: ndarray::Array2<T>,
    sin: ndarray::Array2<T>,
    kvs: Vec<Option<(ndarray::Array3<T>, ndarray::Array3<T>)>>,
}

pub struct LlamaModel<'a, T> {
    embedding: Embedding<'a, T>,
    blocks: Vec<LlamaBlock<'a, T>>,
    ln_f: RmsNorm<'a, T>,
    lm_head: Linear<'a, T>,
}

pub struct LlamaBlock<'a, T> {
    attn: LlamaAttention<'a, T>,
    mlp: LlamaMlp<'a, T>,
    ln_1: RmsNorm<'a, T>,
    ln_2: RmsNorm<'a, T>,
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

pub struct LlamaMlp<'a, T> {
    c_fc1: Linear<'a, T>,
    c_fc2: Linear<'a, T>,
    c_proj: Linear<'a, T>,
}