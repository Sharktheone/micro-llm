use crate::bf16_wrapper::Bf16Wrapper;
use crate::load::{load_array1, load_array2};
use crate::model::Model;
use crate::nn::{multinomial, silu, softmax1};
use ndarray::{s, Array1, Array2, ArrayView2, Axis, concatenate};
use safetensors::SafeTensors;

fn to_f32_2(v: ArrayView2<Bf16Wrapper>) -> Array2<f32> {
    v.mapv(|x| x.to_f32())
}

fn to_f32_1(v: ndarray::ArrayView1<Bf16Wrapper>) -> Array1<f32> {
    v.mapv(|x| x.to_f32())
}

#[derive(Clone)]
struct LinearWB {
    // weight layout: [out, in]
    w: Array2<f32>,
}
impl LinearWB {
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // x: [seq, in], w: [out, in] => x * w^T => [seq, out]
        x.dot(&self.w.t())
    }
}

#[derive(Clone)]
struct RmsNorm {
    w: Array1<f32>,
    eps: f32,
}
impl RmsNorm {
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let (n, h) = (x.shape()[0], x.shape()[1]);
        let mut inv_denom = Array1::<f32>::zeros(n);
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let ms = row.iter().map(|v| v * v).sum::<f32>() / (h as f32);
            inv_denom[i] = 1.0f32 / (ms + self.eps).sqrt();
        }
        // y = x * inv_denom[row]
        let mut y = x.clone();
        for (i, mut row) in y.axis_iter_mut(Axis(0)).enumerate() {
            let s = inv_denom[i];
            row.mapv_inplace(|v| v * s);
        }
        // scale by weight per feature
        for j in 0..h {
            let wj = self.w[j];
            let mut col = y.column_mut(j);
            col.mapv_inplace(|v| v * wj);
        }
        y
    }
}

#[derive(Clone)]
struct Mlp {
    gate: LinearWB,
    up: LinearWB,
    down: LinearWB,
}
impl Mlp {
    fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let g = self.gate.forward(x);
        let u = self.up.forward(x);
        let g = silu(g);
        let gu = g * u;
        self.down.forward(&gu)
    }
}

#[derive(Default, Clone)]
struct LayerCache {
    // cached after RoPE and head repetition: [t, n_heads*head_dim]
    k: Array2<f32>,
    v: Array2<f32>,
}
impl LayerCache {
    fn len(&self) -> usize { self.k.shape().get(0).copied().unwrap_or(0) }
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn append(&mut self, k_new: &Array2<f32>, v_new: &Array2<f32>) {
        if self.is_empty() {
            self.k = k_new.clone();
            self.v = v_new.clone();
        } else {
            self.k = concatenate![Axis(0), self.k.view(), k_new.view()];
            self.v = concatenate![Axis(0), self.v.view(), v_new.view()];
        }
    }
}

#[derive(Default, Clone)]
pub struct LlamaCache {
    layers: Vec<LayerCache>,
}
impl LlamaCache {
    fn with_layers(n: usize) -> Self { Self { layers: vec![LayerCache::default(); n] } }
}

fn build_inv_freq(head_dim: usize, theta: f32) -> Array1<f32> {
    // size: head_dim/2, value: 1.0 / theta^{i*2/head_dim}
    let half = head_dim / 2;
    let mut v = Array1::<f32>::zeros(half);
    for i in 0..half {
        let p = (2 * i) as f32 / head_dim as f32;
        v[i] = (theta.powf(p)).recip();
    }
    v
}

fn rope_cos_sin(seq_len: usize, head_dim: usize, theta: f32) -> (Array2<f32>, Array2<f32>) {
    // returns [seq, head_dim]
    let inv = build_inv_freq(head_dim, theta);
    let half = head_dim / 2;
    let mut cos = Array2::<f32>::zeros((seq_len, head_dim));
    let mut sin = Array2::<f32>::zeros((seq_len, head_dim));
    for p in 0..seq_len {
        for j in 0..half {
            let t = (p as f32) * inv[j];
            let c = t.cos();
            let s = t.sin();
            cos[(p, j)] = c;
            cos[(p, j + half)] = c;
            sin[(p, j)] = s;
            sin[(p, j + half)] = s;
        }
    }
    (cos, sin)
}

fn rope_cos_sin_for_pos(head_dim: usize, theta: f32, pos: usize) -> (Array1<f32>, Array1<f32>) {
    let inv = build_inv_freq(head_dim, theta);
    let half = head_dim / 2;
    let mut cos = Array1::<f32>::zeros(head_dim);
    let mut sin = Array1::<f32>::zeros(head_dim);
    for j in 0..half {
        let t = (pos as f32) * inv[j];
        let c = t.cos();
        let s = t.sin();
        cos[j] = c;
        cos[j + half] = c;
        sin[j] = s;
        sin[j + half] = s;
    }
    (cos, sin)
}

fn rotate_half_inplace(arr: &mut Array2<f32>) {
    // arr shape [seq, head_dim]
    let head_dim = arr.shape()[1];
    let half = head_dim / 2;
    for mut row in arr.axis_iter_mut(Axis(0)) {
        for i in 0..half {
            let x1 = row[i];
            let x2 = row[i + half];
            row[i] = -x2;
            row[i + half] = x1;
        }
    }
}

fn apply_rope_row(mut row: Array1<f32>, cos: &Array1<f32>, sin: &Array1<f32>) -> Array1<f32> {
    let head_dim = row.len();
    let half = head_dim / 2;
    let mut rot = row.clone();
    for i in 0..half {
        let x1 = row[i];
        let x2 = row[i + half];
        rot[i] = -x2;
        rot[i + half] = x1;
    }
    for i in 0..head_dim {
        row[i] = row[i] * cos[i] + rot[i] * sin[i];
    }
    row
}

#[derive(Clone)]
struct Attention {
    q: LinearWB,
    k: LinearWB,
    v: LinearWB,
    o: LinearWB,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
}

impl Attention {
    fn apply_rope(&self, q: &mut Array2<f32>, k: &mut Array2<f32>, cos: &Array2<f32>, sin: &Array2<f32>) {
        // q,k: [seq, head_dim]
        // q_new = q * cos + rotate_half(q) * sin
        let q_orig = q.clone();
        let k_orig = k.clone();

        let mut q_rot = q.clone();
        rotate_half_inplace(&mut q_rot);
        let mut k_rot = k.clone();
        rotate_half_inplace(&mut k_rot);

        *q = &q_orig * cos + &q_rot * sin;
        *k = &k_orig * cos + &k_rot * sin;
    }

    fn repeat_kv(&self, kv: &Array2<f32>) -> Array2<f32> {
        // kv: [seq, kv_heads*head_dim] -> [seq, n_heads*head_dim] by repeating groups
        if self.n_heads == self.n_kv_heads {
            return kv.clone();
        }
        let groups = self.n_heads / self.n_kv_heads;
        let seq = kv.shape()[0];
        let mut out = Array2::<f32>::zeros((seq, self.n_heads * self.head_dim));
        for h in 0..self.n_kv_heads {
            let src = kv.slice(s![.., h * self.head_dim..(h + 1) * self.head_dim]);
            for g in 0..groups {
                let dst_h = h * groups + g;
                let mut dst = out.slice_mut(s![.., dst_h * self.head_dim..(dst_h + 1) * self.head_dim]);
                dst.assign(&src);
            }
        }
        out
    }

    fn forward(&self, x: &Array2<f32>, past: Option<&LayerCache>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let seq = x.shape()[0];
        let q_all = self.q.forward(x); // [seq, n_heads*head_dim]
        let k_all = self.k.forward(x); // [seq, n_kv_heads*head_dim]
        let v_all = self.v.forward(x); // [seq, n_kv_heads*head_dim]

        // Prepare full k,v per head (after repeat) and apply RoPE
        let mut q_heads: Vec<Array2<f32>> = Vec::with_capacity(self.n_heads);
        let mut k_heads_curr: Vec<Array2<f32>> = Vec::with_capacity(self.n_heads);
        let v_full_curr = self.repeat_kv(&v_all);
        let k_full_curr = self.repeat_kv(&k_all);

        let past_len = past.map(|c| c.len()).unwrap_or(0);

        if seq == 1 {
            let pos = past_len;
            let (cos1, sin1) = rope_cos_sin_for_pos(self.head_dim, self.rope_theta, pos);

            let mut q_rows: Vec<Array1<f32>> = Vec::with_capacity(self.n_heads);
            let mut k_rows: Vec<Array1<f32>> = Vec::with_capacity(self.n_heads);
            for h in 0..self.n_heads {
                let qh1 = q_all.slice(s![0, h * self.head_dim..(h + 1) * self.head_dim]).to_owned();
                let kh1 = k_full_curr
                    .slice(s![0, h * self.head_dim..(h + 1) * self.head_dim])
                    .to_owned();
                q_rows.push(apply_rope_row(qh1, &cos1, &sin1));
                k_rows.push(apply_rope_row(kh1, &cos1, &sin1));
            }

            let mut head_outputs: Vec<Array2<f32>> = Vec::with_capacity(self.n_heads);
            for h in 0..self.n_heads {
                let qh = &q_rows[h]; // [dim]
                let k_curr = &k_rows[h]; // [dim]

                // cached views
                let (k_cache_h, v_cache_h) = if let Some(p) = past {
                    (
                        Some(p.k.slice(s![.., h * self.head_dim..(h + 1) * self.head_dim])),
                        Some(p.v.slice(s![.., h * self.head_dim..(h + 1) * self.head_dim])),
                    )
                } else {
                    (None, None)
                };

                let scale = (self.head_dim as f32).powf(-0.5);
                let t_cache = k_cache_h.as_ref().map(|a| a.shape()[0]).unwrap_or(0);
                let mut logits = Array1::<f32>::zeros(t_cache + 1);
                if let Some(kc) = &k_cache_h {
                    for t in 0..t_cache {
                        let mut acc = 0.0f32;
                        for d in 0..self.head_dim { acc += qh[d] * kc[(t, d)]; }
                        logits[t] = acc * scale;
                    }
                }
                let mut acc = 0.0f32;
                for d in 0..self.head_dim { acc += qh[d] * k_curr[d]; }
                logits[t_cache] = acc * scale;

                let probs = softmax1(logits.view()).unwrap();
                let mut out = Array2::<f32>::zeros((1, self.head_dim));

                if let Some(vc) = &v_cache_h {
                    for d in 0..self.head_dim {
                        let mut s = 0.0f32;
                        for t in 0..t_cache { s += probs[t] * vc[(t, d)]; }
                        out[(0, d)] = s;
                    }
                }

                let v_curr_h = v_full_curr.slice(s![0, h * self.head_dim..(h + 1) * self.head_dim]);
                for d in 0..self.head_dim {
                    out[(0, d)] += probs[t_cache] * v_curr_h[d];
                }
                head_outputs.push(out);
            }

            let mut concat = Array2::<f32>::zeros((1, self.n_heads * self.head_dim));
            for h in 0..self.n_heads {
                let src = &head_outputs[h];
                let mut dst = concat.slice_mut(s![.., h * self.head_dim..(h + 1) * self.head_dim]);
                dst.assign(src);
            }
            let attn_out = self.o.forward(&concat);

            let mut k_full_post = Array2::<f32>::zeros((1, self.n_heads * self.head_dim));
            for h in 0..self.n_heads {
                let mut dst = k_full_post.slice_mut(s![0, h * self.head_dim..(h + 1) * self.head_dim]);
                dst.assign(&k_rows[h]);
            }
            let v_full_post = v_full_curr;

            return (attn_out, k_full_post, v_full_post);
        }

        let (cos_all, sin_all) = rope_cos_sin(seq, self.head_dim, self.rope_theta);
        for h in 0..self.n_heads {
            let qh = q_all.slice(s![.., h * self.head_dim..(h + 1) * self.head_dim]).to_owned();
            let kh = self.repeat_kv(&k_all)
                .slice(s![.., h * self.head_dim..(h + 1) * self.head_dim])
                .to_owned();
            let cos = cos_all.view();
            let sin = sin_all.view();
            let mut qh_applied = qh.clone();
            let mut kh_applied = kh.clone();
            self.apply_rope(&mut qh_applied, &mut kh_applied, &cos.to_owned(), &sin.to_owned());
            q_heads.push(qh_applied);
            k_heads_curr.push(kh_applied);
        }
        let v_full_post = self.repeat_kv(&v_all);

        let mut head_outputs: Vec<Array2<f32>> = Vec::with_capacity(self.n_heads);
        for h in 0..self.n_heads {
            let qh = &q_heads[h]; // [seq, dim]
            let kh = &k_heads_curr[h];
            let vh = v_full_post.slice(s![.., h * self.head_dim..(h + 1) * self.head_dim]);
            let scale = (self.head_dim as f32).powf(-0.5);
            let mut logits = qh.dot(&kh.t());
            logits.mapv_inplace(|v| v * scale);
            for i in 0..seq { for j in (i + 1)..seq { logits[(i, j)] = -1e9; } }
            let mut out = Array2::<f32>::zeros((seq, self.head_dim));
            for i in 0..seq {
                let probs = softmax1(logits.slice(s![i, ..])).unwrap();
                for d in 0..self.head_dim {
                    let mut acc = 0.0f32;
                    for t in 0..seq { acc += probs[t] * vh[(t, d)]; }
                    out[(i, d)] = acc;
                }
            }
            head_outputs.push(out);
        }
        let mut concat = Array2::<f32>::zeros((seq, self.n_heads * self.head_dim));
        for h in 0..self.n_heads {
            let src = &head_outputs[h];
            let mut dst = concat.slice_mut(s![.., h * self.head_dim..(h + 1) * self.head_dim]);
            dst.assign(src);
        }
        let attn_out = self.o.forward(&concat);

        let mut k_full_post = Array2::<f32>::zeros((seq, self.n_heads * self.head_dim));
        for h in 0..self.n_heads {
            let src = &k_heads_curr[h];
            let mut dst = k_full_post.slice_mut(s![.., h * self.head_dim..(h + 1) * self.head_dim]);
            dst.assign(src);
        }

        (attn_out, k_full_post, v_full_post)
    }
}

#[derive(Clone)]
struct DecoderLayer {
    attn: Attention,
    input_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    mlp: Mlp,
}
impl DecoderLayer {
    fn forward(&self, x: &Array2<f32>, cache: Option<&LayerCache>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let x_norm = self.input_norm.forward(x);
        let (attn, k_new, v_new) = self.attn.forward(&x_norm, cache);
        let x1 = x + &attn; // residual add

        let x_norm2 = self.post_attn_norm.forward(&x1);
        let ff = self.mlp.forward(&x_norm2);
        let x_out = &x1 + &ff;
        (x_out, k_new, v_new)
    }
}

pub struct LlamaModel {
    embed: Array2<f32>, // [vocab, hidden]
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Array2<f32>, // [vocab, hidden]
    hidden_size: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
}

impl LlamaModel {
    fn linear(model: &SafeTensors, name: &str) -> anyhow::Result<LinearWB> {
        let w = to_f32_2(load_array2::<Bf16Wrapper>(model, &format!("{}", name))?);
        Ok(LinearWB { w })
    }

    fn rmsnorm(model: &SafeTensors, name: &str, eps: f32) -> anyhow::Result<RmsNorm> {
        let w = to_f32_1(load_array1::<Bf16Wrapper>(model, &format!("{}", name))?);
        Ok(RmsNorm { w, eps })
    }
}

impl<'a> Model<'a> for LlamaModel {
    type Cache = LlamaCache;

    fn from_safetensors(tensors: SafeTensors<'a>) -> anyhow::Result<Self> {
        let embed = to_f32_2(load_array2::<Bf16Wrapper>(&tensors, "model.embed_tokens.weight")?);
        let hidden_size = embed.shape()[1];

        // using layer 0 to infer heads
        let q_w = to_f32_2(load_array2::<Bf16Wrapper>(&tensors, "model.layers.0.self_attn.q_proj.weight")?);
        let k_w = to_f32_2(load_array2::<Bf16Wrapper>(&tensors, "model.layers.0.self_attn.k_proj.weight")?);
        let q_out = q_w.shape()[0];
        let k_out = k_w.shape()[0];

        let mut candidate_head_dims = [32usize, 40, 48, 64, 80, 96, 112, 128, 160, 256];
        candidate_head_dims.sort();

        let head_dim = candidate_head_dims
            .iter()
            .copied()
            .filter(|&d| d != 0 && q_out % d == 0 && k_out % d == 0)
            .find(|&d| d == 64)
            .or_else(|| candidate_head_dims.iter().copied().find(|&d| d != 0 && q_out % d == 0 && k_out % d == 0))
            .unwrap_or_else(|| {
                for d in 1..=512 {
                    if q_out % d == 0 && k_out % d == 0 {
                        return d;
                    }
                }
                if q_out % k_out == 0 { k_out } else { 64 }
            });
        let n_heads = q_out / head_dim;
        let n_kv_heads = k_out / head_dim;
        debug_assert_eq!(n_heads * head_dim, q_out);
        debug_assert_eq!(n_kv_heads * head_dim, k_out);
        debug_assert_eq!(hidden_size % head_dim, 0);

        let rope_theta = 500000.0f32;

        let mut layers = Vec::new();
        let mut idx = 0;
        loop {
            let base = format!("model.layers.{}.", idx);
            let try_name = format!("{}self_attn.q_proj.weight", base);
            if tensors.tensor(&try_name).is_err() {
                break;
            }
            let attn = Attention {
                q: Self::linear(&tensors, &try_name)?,
                k: Self::linear(&tensors, &format!("{}self_attn.k_proj.weight", base))?,
                v: Self::linear(&tensors, &format!("{}self_attn.v_proj.weight", base))?,
                o: Self::linear(&tensors, &format!("{}self_attn.o_proj.weight", base))?,
                n_heads,
                n_kv_heads,
                head_dim,
                rope_theta,
            };
            let input_norm = Self::rmsnorm(&tensors, &format!("{}input_layernorm.weight", base), 1e-6)?;
            let post_attn_norm = Self::rmsnorm(&tensors, &format!("{}post_attention_layernorm.weight", base), 1e-6)?;
            let mlp = Mlp {
                gate: Self::linear(&tensors, &format!("{}mlp.gate_proj.weight", base))?,
                up: Self::linear(&tensors, &format!("{}mlp.up_proj.weight", base))?,
                down: Self::linear(&tensors, &format!("{}mlp.down_proj.weight", base))?,
            };
            layers.push(DecoderLayer {
                attn,
                input_norm,
                post_attn_norm,
                mlp,
            });
            idx += 1;
        }

        let norm = Self::rmsnorm(&tensors, "model.norm.weight", 1e-6)?;
        let lm_head = if tensors.tensor("lm_head.weight").is_ok() {
            to_f32_2(load_array2::<Bf16Wrapper>(&tensors, "lm_head.weight")?)
        } else {
            embed.clone()
        };

        Ok(Self {
            embed,
            layers,
            norm,
            lm_head,
            hidden_size,
            n_heads,
            n_kv_heads,
            head_dim,
            rope_theta,
        })
    }

    fn new_cache(&self) -> Self::Cache { LlamaCache::with_layers(self.layers.len()) }

    fn forward(&self, tokens: &[u32], cache: &mut Self::Cache) -> anyhow::Result<u32> {
        let (mut x, seq_len) = if cache.layers[0].is_empty() {
            let idxs: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();
            (self.embed.select(Axis(0), &idxs).to_owned(), tokens.len())
        } else {
            let last = *tokens.last().unwrap() as usize;
            (self.embed.select(Axis(0), &[last]).to_owned(), 1)
        };

        for (i, layer) in self.layers.iter().enumerate() {
            let past_layer = if cache.layers[i].is_empty() { None } else { Some(&cache.layers[i]) };
            let (x_out, k_new, v_new) = layer.forward(&x, past_layer);
            cache.layers[i].append(&k_new, &v_new);
            x = x_out;
        }

        x = self.norm.forward(&x);
        let last_hidden: Array2<f32> = if seq_len == 1 { x } else { x.slice(s![-1, ..]).to_owned().insert_axis(Axis(0)) };
        let logits: Array2<f32> = last_hidden.dot(&self.lm_head.t());
        let probs = softmax1(logits.slice(s![0, ..]))?;
        let sample = multinomial(probs, 1);
        Ok(sample[0] as u32)
    }
}

