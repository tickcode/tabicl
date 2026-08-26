#include "model_forward.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "ggml.h"

namespace tabicl {

namespace {

// ---------------------------------------------------------------------------
// Weight lookup helpers
// ---------------------------------------------------------------------------

struct MabWeights {
  ggml_tensor* in_proj_w = nullptr;   // ggml (E_in, 3E)
  ggml_tensor* in_proj_b = nullptr;   // (3E)
  ggml_tensor* out_proj_w = nullptr;  // (E, E)
  ggml_tensor* out_proj_b = nullptr;  // (E)
  ggml_tensor* linear1_w = nullptr;   // (E, ff)
  ggml_tensor* linear1_b = nullptr;
  ggml_tensor* linear2_w = nullptr;   // (ff, E)
  ggml_tensor* linear2_b = nullptr;
  ggml_tensor* norm1_w = nullptr;
  ggml_tensor* norm1_b = nullptr;  // null when bias_free_ln
  ggml_tensor* norm2_w = nullptr;
  ggml_tensor* norm2_b = nullptr;
  ggml_tensor* ssmax_base0_w = nullptr;   // (1, 64)
  ggml_tensor* ssmax_base0_b = nullptr;   // (64)
  ggml_tensor* ssmax_base2_w = nullptr;   // (64, nh*hd)
  ggml_tensor* ssmax_base2_b = nullptr;   // (nh*hd)
  ggml_tensor* ssmax_query0_w = nullptr;  // (hd, 64)
  ggml_tensor* ssmax_query0_b = nullptr;
  ggml_tensor* ssmax_query2_w = nullptr;  // (64, hd)
  ggml_tensor* ssmax_query2_b = nullptr;
};

MabWeights mab_weights(const Model& m, const std::string& attn_prefix,
                       const std::string& block_prefix, bool ssmax) {
  MabWeights w;
  w.in_proj_w = m.tensor(attn_prefix + "in_proj_weight");
  w.in_proj_b = m.tensor(attn_prefix + "in_proj_bias");
  w.out_proj_w = m.tensor(attn_prefix + "out_proj.weight");
  w.out_proj_b = m.tensor(attn_prefix + "out_proj.bias");
  w.linear1_w = m.tensor(block_prefix + "linear1.weight");
  w.linear1_b = m.tensor(block_prefix + "linear1.bias");
  w.linear2_w = m.tensor(block_prefix + "linear2.weight");
  w.linear2_b = m.tensor(block_prefix + "linear2.bias");
  w.norm1_w = m.tensor(block_prefix + "norm1.weight");
  w.norm1_b = m.tensor_or_null(block_prefix + "norm1.bias");
  w.norm2_w = m.tensor(block_prefix + "norm2.weight");
  w.norm2_b = m.tensor_or_null(block_prefix + "norm2.bias");
  if (ssmax) {
    w.ssmax_base0_w = m.tensor(attn_prefix + "ssmax.base_mlp.0.weight");
    w.ssmax_base0_b = m.tensor(attn_prefix + "ssmax.base_mlp.0.bias");
    w.ssmax_base2_w = m.tensor(attn_prefix + "ssmax.base_mlp.2.weight");
    w.ssmax_base2_b = m.tensor(attn_prefix + "ssmax.base_mlp.2.bias");
    w.ssmax_query0_w = m.tensor(attn_prefix + "ssmax.query_mlp.0.weight");
    w.ssmax_query0_b = m.tensor(attn_prefix + "ssmax.query_mlp.0.bias");
    w.ssmax_query2_w = m.tensor(attn_prefix + "ssmax.query_mlp.2.weight");
    w.ssmax_query2_b = m.tensor(attn_prefix + "ssmax.query_mlp.2.bias");
  }
  return w;
}

// ---------------------------------------------------------------------------
// Graph fragments
// ---------------------------------------------------------------------------

ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w,
                        ggml_tensor* b, float eps) {
  ggml_tensor* y = ggml_norm(ctx, x, eps);
  y = ggml_mul(ctx, y, w);
  if (b) y = ggml_add(ctx, y, b);
  return y;
}

ggml_tensor* linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b,
                    ggml_tensor* x) {
  ggml_tensor* y = ggml_mul_mat(ctx, w, x);
  if (b) y = ggml_add(ctx, y, b);
  return y;
}

// Split (E, S, b2, b3) -> heads (hd, S, nh, b2*b3). Requires contiguous input.
ggml_tensor* split_heads(ggml_context* ctx, ggml_tensor* x, int64_t nh) {
  const int64_t E = x->ne[0], S = x->ne[1], batch = x->ne[2] * x->ne[3];
  ggml_tensor* t = ggml_reshape_4d(ctx, x, E / nh, nh, S, batch);
  return ggml_cont(ctx, ggml_permute(ctx, t, 0, 2, 1, 3));
}

// Merge (hd, S, nh, batch) -> (E, S, batch).
ggml_tensor* merge_heads(ggml_context* ctx, ggml_tensor* x) {
  const int64_t hd = x->ne[0], S = x->ne[1], nh = x->ne[2], batch = x->ne[3];
  ggml_tensor* t = ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
  return ggml_reshape_3d(ctx, t, hd * nh, S, batch);
}

// Non-interleaved RoPE: x*cos + rotate_half(x)*sin; cos/sin (hd, S, 1, 1)
// broadcast over heads and batch. x is (hd, S, nh, batch), contiguous.
ggml_tensor* apply_rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cos_t,
                        ggml_tensor* sin_t) {
  const int64_t hd = x->ne[0];
  ggml_tensor* lo = ggml_view_4d(ctx, x, hd / 2, x->ne[1], x->ne[2], x->ne[3],
                                 x->nb[1], x->nb[2], x->nb[3], 0);
  ggml_tensor* hi = ggml_view_4d(ctx, x, hd / 2, x->ne[1], x->ne[2], x->ne[3],
                                 x->nb[1], x->nb[2], x->nb[3],
                                 (hd / 2) * sizeof(float));
  ggml_tensor* rot = ggml_concat(ctx, ggml_neg(ctx, ggml_cont(ctx, hi)),
                                 ggml_cont(ctx, lo), 0);
  return ggml_add(ctx, ggml_mul(ctx, x, cos_t), ggml_mul(ctx, rot, sin_t));
}

struct MabOptions {
  int64_t nh = 8;
  float eps = 1e-5f;
  ggml_tensor* ssmax_base_scales = nullptr;  // (hd, 1, nh, 1); null = no SSMax
  ggml_tensor* rope_cos_q = nullptr;         // (hd, Sq, 1, 1); null = no RoPE
  ggml_tensor* rope_sin_q = nullptr;
  ggml_tensor* rope_cos_k = nullptr;
  ggml_tensor* rope_sin_k = nullptr;
  int64_t kv_train_prefix = -1;  // >=0: k = v = normed(q)[:, :prefix]
  int64_t repeat_batch = 0;      // >0: repeat projected Q to this ne3
  bool need_kv = false;          // expose the projected K/V head tensors
  ggml_tensor* cached_k = nullptr;  // (hd, Sk, nh, batch): use instead of
  ggml_tensor* cached_v = nullptr;  // computing K/V (cache-use path)
};

struct MabOut {
  ggml_tensor* out = nullptr;
  ggml_tensor* k = nullptr;  // set when need_kv
  ggml_tensor* v = nullptr;
};

// One MultiheadAttentionBlock (pre-norm TransformerEncoderLayer).
// q: (E, Sq, batch...); kv: nullptr/q for self-attention, or distinct tensor.
MabOut mab_block(ggml_context* ctx, const MabWeights& w, ggml_tensor* q,
                 ggml_tensor* kv, const MabOptions& o) {
  const int64_t E = q->ne[0];
  const int64_t hd = E / o.nh;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  ggml_tensor* qn = layer_norm(ctx, q, w.norm1_w, w.norm1_b, o.eps);

  // Packed in_proj slices (torch rows [0:E]=q, [E:2E]=k, [2E:3E]=v).
  const size_t nb1 = w.in_proj_w->nb[1];
  const int64_t ein = w.in_proj_w->ne[0];
  ggml_tensor* wq = ggml_view_2d(ctx, w.in_proj_w, ein, E, nb1, 0);
  ggml_tensor* bq = ggml_view_1d(ctx, w.in_proj_b, E, 0);
  ggml_tensor* Q = split_heads(ctx, ggml_add(ctx, ggml_mul_mat(ctx, wq, qn), bq), o.nh);

  ggml_tensor* K = nullptr;
  ggml_tensor* V = nullptr;
  if (o.cached_k) {
    K = o.cached_k;
    V = o.cached_v;
  } else {
    ggml_tensor* kn;
    if (o.kv_train_prefix >= 0) {
      kn = ggml_cont(ctx, ggml_view_4d(ctx, qn, E, o.kv_train_prefix, qn->ne[2],
                                       qn->ne[3], qn->nb[1], qn->nb[2], qn->nb[3], 0));
    } else if (kv == nullptr || kv == q) {
      kn = qn;
    } else {
      kn = layer_norm(ctx, kv, w.norm1_w, w.norm1_b, o.eps);
    }
    ggml_tensor* wk = ggml_view_2d(ctx, w.in_proj_w, ein, E, nb1, E * nb1);
    ggml_tensor* wv = ggml_view_2d(ctx, w.in_proj_w, ein, E, nb1, 2 * E * nb1);
    ggml_tensor* bk = ggml_view_1d(ctx, w.in_proj_b, E, E * sizeof(float));
    ggml_tensor* bv = ggml_view_1d(ctx, w.in_proj_b, E, 2 * E * sizeof(float));
    K = split_heads(ctx, ggml_add(ctx, ggml_mul_mat(ctx, wk, kn), bk), o.nh);
    V = split_heads(ctx, ggml_add(ctx, ggml_mul_mat(ctx, wv, kn), bv), o.nh);
  }

  if (o.rope_cos_q) Q = apply_rope(ctx, Q, o.rope_cos_q, o.rope_sin_q);
  if (o.rope_cos_k && !o.cached_k) K = apply_rope(ctx, K, o.rope_cos_k, o.rope_sin_k);

  if (o.repeat_batch > 0)
    Q = ggml_repeat_4d(ctx, Q, Q->ne[0], Q->ne[1], Q->ne[2], o.repeat_batch);

  if (o.ssmax_base_scales) {
    // modulation = 1 + tanh(query_mlp(q)); q *= base_scales * modulation
    ggml_tensor* m = linear(ctx, w.ssmax_query0_w, w.ssmax_query0_b, Q);
    m = ggml_gelu_erf(ctx, m);
    m = linear(ctx, w.ssmax_query2_w, w.ssmax_query2_b, m);
    m = ggml_scale_bias(ctx, ggml_tanh(ctx, m), 1.0f, 1.0f);  // 1 + tanh
    ggml_tensor* s = ggml_mul(ctx, m, o.ssmax_base_scales);
    Q = ggml_mul(ctx, Q, s);
  }

  ggml_tensor* scores = ggml_mul_mat(ctx, K, Q);  // (Sk, Sq, nh, batch)
  ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);
  ggml_tensor* Vt = ggml_cont(ctx, ggml_transpose(ctx, V));  // (Sk, hd, nh, batch)
  ggml_tensor* attn = ggml_mul_mat(ctx, Vt, probs);          // (hd, Sq, nh, batch)
  ggml_tensor* attn_out = linear(ctx, w.out_proj_w, w.out_proj_b,
                                 merge_heads(ctx, attn));

  // Residual on the RAW q (broadcasts when q is shared across the batch).
  ggml_tensor* x = ggml_add(ctx, attn_out, q);

  ggml_tensor* h = layer_norm(ctx, x, w.norm2_w, w.norm2_b, o.eps);
  h = linear(ctx, w.linear1_w, w.linear1_b, h);
  h = ggml_gelu_erf(ctx, h);
  h = linear(ctx, w.linear2_w, w.linear2_b, h);
  MabOut result;
  result.out = ggml_add(ctx, x, h);
  if (o.need_kv) {
    result.k = K;
    result.v = V;
  }
  return result;
}

// Host: SSMax base_mlp(ln(max(n,1))) -> (hd*nh) buffer laid out as the
// (hd, 1, nh, 1) input tensor. Computed in double, cast to fp32.
std::vector<float> ssmax_base_scales_host(const MabWeights& w, int64_t n) {
  const double logn = std::log(static_cast<double>(std::max<int64_t>(n, 1)));
  const auto* w0 = static_cast<const float*>(w.ssmax_base0_w->data);  // (1, 64)
  const auto* b0 = static_cast<const float*>(w.ssmax_base0_b->data);
  const auto* w2 = static_cast<const float*>(w.ssmax_base2_w->data);  // (64, nh*hd)
  const auto* b2 = static_cast<const float*>(w.ssmax_base2_b->data);
  const int64_t hidden = w.ssmax_base0_w->ne[1];
  const int64_t out_dim = w.ssmax_base2_w->ne[1];
  std::vector<double> h(static_cast<size_t>(hidden));
  for (int64_t i = 0; i < hidden; ++i) {
    const double v = static_cast<double>(w0[i]) * logn + static_cast<double>(b0[i]);
    h[static_cast<size_t>(i)] = 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0)));
  }
  std::vector<float> out(static_cast<size_t>(out_dim));
  for (int64_t o = 0; o < out_dim; ++o) {
    double acc = static_cast<double>(b2[o]);
    for (int64_t i = 0; i < hidden; ++i)
      acc += static_cast<double>(w2[o * hidden + i]) * h[static_cast<size_t>(i)];
    // Python view (1, nh, 1, hd): value for (head, dim) at flat h*hd + d.
    // ggml (hd, 1, nh, 1) buffer: flat d + hd*h — same flat index order.
    out[static_cast<size_t>(o)] = static_cast<float>(acc);
  }
  return out;
}

// Host: RoPE cos/sin tables, ggml (hd, n_pos) buffers, from checkpoint freqs.
void rope_tables_host(const Model& model, int64_t n_pos, std::vector<float>& cos_out,
                      std::vector<float>& sin_out) {
  ggml_tensor* freqs = model.tensor("row.rope.freqs");
  const auto* f = static_cast<const float*>(freqs->data);
  const int64_t half = freqs->ne[0];
  const int64_t hd = half * 2;
  cos_out.resize(static_cast<size_t>(hd * n_pos));
  sin_out.resize(static_cast<size_t>(hd * n_pos));
  for (int64_t p = 0; p < n_pos; ++p) {
    for (int64_t j = 0; j < half; ++j) {
      const float angle = static_cast<float>(p) * f[j];
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      cos_out[static_cast<size_t>(p * hd + j)] = c;
      cos_out[static_cast<size_t>(p * hd + half + j)] = c;
      sin_out[static_cast<size_t>(p * hd + j)] = s;
      sin_out[static_cast<size_t>(p * hd + half + j)] = s;
    }
  }
}

// Host: y-embedding lookup: W[:, y]+b (classification) or w*y+b (regression).
void y_embedding_host(const Model& model, const std::string& prefix,
                      const float* y, int64_t count, bool classification,
                      float* out /* (count, E) row-major */) {
  ggml_tensor* wt = model.tensor(prefix + ".weight");
  ggml_tensor* bt = model.tensor(prefix + ".bias");
  const auto* w = static_cast<const float*>(wt->data);
  const auto* b = static_cast<const float*>(bt->data);
  const int64_t in_dim = wt->ne[0];
  const int64_t E = wt->ne[1];
  for (int64_t i = 0; i < count; ++i) {
    if (classification) {
      const auto cls = static_cast<int64_t>(y[i]);
      for (int64_t e = 0; e < E; ++e) out[i * E + e] = w[e * in_dim + cls] + b[e];
    } else {
      for (int64_t e = 0; e < E; ++e) out[i * E + e] = w[e * in_dim] * y[i] + b[e];
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Stage 1: ColEmbedding
// ---------------------------------------------------------------------------

std::vector<float> col_embedding_forward(const Model& model, GraphRunner& runner,
                                         const float* X, const float* y_train,
                                         int64_t B, int64_t T, int64_t H,
                                         int64_t train_size, int64_t num_classes) {
  (void)num_classes;
  const ModelConfig& c = model.config();
  const int64_t E = c.embed_dim;
  const int64_t gs = c.col_feature_group_size;
  const int64_t nh = c.col_num_heads;
  const int64_t hd = E / nh;
  const bool clf = c.task == Task::Classification;

  // Mixed-radix ensembling: with C > max_classes the set transformer runs
  // once per mixed-radix digit of y and the outputs are averaged
  // (embedding.py _compute_embeddings).
  int64_t y_max = 0;
  if (clf)
    for (int64_t i = 0; i < B * train_size; ++i)
      y_max = std::max(y_max, static_cast<int64_t>(y_train[i]));
  const int64_t nc_seen = clf ? y_max + 1 : 0;
  std::vector<int64_t> bases{nc_seen};
  if (clf && nc_seen > c.max_classes) {
    const int64_t K = c.max_classes;
    const int64_t D =
        static_cast<int64_t>(std::ceil(std::log(static_cast<double>(nc_seen)) /
                                       std::log(static_cast<double>(K))));
    int64_t k = static_cast<int64_t>(
        std::ceil(std::pow(static_cast<double>(nc_seen), 1.0 / static_cast<double>(D))));
    k = std::min(k, K);
    bases.assign(static_cast<size_t>(D), k);
    int64_t product = 1;
    for (int64_t i = 0; i < D; ++i) product *= k;
    int64_t idx = 0;
    while (product < nc_seen && idx < D) {
      if (bases[static_cast<size_t>(idx)] < K) {
        product = product / bases[static_cast<size_t>(idx)] *
                  (bases[static_cast<size_t>(idx)] + 1);
        bases[static_cast<size_t>(idx)] += 1;
      }
      idx++;
    }
  }
  const int64_t n_digits =
      (clf && nc_seen > c.max_classes) ? static_cast<int64_t>(bases.size()) : 1;

  GraphExec ge(runner, static_cast<size_t>(4096 * n_digits));
  ggml_context* ctx = ge.ctx();

  ggml_tensor* xg = ge.input(gs, T, H, B);  // grouped features
  ggml_tensor* src0 = linear(ctx, model.tensor("col.in_linear.weight"),
                             model.tensor("col.in_linear.bias"), xg);  // (E,T,H,B)

  std::vector<ggml_tensor*> base_scale_inputs;  // one per ISAB block (shared)
  for (int b = 0; b < c.col_num_blocks; ++b) base_scale_inputs.push_back(ge.input(hd, 1, nh, 1));

  std::vector<ggml_tensor*> y_inputs;  // one per digit pass
  ggml_tensor* accum = nullptr;
  for (int64_t digit = 0; digit < n_digits; ++digit) {
    ggml_tensor* yfull = ge.input(E, T, 1, B);
    y_inputs.push_back(yfull);
    ggml_tensor* src = ggml_add(ctx, src0, yfull);

    for (int b = 0; b < c.col_num_blocks; ++b) {
      const std::string blk = "col.blk." + std::to_string(b) + ".";
      const MabWeights w1 = mab_weights(model, blk + "attn1.", blk + "attn1.", true);
      const MabWeights w2 = mab_weights(model, blk + "attn2.", blk + "attn2.", false);
      ggml_tensor* ind = model.tensor(blk + "ind_vectors");  // (E, M)

      ggml_tensor* src3 = ggml_reshape_3d(ctx, src, E, T, H * B);
      ggml_tensor* src_train = ggml_cont(
          ctx, ggml_view_3d(ctx, src3, E, train_size, H * B, src3->nb[1], src3->nb[2], 0));

      MabOptions o1;
      o1.nh = nh;
      o1.eps = c.norm_eps;
      o1.ssmax_base_scales = base_scale_inputs[static_cast<size_t>(b)];
      o1.repeat_batch = H * B;
      ggml_tensor* hidden = mab_block(ctx, w1, ind, src_train, o1).out;  // (E, M, H*B)

      MabOptions o2;
      o2.nh = nh;
      o2.eps = c.norm_eps;
      ggml_tensor* out3 = mab_block(ctx, w2, src3, hidden, o2).out;  // (E, T, H*B)
      src = ggml_reshape_4d(ctx, out3, E, T, H, B);
    }
    accum = accum ? ggml_add(ctx, accum, src) : src;
  }
  ggml_tensor* src = n_digits > 1
                         ? ggml_scale(ctx, accum, 1.0f / static_cast<float>(n_digits))
                         : accum;

  ge.finalize(src);

  {  // grouped features: X_grouped[b,t,g,i] = X[b, t, (g + 2^i) % H]
    std::vector<float> buf(static_cast<size_t>(gs * T * H * B));
    for (int64_t b = 0; b < B; ++b)
      for (int64_t g = 0; g < H; ++g)
        for (int64_t t = 0; t < T; ++t)
          for (int64_t i = 0; i < gs; ++i)
            buf[static_cast<size_t>(((b * H + g) * T + t) * gs + i)] =
                X[(b * T + t) * H + ((g + (int64_t(1) << i)) % H)];
    ge.set(xg, buf.data());
  }
  for (int64_t digit = 0; digit < n_digits; ++digit) {
    // y (or its mixed-radix digit) embedding, zero beyond train rows
    std::vector<float> buf(static_cast<size_t>(E * T * B), 0.0f);
    std::vector<float> ydig(static_cast<size_t>(train_size));
    for (int64_t b = 0; b < B; ++b) {
      const float* yb = y_train + b * train_size;
      if (n_digits > 1) {
        int64_t radix = 1;
        for (size_t j = static_cast<size_t>(digit) + 1; j < bases.size(); ++j)
          radix *= bases[j];
        for (int64_t t = 0; t < train_size; ++t) {
          const int64_t yi = static_cast<int64_t>(yb[t]);
          ydig[static_cast<size_t>(t)] =
              static_cast<float>((yi / radix) % bases[static_cast<size_t>(digit)]);
        }
        yb = ydig.data();
      }
      y_embedding_host(model, "col.y_encoder", yb, train_size, clf,
                       &buf[static_cast<size_t>(b * T * E)]);
    }
    ge.set(y_inputs[static_cast<size_t>(digit)], buf.data());
  }
  for (int b = 0; b < c.col_num_blocks; ++b) {
    const std::string blk = "col.blk." + std::to_string(b) + ".";
    const MabWeights w1 = mab_weights(model, blk + "attn1.", blk + "attn1.", true);
    const auto scales = ssmax_base_scales_host(w1, train_size);
    ge.set(base_scale_inputs[static_cast<size_t>(b)], scales.data());
  }

  ge.compute();

  // ggml (E, T, H, B): flat e + E*(t + T*(h + H*b)) -> host (B, T, H, E).
  const auto* od = static_cast<const float*>(src->data);
  std::vector<float> result(static_cast<size_t>(B * T * H * E));
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < H; ++h)
      for (int64_t t = 0; t < T; ++t)
        std::memcpy(&result[static_cast<size_t>(((b * T + t) * H + h) * E)],
                    &od[static_cast<size_t>(E * (t + T * (h + H * b)))],
                    static_cast<size_t>(E) * sizeof(float));
  return result;
}

// ---------------------------------------------------------------------------
// Stage 2: RowInteraction
// ---------------------------------------------------------------------------

std::vector<float> row_interaction_forward(const Model& model, GraphRunner& runner,
                                           const float* col_out, int64_t B,
                                           int64_t T, int64_t H) {
  const ModelConfig& c = model.config();
  const int64_t E = c.embed_dim;
  const int64_t C = c.row_num_cls;
  const int64_t S = H + C;  // sequence: CLS slots then features
  const int64_t nh = c.row_num_heads;
  const int64_t hd = E / nh;
  const int64_t D = c.icl_dim();

  GraphExec ge(runner, 4096);
  ggml_context* ctx = ge.ctx();

  ggml_tensor* emb = ge.input(E, S, T * B);
  ggml_tensor* cos_full = ge.input(hd, S);
  ggml_tensor* sin_full = ge.input(hd, S);
  // CLS query positions are 0..C-1: views of the full tables.
  ggml_tensor* cos_cls = ggml_view_2d(ctx, cos_full, hd, C, cos_full->nb[1], 0);
  ggml_tensor* sin_cls = ggml_view_2d(ctx, sin_full, hd, C, sin_full->nb[1], 0);

  ggml_tensor* x = emb;
  for (int b = 0; b < c.row_num_blocks - 1; ++b) {
    const std::string blk = "row.blk." + std::to_string(b) + ".";
    const MabWeights w = mab_weights(model, blk + "attn.", blk, false);
    MabOptions o;
    o.nh = nh;
    o.eps = c.norm_eps;
    o.rope_cos_q = cos_full;
    o.rope_sin_q = sin_full;
    o.rope_cos_k = cos_full;
    o.rope_sin_k = sin_full;
    x = mab_block(ctx, w, x, nullptr, o).out;
  }
  {  // last block: cross-attention with only the CLS queries
    const std::string blk =
        "row.blk." + std::to_string(c.row_num_blocks - 1) + ".";
    const MabWeights w = mab_weights(model, blk + "attn.", blk, false);
    ggml_tensor* qcls = ggml_cont(
        ctx, ggml_view_3d(ctx, x, E, C, x->ne[2], x->nb[1], x->nb[2], 0));
    MabOptions o;
    o.nh = nh;
    o.eps = c.norm_eps;
    o.rope_cos_q = cos_cls;
    o.rope_sin_q = sin_cls;
    o.rope_cos_k = cos_full;
    o.rope_sin_k = sin_full;
    x = mab_block(ctx, w, qcls, x, o).out;  // (E, C, T*B)
  }
  x = layer_norm(ctx, x, model.tensor("row.out_ln.weight"),
                 model.tensor_or_null("row.out_ln.bias"), c.norm_eps);

  ge.finalize(x);

  {  // emb: CLS tokens then col embeddings, per (b, t)
    ggml_tensor* cls = model.tensor("row.cls_tokens");  // (E, C)
    const auto* cd = static_cast<const float*>(cls->data);
    std::vector<float> buf(static_cast<size_t>(E * S * T * B));
    for (int64_t b = 0; b < B; ++b)
      for (int64_t t = 0; t < T; ++t) {
        float* dst = &buf[static_cast<size_t>(((b * T + t)) * S * E)];
        std::memcpy(dst, cd, static_cast<size_t>(C * E) * sizeof(float));
        std::memcpy(dst + C * E, &col_out[static_cast<size_t>(((b * T + t) * H) * E)],
                    static_cast<size_t>(H * E) * sizeof(float));
      }
    ge.set(emb, buf.data());
  }
  {
    std::vector<float> cos_t, sin_t;
    rope_tables_host(model, S, cos_t, sin_t);
    ge.set(cos_full, cos_t.data());
    ge.set(sin_full, sin_t.data());
  }

  ge.compute();

  // ggml (E, C, T*B) buffer == host (B, T, C*E) row-major directly.
  const auto* od = static_cast<const float*>(x->data);
  return std::vector<float>(od, od + B * T * D);
}

// ---------------------------------------------------------------------------
// Stage 3: ICLearning
// ---------------------------------------------------------------------------

std::vector<float> icl_forward(const Model& model, GraphRunner& runner,
                               const float* reprs, const float* y_train,
                               int64_t B, int64_t T, int64_t train_size,
                               int64_t num_classes) {
  (void)num_classes;
  const ModelConfig& c = model.config();
  const int64_t D = c.icl_dim();
  const int64_t nh = c.icl_num_heads;
  const int64_t hd = D / nh;
  const bool clf = c.task == Task::Classification;
  const int64_t out_dim = clf ? c.max_classes : c.num_quantiles;

  GraphExec ge(runner, 4096);
  ggml_context* ctx = ge.ctx();

  ggml_tensor* R = ge.input(D, T, B);
  std::vector<ggml_tensor*> base_scale_inputs;

  ggml_tensor* x = R;
  for (int b = 0; b < c.icl_num_blocks; ++b) {
    const std::string blk = "icl.blk." + std::to_string(b) + ".";
    const MabWeights w = mab_weights(model, blk + "attn.", blk, true);
    ggml_tensor* base_in = ge.input(hd, 1, nh, 1);
    base_scale_inputs.push_back(base_in);
    MabOptions o;
    o.nh = nh;
    o.eps = c.norm_eps;
    o.ssmax_base_scales = base_in;
    o.kv_train_prefix = train_size;
    x = mab_block(ctx, w, x, nullptr, o).out;
  }
  x = layer_norm(ctx, x, model.tensor("icl.ln.weight"),
                 model.tensor_or_null("icl.ln.bias"), c.norm_eps);
  x = linear(ctx, model.tensor("icl.decoder.0.weight"),
             model.tensor("icl.decoder.0.bias"), x);
  x = ggml_gelu_erf(ctx, x);
  x = linear(ctx, model.tensor("icl.decoder.2.weight"),
             model.tensor("icl.decoder.2.bias"), x);  // (out_dim, T, B)

  ge.finalize(x);

  {  // R with y embedding added to the train prefix (skipped when y_train is
     // null — repr-cache use path, where y was injected at cache build time)
    std::vector<float> buf(reprs, reprs + B * T * D);
    if (y_train) {
      std::vector<float> yemb(static_cast<size_t>(train_size * D));
      for (int64_t b = 0; b < B; ++b) {
        y_embedding_host(model, "icl.y_encoder", y_train + b * train_size, train_size,
                         clf, yemb.data());
        for (int64_t t = 0; t < train_size; ++t)
          for (int64_t e = 0; e < D; ++e)
            buf[static_cast<size_t>((b * T + t) * D + e)] +=
                yemb[static_cast<size_t>(t * D + e)];
      }
    }
    ge.set(R, buf.data());
  }
  for (int b = 0; b < c.icl_num_blocks; ++b) {
    const std::string blk = "icl.blk." + std::to_string(b) + ".";
    const MabWeights w = mab_weights(model, blk + "attn.", blk, true);
    const auto scales = ssmax_base_scales_host(w, train_size);
    ge.set(base_scale_inputs[static_cast<size_t>(b)], scales.data());
  }

  ge.compute();

  const auto* od = static_cast<const float*>(x->data);
  return std::vector<float>(od, od + B * T * out_dim);
}

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

void TabICLCache::append(const TabICLCache& other) {
  if (B == 0) {
    *this = other;
    return;
  }
  if (mode != other.mode || train_size != other.train_size || H != other.H)
    throw std::runtime_error("cache append: geometry mismatch");
  if (mode == Mode::KV) {
    // col buffers are (hd, M, nh, H*B): concatenating along ne3 is a plain
    // buffer concat; same for icl (hd, train, nh, B) and repr (B, train, D).
    for (size_t i = 0; i < col_k.size(); ++i) {
      col_k[i].insert(col_k[i].end(), other.col_k[i].begin(), other.col_k[i].end());
      col_v[i].insert(col_v[i].end(), other.col_v[i].begin(), other.col_v[i].end());
    }
    for (size_t i = 0; i < icl_k.size(); ++i) {
      icl_k[i].insert(icl_k[i].end(), other.icl_k[i].begin(), other.icl_k[i].end());
      icl_v[i].insert(icl_v[i].end(), other.icl_v[i].begin(), other.icl_v[i].end());
    }
  } else {
    for (size_t i = 0; i < col_k.size(); ++i) {
      col_k[i].insert(col_k[i].end(), other.col_k[i].begin(), other.col_k[i].end());
      col_v[i].insert(col_v[i].end(), other.col_v[i].begin(), other.col_v[i].end());
    }
    row_repr.insert(row_repr.end(), other.row_repr.begin(), other.row_repr.end());
  }
  B += other.B;
}

namespace {

// Col stage on training rows with attn2 K/V capture. Returns the col
// embeddings (host (B, train, H, E)) and fills cache col_k/col_v.
std::vector<float> col_stage_build_cache(const Model& model, GraphRunner& runner,
                                         const float* X, const float* y_train,
                                         int64_t B, int64_t train, int64_t H,
                                         TabICLCache& cache) {
  const ModelConfig& c = model.config();
  const int64_t E = c.embed_dim;
  const int64_t gs = c.col_feature_group_size;
  const int64_t nh = c.col_num_heads;
  const int64_t hd = E / nh;
  const bool clf = c.task == Task::Classification;
  const int64_t T = train;  // train-only store pass

  GraphExec ge(runner, 4096);
  ggml_context* ctx = ge.ctx();
  ggml_tensor* xg = ge.input(gs, T, H, B);
  ggml_tensor* yfull = ge.input(E, T, 1, B);
  ggml_tensor* src = linear(ctx, model.tensor("col.in_linear.weight"),
                            model.tensor("col.in_linear.bias"), xg);
  src = ggml_add(ctx, src, yfull);

  std::vector<ggml_tensor*> base_scale_inputs, kv_outs;
  for (int b = 0; b < c.col_num_blocks; ++b) {
    const std::string blk = "col.blk." + std::to_string(b) + ".";
    const MabWeights w1 = mab_weights(model, blk + "attn1.", blk + "attn1.", true);
    const MabWeights w2 = mab_weights(model, blk + "attn2.", blk + "attn2.", false);
    ggml_tensor* ind = model.tensor(blk + "ind_vectors");

    ggml_tensor* src3 = ggml_reshape_3d(ctx, src, E, T, H * B);
    ggml_tensor* src_train = ggml_cont(
        ctx, ggml_view_3d(ctx, src3, E, T, H * B, src3->nb[1], src3->nb[2], 0));
    ggml_tensor* base_in = ge.input(hd, 1, nh, 1);
    base_scale_inputs.push_back(base_in);

    MabOptions o1;
    o1.nh = nh;
    o1.eps = c.norm_eps;
    o1.ssmax_base_scales = base_in;
    o1.repeat_batch = H * B;
    ggml_tensor* hidden = mab_block(ctx, w1, ind, src_train, o1).out;

    MabOptions o2;
    o2.nh = nh;
    o2.eps = c.norm_eps;
    o2.need_kv = true;
    MabOut r2 = mab_block(ctx, w2, src3, hidden, o2);
    kv_outs.push_back(r2.k);
    kv_outs.push_back(r2.v);
    src = ggml_reshape_4d(ctx, r2.out, E, T, H, B);
  }

  std::vector<ggml_tensor*> outs = kv_outs;
  outs.push_back(src);
  ge.finalize(outs);

  {
    std::vector<float> buf(static_cast<size_t>(gs * T * H * B));
    for (int64_t b = 0; b < B; ++b)
      for (int64_t g = 0; g < H; ++g)
        for (int64_t t = 0; t < T; ++t)
          for (int64_t i = 0; i < gs; ++i)
            buf[static_cast<size_t>(((b * H + g) * T + t) * gs + i)] =
                X[(b * T + t) * H + ((g + (int64_t(1) << i)) % H)];
    ge.set(xg, buf.data());
  }
  {
    std::vector<float> buf(static_cast<size_t>(E * T * B), 0.0f);
    for (int64_t b = 0; b < B; ++b)
      y_embedding_host(model, "col.y_encoder", y_train + b * train, train, clf,
                       &buf[static_cast<size_t>(b * T * E)]);
    ge.set(yfull, buf.data());
  }
  for (int b = 0; b < c.col_num_blocks; ++b) {
    const std::string blk = "col.blk." + std::to_string(b) + ".";
    const MabWeights w1 = mab_weights(model, blk + "attn1.", blk + "attn1.", true);
    const auto scales = ssmax_base_scales_host(w1, train);
    ge.set(base_scale_inputs[static_cast<size_t>(b)], scales.data());
  }
  ge.compute();

  cache.col_k.clear();
  cache.col_v.clear();
  for (int b = 0; b < c.col_num_blocks; ++b) {
    const auto* kd = static_cast<const float*>(kv_outs[static_cast<size_t>(2 * b)]->data);
    const auto* vd = static_cast<const float*>(kv_outs[static_cast<size_t>(2 * b + 1)]->data);
    const int64_t n = ggml_nelements(kv_outs[static_cast<size_t>(2 * b)]);
    cache.col_k.emplace_back(kd, kd + n);
    cache.col_v.emplace_back(vd, vd + n);
  }

  const auto* od = static_cast<const float*>(src->data);
  std::vector<float> result(static_cast<size_t>(B * T * H * E));
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < H; ++h)
      for (int64_t t = 0; t < T; ++t)
        std::memcpy(&result[static_cast<size_t>(((b * T + t) * H + h) * E)],
                    &od[static_cast<size_t>(E * (t + T * (h + H * b)))],
                    static_cast<size_t>(E) * sizeof(float));
  return result;
}

// Col stage on test rows using cached attn2 K/V. Returns (B, n_test, H, E).
std::vector<float> col_stage_use_cache(const Model& model, GraphRunner& runner,
                                       const float* X_test, int64_t n_test,
                                       const TabICLCache& cache) {
  const ModelConfig& c = model.config();
  const int64_t E = c.embed_dim;
  const int64_t gs = c.col_feature_group_size;
  const int64_t nh = c.col_num_heads;
  const int64_t hd = E / nh;
  const int64_t M = c.col_num_inds;
  const int64_t H = cache.H, B = cache.B, T = n_test;

  GraphExec ge(runner, 4096);
  ggml_context* ctx = ge.ctx();
  ggml_tensor* xg = ge.input(gs, T, H, B);
  ggml_tensor* src = linear(ctx, model.tensor("col.in_linear.weight"),
                            model.tensor("col.in_linear.bias"), xg);
  // No y embedding: test rows only.

  std::vector<ggml_tensor*> cache_inputs;
  for (int b = 0; b < c.col_num_blocks; ++b) {
    const std::string blk = "col.blk." + std::to_string(b) + ".";
    const MabWeights w2 = mab_weights(model, blk + "attn2.", blk + "attn2.", false);
    ggml_tensor* src3 = ggml_reshape_3d(ctx, src, E, T, H * B);
    ggml_tensor* ck = ge.input(hd, M, nh, H * B);
    ggml_tensor* cv = ge.input(hd, M, nh, H * B);
    cache_inputs.push_back(ck);
    cache_inputs.push_back(cv);
    MabOptions o2;
    o2.nh = nh;
    o2.eps = c.norm_eps;
    o2.cached_k = ck;
    o2.cached_v = cv;
    ggml_tensor* out3 = mab_block(ctx, w2, src3, nullptr, o2).out;
    src = ggml_reshape_4d(ctx, out3, E, T, H, B);
  }
  ge.finalize(src);

  {
    std::vector<float> buf(static_cast<size_t>(gs * T * H * B));
    for (int64_t b = 0; b < B; ++b)
      for (int64_t g = 0; g < H; ++g)
        for (int64_t t = 0; t < T; ++t)
          for (int64_t i = 0; i < gs; ++i)
            buf[static_cast<size_t>(((b * H + g) * T + t) * gs + i)] =
                X_test[(b * T + t) * H + ((g + (int64_t(1) << i)) % H)];
    ge.set(xg, buf.data());
  }
  for (int b = 0; b < c.col_num_blocks; ++b) {
    ge.set(cache_inputs[static_cast<size_t>(2 * b)], cache.col_k[static_cast<size_t>(b)].data());
    ge.set(cache_inputs[static_cast<size_t>(2 * b + 1)], cache.col_v[static_cast<size_t>(b)].data());
  }
  ge.compute();

  const auto* od = static_cast<const float*>(src->data);
  std::vector<float> result(static_cast<size_t>(B * T * H * E));
  for (int64_t b = 0; b < B; ++b)
    for (int64_t h = 0; h < H; ++h)
      for (int64_t t = 0; t < T; ++t)
        std::memcpy(&result[static_cast<size_t>(((b * T + t) * H + h) * E)],
                    &od[static_cast<size_t>(E * (t + T * (h + H * b)))],
                    static_cast<size_t>(E) * sizeof(float));
  return result;
}

// ICL on training rows, capturing per-layer train K/V into the cache.
void icl_build_cache(const Model& model, GraphRunner& runner, const float* reprs,
                     const float* y_train, int64_t B, int64_t train,
                     TabICLCache& cache) {
  const ModelConfig& c = model.config();
  const int64_t D = c.icl_dim();
  const int64_t nh = c.icl_num_heads;
  const int64_t hd = D / nh;
  const bool clf = c.task == Task::Classification;

  GraphExec ge(runner, 4096);
  ggml_context* ctx = ge.ctx();
  ggml_tensor* R = ge.input(D, train, B);
  std::vector<ggml_tensor*> base_scale_inputs, kv_outs;

  ggml_tensor* x = R;
  for (int b = 0; b < c.icl_num_blocks; ++b) {
    const std::string blk = "icl.blk." + std::to_string(b) + ".";
    const MabWeights w = mab_weights(model, blk + "attn.", blk, true);
    ggml_tensor* base_in = ge.input(hd, 1, nh, 1);
    base_scale_inputs.push_back(base_in);
    MabOptions o;
    o.nh = nh;
    o.eps = c.norm_eps;
    o.ssmax_base_scales = base_in;
    o.kv_train_prefix = train;
    o.need_kv = true;
    MabOut r = mab_block(ctx, w, x, nullptr, o);
    kv_outs.push_back(r.k);
    kv_outs.push_back(r.v);
    x = r.out;
  }
  std::vector<ggml_tensor*> outs = kv_outs;
  outs.push_back(x);
  ge.finalize(outs);

  {
    std::vector<float> buf(reprs, reprs + B * train * D);
    std::vector<float> yemb(static_cast<size_t>(train * D));
    for (int64_t b = 0; b < B; ++b) {
      y_embedding_host(model, "icl.y_encoder", y_train + b * train, train, clf,
                       yemb.data());
      for (int64_t t = 0; t < train; ++t)
        for (int64_t e = 0; e < D; ++e)
          buf[static_cast<size_t>((b * train + t) * D + e)] +=
              yemb[static_cast<size_t>(t * D + e)];
    }
    ge.set(R, buf.data());
  }
  for (int b = 0; b < c.icl_num_blocks; ++b) {
    const std::string blk = "icl.blk." + std::to_string(b) + ".";
    const MabWeights w = mab_weights(model, blk + "attn.", blk, true);
    const auto scales = ssmax_base_scales_host(w, train);
    ge.set(base_scale_inputs[static_cast<size_t>(b)], scales.data());
  }
  ge.compute();

  cache.icl_k.clear();
  cache.icl_v.clear();
  for (int b = 0; b < c.icl_num_blocks; ++b) {
    const auto* kd = static_cast<const float*>(kv_outs[static_cast<size_t>(2 * b)]->data);
    const auto* vd = static_cast<const float*>(kv_outs[static_cast<size_t>(2 * b + 1)]->data);
    const int64_t n = ggml_nelements(kv_outs[static_cast<size_t>(2 * b)]);
    cache.icl_k.emplace_back(kd, kd + n);
    cache.icl_v.emplace_back(vd, vd + n);
  }
}

// ICL on test rows against cached per-layer train K/V; returns raw decoder
// output (B, n_test, out_dim_full).
std::vector<float> icl_use_cache(const Model& model, GraphRunner& runner,
                                 const float* reprs_test, int64_t n_test,
                                 const TabICLCache& cache) {
  const ModelConfig& c = model.config();
  const int64_t D = c.icl_dim();
  const int64_t nh = c.icl_num_heads;
  const int64_t hd = D / nh;
  const int64_t B = cache.B, train = cache.train_size;
  const int64_t out_dim = c.task == Task::Classification ? c.max_classes
                                                         : c.num_quantiles;

  GraphExec ge(runner, 4096);
  ggml_context* ctx = ge.ctx();
  ggml_tensor* R = ge.input(D, n_test, B);
  std::vector<ggml_tensor*> base_scale_inputs, cache_inputs;

  ggml_tensor* x = R;
  for (int b = 0; b < c.icl_num_blocks; ++b) {
    const std::string blk = "icl.blk." + std::to_string(b) + ".";
    const MabWeights w = mab_weights(model, blk + "attn.", blk, true);
    ggml_tensor* base_in = ge.input(hd, 1, nh, 1);
    base_scale_inputs.push_back(base_in);
    ggml_tensor* ck = ge.input(hd, train, nh, B);
    ggml_tensor* cv = ge.input(hd, train, nh, B);
    cache_inputs.push_back(ck);
    cache_inputs.push_back(cv);
    MabOptions o;
    o.nh = nh;
    o.eps = c.norm_eps;
    o.ssmax_base_scales = base_in;
    o.cached_k = ck;
    o.cached_v = cv;
    x = mab_block(ctx, w, x, nullptr, o).out;
  }
  x = layer_norm(ctx, x, model.tensor("icl.ln.weight"),
                 model.tensor_or_null("icl.ln.bias"), c.norm_eps);
  x = linear(ctx, model.tensor("icl.decoder.0.weight"),
             model.tensor("icl.decoder.0.bias"), x);
  x = ggml_gelu_erf(ctx, x);
  x = linear(ctx, model.tensor("icl.decoder.2.weight"),
             model.tensor("icl.decoder.2.bias"), x);
  ge.finalize(x);

  ge.set(R, reprs_test);
  for (int b = 0; b < c.icl_num_blocks; ++b) {
    const std::string blk = "icl.blk." + std::to_string(b) + ".";
    const MabWeights w = mab_weights(model, blk + "attn.", blk, true);
    // SSMax n = cached key length = train_size (matches Python).
    const auto scales = ssmax_base_scales_host(w, train);
    ge.set(base_scale_inputs[static_cast<size_t>(b)], scales.data());
    ge.set(cache_inputs[static_cast<size_t>(2 * b)], cache.icl_k[static_cast<size_t>(b)].data());
    ge.set(cache_inputs[static_cast<size_t>(2 * b + 1)], cache.icl_v[static_cast<size_t>(b)].data());
  }
  ge.compute();

  const auto* od = static_cast<const float*>(x->data);
  return std::vector<float>(od, od + B * n_test * out_dim);
}

}  // namespace

TabICLCache tabicl_build_cache(const Model& model, GraphRunner& runner,
                               const float* X_train, const float* y_train,
                               int64_t B, int64_t train_size, int64_t H,
                               TabICLCache::Mode mode) {
  TabICLCache cache;
  cache.mode = mode;
  cache.B = B;
  cache.train_size = train_size;
  cache.H = H;

  const auto col = col_stage_build_cache(model, runner, X_train, y_train, B,
                                         train_size, H, cache);
  const auto reprs = row_interaction_forward(model, runner, col.data(), B,
                                             train_size, H);
  if (mode == TabICLCache::Mode::Repr) {
    // Store representations WITH the y embedding injected (prepare_repr_cache).
    const ModelConfig& c = model.config();
    const int64_t D = c.icl_dim();
    const bool clf = c.task == Task::Classification;
    cache.row_repr = reprs;
    std::vector<float> yemb(static_cast<size_t>(train_size * D));
    for (int64_t b = 0; b < B; ++b) {
      y_embedding_host(model, "icl.y_encoder", y_train + b * train_size, train_size,
                       clf, yemb.data());
      for (int64_t t = 0; t < train_size; ++t)
        for (int64_t e = 0; e < D; ++e)
          cache.row_repr[static_cast<size_t>((b * train_size + t) * D + e)] +=
              yemb[static_cast<size_t>(t * D + e)];
    }
  } else {
    icl_build_cache(model, runner, reprs.data(), y_train, B, train_size, cache);
  }
  return cache;
}

std::vector<float> tabicl_forward_cached(const Model& model, GraphRunner& runner,
                                         const float* X_test, int64_t n_test,
                                         const TabICLCache& cache,
                                         const ForwardOptions& opts) {
  const ModelConfig& c = model.config();
  const bool clf = c.task == Task::Classification;
  const int64_t B = cache.B, train = cache.train_size, H = cache.H;
  const int64_t full_out = clf ? c.max_classes : c.num_quantiles;
  const int64_t out_dim = clf ? opts.num_classes : c.num_quantiles;
  const int64_t D = c.icl_dim();

  const auto col = col_stage_use_cache(model, runner, X_test, n_test, cache);
  const auto reprs = row_interaction_forward(model, runner, col.data(), B, n_test, H);

  std::vector<float> raw;
  int64_t raw_T = 0, raw_off = 0;
  if (cache.mode == TabICLCache::Mode::KV) {
    raw = icl_use_cache(model, runner, reprs.data(), n_test, cache);
    raw_T = n_test;
    raw_off = 0;
  } else {
    // Repr mode: concat cached (y-injected) train reprs with test reprs and
    // run the full ICL stack (y must NOT be injected again).
    const int64_t T = train + n_test;
    std::vector<float> full(static_cast<size_t>(B * T * D));
    for (int64_t b = 0; b < B; ++b) {
      std::memcpy(&full[static_cast<size_t>(b * T * D)],
                  &cache.row_repr[static_cast<size_t>(b * train * D)],
                  static_cast<size_t>(train * D) * sizeof(float));
      std::memcpy(&full[static_cast<size_t>((b * T + train) * D)],
                  &reprs[static_cast<size_t>(b * n_test * D)],
                  static_cast<size_t>(n_test * D) * sizeof(float));
    }
    // icl_forward would inject y; call with a zero-length y via the dedicated
    // no-inject path: y embedding add is skipped when y_train == nullptr.
    raw = icl_forward(model, runner, full.data(), nullptr, B, T, train, opts.num_classes);
    raw_T = T;
    raw_off = train;
  }

  std::vector<float> out(static_cast<size_t>(B * n_test * out_dim));
  for (int64_t b = 0; b < B; ++b)
    for (int64_t t = 0; t < n_test; ++t)
      std::memcpy(&out[static_cast<size_t>((b * n_test + t) * out_dim)],
                  &raw[static_cast<size_t>((b * raw_T + raw_off + t) * full_out)],
                  static_cast<size_t>(out_dim) * sizeof(float));
  return out;
}

// ---------------------------------------------------------------------------
// Full forward
// ---------------------------------------------------------------------------

std::vector<float> tabicl_forward(const Model& model, GraphRunner& runner,
                                  const float* X, const float* y_train,
                                  int64_t B, int64_t T, int64_t H,
                                  const ForwardOptions& opts) {
  const ModelConfig& c = model.config();
  const bool clf = c.task == Task::Classification;
  const int64_t train = opts.train_size;
  const int64_t test = T - train;
  const int64_t full_out = clf ? c.max_classes : c.num_quantiles;
  const int64_t out_dim = clf ? opts.num_classes : c.num_quantiles;

  const auto col = col_embedding_forward(model, runner, X, y_train, B, T, H, train,
                                         opts.num_classes);
  const auto reprs = row_interaction_forward(model, runner, col.data(), B, T, H);
  const auto raw =
      icl_forward(model, runner, reprs.data(), y_train, B, T, train, opts.num_classes);

  // Slice test rows and (classifier) the first num_classes outputs.
  std::vector<float> out(static_cast<size_t>(B * test * out_dim));
  for (int64_t b = 0; b < B; ++b)
    for (int64_t t = 0; t < test; ++t)
      std::memcpy(&out[static_cast<size_t>((b * test + t) * out_dim)],
                  &raw[static_cast<size_t>((b * T + train + t) * full_out)],
                  static_cast<size_t>(out_dim) * sizeof(float));

  if (clf && !opts.return_logits) {
    // softmax(logits / temperature), max-subtracted, per row (fp32).
    const float tau = c.softmax_temperature;
    for (int64_t r = 0; r < B * test; ++r) {
      float* row = &out[static_cast<size_t>(r * out_dim)];
      float mx = -std::numeric_limits<float>::infinity();
      for (int64_t k = 0; k < out_dim; ++k) {
        row[k] /= tau;
        mx = std::max(mx, row[k]);
      }
      float denom = 0.0f;
      for (int64_t k = 0; k < out_dim; ++k) {
        row[k] = std::exp(row[k] - mx);
        denom += row[k];
      }
      for (int64_t k = 0; k < out_dim; ++k) row[k] /= denom;
    }
  }
  return out;
}

}  // namespace tabicl
