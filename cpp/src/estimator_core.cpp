#include "estimator_core.h"

#include <cmath>
#include <stdexcept>

#include "ggml.h"
#include "hierarchical.h"
#include "io/sha256.h"
#include "model_forward.h"

namespace tabicl {

std::vector<int64_t> array_split_sizes(int64_t n, int64_t k) {
  std::vector<int64_t> sizes;
  if (k <= 0) return sizes;
  const int64_t base = n / k, extra = n % k;
  for (int64_t i = 0; i < k; ++i) sizes.push_back(base + (i < extra ? 1 : 0));
  return sizes;
}

void EstimatorCore::fit(const double* X, int64_t n, int64_t d,
                        std::vector<float> y_model, int64_t n_classes,
                        const EstimatorOptions& opts, const Model* model) {
  opts_ = opts;
  d_in_ = d;
  n_classes_ = n_classes;
  y_model_ = std::move(y_model);
  if (static_cast<int64_t>(y_model_.size()) != n)
    throw std::runtime_error("fit: X and y size mismatch");

  Matrix Xm{std::vector<double>(X, X + n * d), n, d};
  imputer_.fit(Xm);
  Matrix Xi = imputer_.transform(Xm);
  for (double v : Xi.data)
    if (std::isinf(v))
      throw std::runtime_error("fit: input contains infinity");
  filter_.fit(Xi);
  Matrix Xu = filter_.transform(Xi);
  if (Xu.d == 0)
    throw std::runtime_error("fit: no informative features after filtering");

  configs_ = generate_ensemble_configs(Xu.d, n_classes_, opts_.n_estimators,
                                       opts_.norm_methods, "latin", "shift",
                                       opts_.random_state);
  pipelines_.clear();
  for (const std::string& method : configs_.method_order) {
    auto p = std::make_unique<PreprocessingPipeline>(method, 4.0, opts_.random_state);
    p->fit(Xu);
    pipelines_.push_back(std::move(p));
  }
  runner_ = std::make_unique<GraphRunner>(opts_.n_threads);

  caches_.clear();
  if (opts_.cache != CacheMode::None) {
    if (!model) throw std::runtime_error("fit: cache requested but no model given");
    if (n_classes_ > model->config().max_classes)
      throw std::runtime_error("fit: KV cache incompatible with > max_classes classes");
    const auto mode = opts_.cache == CacheMode::KV ? TabICLCache::Mode::KV
                                                   : TabICLCache::Mode::Repr;
    const int64_t train = n;
    const int64_t H = Xu.d;
    caches_.resize(configs_.method_order.size());
    for (size_t gi = 0; gi < configs_.method_order.size(); ++gi) {
      const auto& members = configs_.members[gi];
      const Matrix& train_pp = pipelines_[gi]->train_transformed();
      const int64_t n_members = static_cast<int64_t>(members.size());
      const int64_t n_batches = (n_members + opts_.batch_size - 1) / opts_.batch_size;
      const auto sizes = array_split_sizes(n_members, n_batches);
      int64_t mi = 0;
      for (int64_t bsize : sizes) {
        std::vector<float> Xs(static_cast<size_t>(bsize * train * H));
        std::vector<float> ys(static_cast<size_t>(bsize * train));
        for (int64_t b = 0; b < bsize; ++b) {
          const EnsembleMember& m = members[static_cast<size_t>(mi + b)];
          for (int64_t t = 0; t < train; ++t) {
            float* dst = &Xs[static_cast<size_t>((b * train + t) * H)];
            for (int64_t j = 0; j < H; ++j)
              dst[j] = static_cast<float>(
                  train_pp.at(t, m.feature_shuffle[static_cast<size_t>(j)]));
            const float yv = y_model_[static_cast<size_t>(t)];
            ys[static_cast<size_t>(b * train + t)] =
                n_classes_ > 0
                    ? static_cast<float>(m.class_shuffle[static_cast<size_t>(
                          static_cast<int64_t>(yv))])
                    : yv;
          }
        }
        caches_[gi].push_back(tabicl_build_cache(*model, *runner_, Xs.data(),
                                                 ys.data(), bsize, train, H, mode,
                                                 opts_.max_scratch_bytes));
        mi += bsize;
      }
    }
  }
}

EstimatorCore::EnsembleOutputs EstimatorCore::predict_outputs(
    const Model& model, const double* X_test, int64_t n_test) const {
  const ModelConfig& c = model.config();
  const bool clf = c.task == Task::Classification;
  const int64_t train = train_size();
  const int64_t T = train + n_test;
  const int64_t out_dim = clf ? n_classes_ : c.num_quantiles;

  Matrix Xm{std::vector<double>(X_test, X_test + n_test * d_in_), n_test, d_in_};
  Matrix Xi = imputer_.transform(Xm);
  Matrix Xu = filter_.transform(Xi);
  const int64_t H = Xu.d;

  EnsembleOutputs result;
  result.out_dim = out_dim;

  for (size_t gi = 0; gi < configs_.method_order.size(); ++gi) {
    const auto& members = configs_.members[gi];
    const Matrix& train_pp = pipelines_[gi]->train_transformed();
    const Matrix test_pp = pipelines_[gi]->transform(Xu);

    // Batches within this norm-method group (np.array_split semantics).
    const int64_t n_members = static_cast<int64_t>(members.size());
    const int64_t n_batches =
        (n_members + opts_.batch_size - 1) / opts_.batch_size;
    const auto sizes = array_split_sizes(n_members, n_batches);

    const bool use_cache = !caches_.empty();
    int64_t mi = 0;
    int64_t batch_idx = 0;
    for (int64_t bsize : sizes) {
      ForwardOptions fo;
      fo.train_size = train;
      fo.num_classes = n_classes_;
      fo.return_logits = true;
      fo.max_scratch_bytes = opts_.max_scratch_bytes;

      std::vector<float> out;
      if (clf && n_classes_ > c.max_classes) {
        // Hierarchical >max_classes path: members processed one at a time
        // (mixed-radix col embedding + class-tree ICL), mirroring Python's
        // per-table loop. Output = pseudo-logits (n_test, n_classes).
        out.resize(static_cast<size_t>(bsize * n_test * out_dim));
        for (int64_t b = 0; b < bsize; ++b) {
          const EnsembleMember& m = members[static_cast<size_t>(mi + b)];
          std::vector<float> Xs(static_cast<size_t>(T * H));
          std::vector<float> ys(static_cast<size_t>(train));
          for (int64_t t = 0; t < T; ++t) {
            const Matrix& src = t < train ? train_pp : test_pp;
            const int64_t r = t < train ? t : t - train;
            for (int64_t j = 0; j < H; ++j)
              Xs[static_cast<size_t>(t * H + j)] = static_cast<float>(
                  src.at(r, m.feature_shuffle[static_cast<size_t>(j)]));
          }
          for (int64_t t = 0; t < train; ++t)
            ys[static_cast<size_t>(t)] = static_cast<float>(
                m.class_shuffle[static_cast<size_t>(
                    static_cast<int64_t>(y_model_[static_cast<size_t>(t)]))]);
          const auto col = col_embedding_forward(model, *runner_, Xs.data(),
                                                 ys.data(), 1, T, H, train,
                                                 n_classes_);
          const auto reprs =
              row_interaction_forward(model, *runner_, col.data(), 1, T, H);
          const auto logits = hierarchical_member_logits(
              model, *runner_, reprs.data(), ys.data(), T, train, n_classes_);
          std::copy(logits.begin(), logits.end(),
                    out.begin() + b * n_test * out_dim);
        }
      } else if (use_cache) {
        // Cached path: only test rows go through the model.
        std::vector<float> Xs(static_cast<size_t>(bsize * n_test * H));
        for (int64_t b = 0; b < bsize; ++b) {
          const EnsembleMember& m = members[static_cast<size_t>(mi + b)];
          for (int64_t t = 0; t < n_test; ++t) {
            float* dst = &Xs[static_cast<size_t>((b * n_test + t) * H)];
            for (int64_t j = 0; j < H; ++j)
              dst[j] = static_cast<float>(
                  test_pp.at(t, m.feature_shuffle[static_cast<size_t>(j)]));
          }
        }
        out = tabicl_forward_cached(model, *runner_, Xs.data(), n_test,
                                    caches_[gi][static_cast<size_t>(batch_idx)], fo);
      } else {
        // Assemble (B, T, H) fp32 with per-member column permutation; train
        // rows precede test rows (fp64 -> fp32 cast at the model boundary).
        std::vector<float> Xs(static_cast<size_t>(bsize * T * H));
        std::vector<float> ys(static_cast<size_t>(bsize * train));
        for (int64_t b = 0; b < bsize; ++b) {
          const EnsembleMember& m = members[static_cast<size_t>(mi + b)];
          for (int64_t t = 0; t < T; ++t) {
            const Matrix& src = t < train ? train_pp : test_pp;
            const int64_t r = t < train ? t : t - train;
            float* dst = &Xs[static_cast<size_t>((b * T + t) * H)];
            for (int64_t j = 0; j < H; ++j)
              dst[j] = static_cast<float>(src.at(r, m.feature_shuffle[static_cast<size_t>(j)]));
          }
          for (int64_t t = 0; t < train; ++t) {
            const float yv = y_model_[static_cast<size_t>(t)];
            ys[static_cast<size_t>(b * train + t)] =
                clf ? static_cast<float>(
                          m.class_shuffle[static_cast<size_t>(static_cast<int64_t>(yv))])
                    : yv;
          }
        }
        out = tabicl_forward(model, *runner_, Xs.data(), ys.data(), bsize, T, H, fo);
      }
      // out: (bsize, n_test, out_dim)
      for (int64_t b = 0; b < bsize; ++b) {
        result.member_outputs.emplace_back(
            out.begin() + b * n_test * out_dim,
            out.begin() + (b + 1) * n_test * out_dim);
        result.class_shuffles.push_back(members[static_cast<size_t>(mi + b)].class_shuffle);
      }
      mi += bsize;
      batch_idx++;
    }
  }
  return result;
}

namespace {

std::vector<int32_t> to_i32(const std::vector<int64_t>& v) {
  return std::vector<int32_t>(v.begin(), v.end());
}
std::vector<int32_t> bools_to_i32(const std::vector<bool>& v) {
  std::vector<int32_t> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = v[i] ? 1 : 0;
  return out;
}

void save_cache(FittedWriter& w, const std::string& p, const TabICLCache& c) {
  w.put_u32(p + "mode", c.mode == TabICLCache::Mode::KV ? 0u : 1u);
  w.put_u64(p + "B", static_cast<uint64_t>(c.B));
  w.put_u64(p + "train", static_cast<uint64_t>(c.train_size));
  w.put_u64(p + "H", static_cast<uint64_t>(c.H));
  for (size_t i = 0; i < c.col_k.size(); ++i) {
    w.put_f32_tensor(p + "col_k." + std::to_string(i), c.col_k[i]);
    w.put_f32_tensor(p + "col_v." + std::to_string(i), c.col_v[i]);
  }
  w.put_u32(p + "n_col", static_cast<uint32_t>(c.col_k.size()));
  for (size_t i = 0; i < c.icl_k.size(); ++i) {
    w.put_f32_tensor(p + "icl_k." + std::to_string(i), c.icl_k[i]);
    w.put_f32_tensor(p + "icl_v." + std::to_string(i), c.icl_v[i]);
  }
  w.put_u32(p + "n_icl", static_cast<uint32_t>(c.icl_k.size()));
  if (c.mode == TabICLCache::Mode::Repr) w.put_f32_tensor(p + "repr", c.row_repr);
}

TabICLCache load_cache(const FittedReader& r, const std::string& p) {
  TabICLCache c;
  c.mode = r.get_u32(p + "mode") == 0 ? TabICLCache::Mode::KV
                                      : TabICLCache::Mode::Repr;
  c.B = static_cast<int64_t>(r.get_u64(p + "B"));
  c.train_size = static_cast<int64_t>(r.get_u64(p + "train"));
  c.H = static_cast<int64_t>(r.get_u64(p + "H"));
  const uint32_t n_col = r.get_u32(p + "n_col");
  for (uint32_t i = 0; i < n_col; ++i) {
    c.col_k.push_back(r.get_f32_tensor(p + "col_k." + std::to_string(i)));
    c.col_v.push_back(r.get_f32_tensor(p + "col_v." + std::to_string(i)));
  }
  const uint32_t n_icl = r.get_u32(p + "n_icl");
  for (uint32_t i = 0; i < n_icl; ++i) {
    c.icl_k.push_back(r.get_f32_tensor(p + "icl_k." + std::to_string(i)));
    c.icl_v.push_back(r.get_f32_tensor(p + "icl_v." + std::to_string(i)));
  }
  if (c.mode == TabICLCache::Mode::Repr) c.row_repr = r.get_f32_tensor(p + "repr");
  return c;
}

}  // namespace

std::string model_fingerprint(const Model& model) {
  // Cheap identity check for "same checkpoint": the decoder output projection
  // distinguishes task, head width, and any fine-tune.
  ggml_tensor* t = model.tensor("icl.decoder.2.weight");
  return sha256_hex(t->data, ggml_nbytes(t));
}

void EstimatorCore::save(FittedWriter& w) const {
  w.put_u64("core.d_in", static_cast<uint64_t>(d_in_));
  w.put_u64("core.n_classes", static_cast<uint64_t>(n_classes_));
  w.put_f32_tensor("core.y_model", y_model_);

  w.put_u32("opt.cache", static_cast<uint32_t>(opts_.cache));
  w.put_u32("opt.n_estimators", static_cast<uint32_t>(opts_.n_estimators));
  w.put_u32("opt.batch_size", static_cast<uint32_t>(opts_.batch_size));
  w.put_u64("opt.random_state", opts_.random_state);
  w.put_u32("opt.n_threads", static_cast<uint32_t>(opts_.n_threads));
  w.put_f64("opt.softmax_temperature", opts_.softmax_temperature);
  w.put_bool("opt.average_logits", opts_.average_logits);
  w.put_u64("opt.max_scratch_bytes", static_cast<uint64_t>(opts_.max_scratch_bytes));
  w.put_u32("opt.n_norm_methods", static_cast<uint32_t>(opts_.norm_methods.size()));
  for (size_t i = 0; i < opts_.norm_methods.size(); ++i)
    w.put_str("opt.norm_method." + std::to_string(i), opts_.norm_methods[i]);

  w.put_u64("imputer.d_in", static_cast<uint64_t>(imputer_.d_in()));
  w.put_f64_tensor("imputer.statistics", imputer_.statistics());
  w.put_i32_tensor("imputer.kept", to_i32(imputer_.kept_columns()));
  w.put_u64("filter.d_in", static_cast<uint64_t>(filter_.d_in()));
  w.put_i32_tensor("filter.keep", bools_to_i32(filter_.features_to_keep()));

  w.put_u32("configs.n_groups", static_cast<uint32_t>(configs_.method_order.size()));
  for (size_t gi = 0; gi < configs_.method_order.size(); ++gi) {
    const std::string p = "configs." + std::to_string(gi) + ".";
    w.put_str(p + "method", configs_.method_order[gi]);
    const auto& members = configs_.members[gi];
    w.put_u32(p + "n_members", static_cast<uint32_t>(members.size()));
    std::vector<int32_t> feat, cls;
    for (const auto& m : members) {
      feat.insert(feat.end(), m.feature_shuffle.begin(), m.feature_shuffle.end());
      cls.insert(cls.end(), m.class_shuffle.begin(), m.class_shuffle.end());
    }
    w.put_i32_tensor(p + "feat", feat);
    w.put_i32_tensor(p + "class", cls);
    pipelines_[gi]->save(w, "pipe." + std::to_string(gi) + ".");
  }

  w.put_u32("cache.n_groups", static_cast<uint32_t>(caches_.size()));
  for (size_t gi = 0; gi < caches_.size(); ++gi) {
    w.put_u32("cache." + std::to_string(gi) + ".n_batches",
              static_cast<uint32_t>(caches_[gi].size()));
    for (size_t bi = 0; bi < caches_[gi].size(); ++bi)
      save_cache(w, "cache." + std::to_string(gi) + "." + std::to_string(bi) + ".",
                 caches_[gi][bi]);
  }
}

void EstimatorCore::load(const FittedReader& r, int n_threads_override) {
  d_in_ = static_cast<int64_t>(r.get_u64("core.d_in"));
  n_classes_ = static_cast<int64_t>(r.get_u64("core.n_classes"));
  y_model_ = r.get_f32_tensor("core.y_model");

  opts_.cache = static_cast<CacheMode>(r.get_u32("opt.cache"));
  opts_.n_estimators = static_cast<int>(r.get_u32("opt.n_estimators"));
  opts_.batch_size = static_cast<int>(r.get_u32("opt.batch_size"));
  opts_.random_state = r.get_u64("opt.random_state");
  opts_.n_threads = n_threads_override >= 0
                        ? n_threads_override
                        : static_cast<int>(r.get_u32("opt.n_threads"));
  opts_.softmax_temperature = static_cast<float>(r.get_f64("opt.softmax_temperature"));
  opts_.average_logits = r.get_bool("opt.average_logits");
  opts_.max_scratch_bytes = static_cast<int64_t>(r.get_u64("opt.max_scratch_bytes"));
  opts_.norm_methods.clear();
  const uint32_t n_methods = r.get_u32("opt.n_norm_methods");
  for (uint32_t i = 0; i < n_methods; ++i)
    opts_.norm_methods.push_back(r.get_str("opt.norm_method." + std::to_string(i)));

  {
    std::vector<int64_t> kept;
    for (int32_t v : r.get_i32_tensor("imputer.kept")) kept.push_back(v);
    imputer_.restore(r.get_f64_tensor("imputer.statistics"), std::move(kept),
                     static_cast<int64_t>(r.get_u64("imputer.d_in")));
  }
  {
    std::vector<bool> keep;
    for (int32_t v : r.get_i32_tensor("filter.keep")) keep.push_back(v != 0);
    filter_.restore(std::move(keep), static_cast<int64_t>(r.get_u64("filter.d_in")));
  }

  configs_ = EnsembleConfigs{};
  pipelines_.clear();
  const uint32_t n_groups = r.get_u32("configs.n_groups");
  for (uint32_t gi = 0; gi < n_groups; ++gi) {
    const std::string p = "configs." + std::to_string(gi) + ".";
    const std::string method = r.get_str(p + "method");
    configs_.method_order.push_back(method);
    const uint32_t n_members = r.get_u32(p + "n_members");
    const auto feat = r.get_i32_tensor(p + "feat");
    const auto cls = r.get_i32_tensor(p + "class");
    const int64_t H = n_members > 0 ? static_cast<int64_t>(feat.size()) / n_members : 0;
    const int64_t C =
        n_members > 0 ? static_cast<int64_t>(cls.size()) / n_members : 0;
    std::vector<EnsembleMember> members(n_members);
    for (uint32_t m = 0; m < n_members; ++m) {
      members[m].feature_shuffle.assign(feat.begin() + m * H,
                                        feat.begin() + (m + 1) * H);
      if (C > 0)
        members[m].class_shuffle.assign(cls.begin() + m * C,
                                        cls.begin() + (m + 1) * C);
    }
    configs_.members.push_back(std::move(members));

    auto pipe = std::make_unique<PreprocessingPipeline>(method, 4.0,
                                                        opts_.random_state);
    pipe->load(r, "pipe." + std::to_string(gi) + ".");
    pipelines_.push_back(std::move(pipe));
  }

  caches_.clear();
  const uint32_t cache_groups = r.get_u32("cache.n_groups");
  caches_.resize(cache_groups);
  for (uint32_t gi = 0; gi < cache_groups; ++gi) {
    const uint32_t n_batches =
        r.get_u32("cache." + std::to_string(gi) + ".n_batches");
    for (uint32_t bi = 0; bi < n_batches; ++bi)
      caches_[gi].push_back(load_cache(
          r, "cache." + std::to_string(gi) + "." + std::to_string(bi) + "."));
  }

  runner_ = std::make_unique<GraphRunner>(opts_.n_threads);
}

}  // namespace tabicl
