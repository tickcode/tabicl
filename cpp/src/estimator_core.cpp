#include "estimator_core.h"

#include <cmath>
#include <stdexcept>

#include "hierarchical.h"
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
                                                 ys.data(), bsize, train, H, mode));
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

}  // namespace tabicl
