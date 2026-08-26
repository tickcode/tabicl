#include "tabicl/classifier.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "estimator_core.h"

namespace tabicl {

TabICLClassifier::TabICLClassifier(std::shared_ptr<Model> model)
    : TabICLClassifier(std::move(model), EstimatorOptions{}) {}

TabICLClassifier::TabICLClassifier(std::shared_ptr<Model> model,
                                   const EstimatorOptions& opts)
    : model_(std::move(model)), core_(std::make_unique<EstimatorCore>()) {
  if (model_->config().task != Task::Classification)
    throw std::runtime_error("TabICLClassifier: model is not a classifier");
  opts_ = std::make_unique<EstimatorOptions>(opts);
}

TabICLClassifier::~TabICLClassifier() = default;

void TabICLClassifier::fit(const double* X, const double* y, int64_t n, int64_t d) {
  // LabelEncoder: classes_ = sorted unique labels; y -> contiguous indices.
  classes_.assign(y, y + n);
  std::sort(classes_.begin(), classes_.end());
  classes_.erase(std::unique(classes_.begin(), classes_.end()), classes_.end());
  for (double v : classes_)
    if (std::isnan(v)) throw std::runtime_error("fit: NaN labels not supported");
  const int64_t n_classes = static_cast<int64_t>(classes_.size());
  if (n_classes < 2) throw std::runtime_error("fit: need at least 2 classes");

  std::vector<float> y_idx(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const auto it = std::lower_bound(classes_.begin(), classes_.end(), y[i]);
    y_idx[static_cast<size_t>(i)] =
        static_cast<float>(it - classes_.begin());
  }
  core_->fit(X, n, d, std::move(y_idx), n_classes, *opts_, model_.get());
}

void TabICLClassifier::predict_proba(const double* X, int64_t n_test,
                                     float* out) const {
  const int64_t C = n_classes();
  auto res = core_->predict_outputs(*model_, X, n_test);
  const int64_t n_members = static_cast<int64_t>(res.member_outputs.size());

  // avg += out[..., class_shuffle], fp32, member order; /= n; softmax(/tau);
  // renormalize rows.
  std::vector<float> avg(static_cast<size_t>(n_test * C), 0.0f);
  for (int64_t m = 0; m < n_members; ++m) {
    const auto& o = res.member_outputs[static_cast<size_t>(m)];
    const auto& shuffle = res.class_shuffles[static_cast<size_t>(m)];
    for (int64_t r = 0; r < n_test; ++r)
      for (int64_t k = 0; k < C; ++k)
        avg[static_cast<size_t>(r * C + k)] +=
            o[static_cast<size_t>(r * C + shuffle[static_cast<size_t>(k)])];
  }
  const float inv = 1.0f / static_cast<float>(n_members);
  for (float& v : avg) v *= inv;

  const auto& opts = core_->options();
  if (opts.average_logits) {
    const float tau = opts.softmax_temperature;
    for (int64_t r = 0; r < n_test; ++r) {
      float* row = &avg[static_cast<size_t>(r * C)];
      float mx = -std::numeric_limits<float>::infinity();
      for (int64_t k = 0; k < C; ++k) mx = std::max(mx, row[k] / tau);
      float denom = 0.0f;
      for (int64_t k = 0; k < C; ++k) {
        row[k] = std::exp(row[k] / tau - mx);
        denom += row[k];
      }
      for (int64_t k = 0; k < C; ++k) row[k] /= denom;
    }
  }
  for (int64_t r = 0; r < n_test; ++r) {
    float* row = &avg[static_cast<size_t>(r * C)];
    float s = 0.0f;
    for (int64_t k = 0; k < C; ++k) s += row[k];
    for (int64_t k = 0; k < C; ++k) row[k] /= s;
  }
  std::copy(avg.begin(), avg.end(), out);
}

void TabICLClassifier::predict(const double* X, int64_t n_test,
                               double* out_labels) const {
  const int64_t C = n_classes();
  std::vector<float> proba(static_cast<size_t>(n_test * C));
  predict_proba(X, n_test, proba.data());
  for (int64_t r = 0; r < n_test; ++r) {
    int64_t best = 0;
    for (int64_t k = 1; k < C; ++k)
      if (proba[static_cast<size_t>(r * C + k)] >
          proba[static_cast<size_t>(r * C + best)])
        best = k;  // strict >: ties resolve to the lowest index (np.argmax)
    out_labels[r] = classes_[static_cast<size_t>(best)];
  }
}

}  // namespace tabicl
