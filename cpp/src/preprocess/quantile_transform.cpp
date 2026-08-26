#include "preprocess/quantile_transform.h"

#include <algorithm>
#include <cmath>

#include "preprocess/cephes_ndtri.h"
#include "preprocess/quantile_utils.h"
#include "preprocess/rng_numpy.h"
#include "preprocess/stats.h"

namespace tabicl {

namespace {
constexpr double kBoundsThreshold = 1e-7;
constexpr double kSpacing1 = 2.220446049250313e-16;  // np.spacing(1)
}  // namespace

void QuantileTransformerNormal::fit(const Matrix& X, int64_t n_quantiles,
                                    int64_t subsample, uint32_t random_state) {
  const int64_t n = X.n;
  d_ = X.d;
  n_quantiles_ = std::max<int64_t>(1, std::min(n_quantiles, n));
  references_ = np_linspace01(n_quantiles_);

  // Subsample without replacement (sklearn resample: shuffle + head).
  const Matrix* src = &X;
  Matrix sub;
  if (subsample < n) {
    NpRandomState rs(random_state);
    std::vector<int64_t> idx(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) idx[static_cast<size_t>(i)] = i;
    rs.shuffle(idx);
    sub = Matrix::zeros(subsample, X.d);
    for (int64_t i = 0; i < subsample; ++i)
      for (int64_t j = 0; j < X.d; ++j)
        sub.at(i, j) = X.at(idx[static_cast<size_t>(i)], j);
    src = &sub;
  }

  // quantiles_ = np.nanpercentile(X, references*100, axis=0), then
  // np.maximum.accumulate along the quantile axis.
  quantiles_.assign(static_cast<size_t>(n_quantiles_ * d_), 0.0);
  for (int64_t j = 0; j < d_; ++j) {
    const auto col = sorted_finite_column(src->data.data(), src->n, src->d, j);
    const int64_t m = static_cast<int64_t>(col.size());
    for (int64_t q = 0; q < n_quantiles_; ++q) {
      const double q100 = references_[static_cast<size_t>(q)] * 100.0;
      quantiles_[static_cast<size_t>(q * d_ + j)] =
          m > 0 ? np_percentile_sorted(col.data(), m, q100)
                : std::numeric_limits<double>::quiet_NaN();
    }
    for (int64_t q = 1; q < n_quantiles_; ++q) {
      double& cur = quantiles_[static_cast<size_t>(q * d_ + j)];
      const double prev = quantiles_[static_cast<size_t>((q - 1) * d_ + j)];
      // np.maximum.accumulate: NaN propagates once encountered.
      if (std::isnan(prev) || cur < prev) cur = prev;
    }
  }
}

Matrix QuantileTransformerNormal::transform(const Matrix& X) const {
  Matrix out = X;
  std::vector<double> qcol(static_cast<size_t>(n_quantiles_));
  std::vector<double> neg_q(static_cast<size_t>(n_quantiles_));
  std::vector<double> neg_ref(static_cast<size_t>(n_quantiles_));
  const double clip_min = ndtri(kBoundsThreshold - kSpacing1);
  const double clip_max = ndtri(1.0 - (kBoundsThreshold - kSpacing1));

  for (int64_t j = 0; j < X.d; ++j) {
    for (int64_t q = 0; q < n_quantiles_; ++q) {
      qcol[static_cast<size_t>(q)] = quantiles_[static_cast<size_t>(q * d_ + j)];
      neg_q[static_cast<size_t>(n_quantiles_ - 1 - q)] = -qcol[static_cast<size_t>(q)];
      neg_ref[static_cast<size_t>(n_quantiles_ - 1 - q)] =
          -references_[static_cast<size_t>(q)];
    }
    const double lower_bound_x = qcol[0];
    const double upper_bound_x = qcol[static_cast<size_t>(n_quantiles_ - 1)];
    for (int64_t i = 0; i < X.n; ++i) {
      const double v = X.at(i, j);
      if (std::isnan(v)) continue;  // NaN passes through
      const bool lower_idx = (v - kBoundsThreshold) < lower_bound_x;
      const bool upper_idx = (v + kBoundsThreshold) > upper_bound_x;
      double t = 0.5 * (np_interp(v, qcol.data(), references_.data(), n_quantiles_) -
                        np_interp(-v, neg_q.data(), neg_ref.data(), n_quantiles_));
      if (upper_idx) t = 1.0;
      if (lower_idx) t = 0.0;
      double z = ndtri(t);
      z = std::clamp(z, clip_min, clip_max);
      out.at(i, j) = z;
    }
  }
  return out;
}

void RTDLQuantileTransformer::fit(const Matrix& X, uint64_t random_state) {
  // n_quantiles = max(min(n // 30, 1000), 10)
  const int64_t nq = std::max<int64_t>(std::min<int64_t>(X.n / 30, 1000), 10);

  // _add_noise: stds = np.std(X, axis=0) (per-column pairwise; ddof 0);
  // noise_std = 1e-3 / max(std, 1e-3); X_noisy = X + noise_std * normal(shape)
  // with draws in C-order (row-major), PCG64(seed).
  std::vector<double> stds(static_cast<size_t>(X.d));
  std_axis0_colpw(X.data.data(), X.n, X.d, /*ddof=*/0, stds.data());
  std::vector<double> noise_std(static_cast<size_t>(X.d));
  for (int64_t j = 0; j < X.d; ++j)
    noise_std[static_cast<size_t>(j)] = 1e-3 / std::max(stds[static_cast<size_t>(j)], 1e-3);

  NpPCG64 gen(random_state);
  Matrix noisy = X;
  for (int64_t i = 0; i < X.n; ++i)
    for (int64_t j = 0; j < X.d; ++j)
      noisy.at(i, j) = X.at(i, j) + noise_std[static_cast<size_t>(j)] * gen.standard_normal();

  qt_.fit(noisy, nq, /*subsample=*/1000000000LL,
          static_cast<uint32_t>(random_state & 0xFFFFFFFFull));
  // Pipeline: quantile transform (no noise at transform time) -> StandardScaler.
  scaler_.fit(qt_.transform(X));
}

Matrix RTDLQuantileTransformer::transform(const Matrix& X) const {
  return scaler_.transform(qt_.transform(X));
}

}  // namespace tabicl
