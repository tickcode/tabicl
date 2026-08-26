// Reductions matching NumPy's accumulation order bit-for-bit:
//  - 2D C-order axis-0 reductions accumulate naively row by row
//    (numpy's add.reduce inner loop runs over the contiguous axis).
//  - 1D reductions use numpy's pairwise summation (block 128, 8-way unroll).
// Verified empirically against numpy 2.4 (see M3 notes in the plan).
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace tabicl {

// numpy pairwise_sum for a contiguous 1D array.
inline double pairwise_sum(const double* a, int64_t n) {
  if (n < 8) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += a[i];
    return s;
  }
  if (n <= 128) {
    double r[8];
    for (int j = 0; j < 8; ++j) r[j] = a[j];
    int64_t i = 8;
    for (; i < n - (n % 8); i += 8)
      for (int j = 0; j < 8; ++j) r[j] += a[i + j];
    double s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) s += a[i];
    return s;
  }
  int64_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
}

// sum over rows (axis 0) of C-order (n, d): out[j] = sum_i X[i*d + j],
// accumulated in row order (numpy 2D axis-0 semantics).
inline void sum_axis0(const double* X, int64_t n, int64_t d, double* out) {
  for (int64_t j = 0; j < d; ++j) out[j] = 0.0;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < d; ++j) out[j] += X[i * d + j];
}

inline void mean_axis0(const double* X, int64_t n, int64_t d, double* out) {
  sum_axis0(X, n, d, out);
  for (int64_t j = 0; j < d; ++j) out[j] /= static_cast<double>(n);
}

// np.std(X, axis=0, ddof): two-pass, squared deviations accumulated row-wise.
inline void std_axis0(const double* X, int64_t n, int64_t d, int ddof, double* out) {
  std::vector<double> mean(static_cast<size_t>(d));
  mean_axis0(X, n, d, mean.data());
  for (int64_t j = 0; j < d; ++j) out[j] = 0.0;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < d; ++j) {
      const double dev = X[i * d + j] - mean[j];
      out[j] += dev * dev;
    }
  for (int64_t j = 0; j < d; ++j)
    out[j] = std::sqrt(out[j] / static_cast<double>(n - ddof));
}

// np.nanmean(X, axis=0): NaNs contribute 0 to the sum and are excluded from
// the count. An all-NaN column yields NaN (numpy warns; we just produce NaN).
inline void nanmean_axis0(const double* X, int64_t n, int64_t d, double* out,
                          int64_t* counts = nullptr) {
  std::vector<int64_t> cnt(static_cast<size_t>(d), 0);
  for (int64_t j = 0; j < d; ++j) out[j] = 0.0;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < d; ++j) {
      const double v = X[i * d + j];
      if (!std::isnan(v)) {
        out[j] += v;
        cnt[static_cast<size_t>(j)]++;
      }
    }
  for (int64_t j = 0; j < d; ++j) {
    out[j] = cnt[static_cast<size_t>(j)] > 0
                 ? out[j] / static_cast<double>(cnt[static_cast<size_t>(j)])
                 : std::numeric_limits<double>::quiet_NaN();
    if (counts) counts[j] = cnt[static_cast<size_t>(j)];
  }
}

// np.nanstd(X, axis=0, ddof).
inline void nanstd_axis0(const double* X, int64_t n, int64_t d, int ddof, double* out) {
  std::vector<double> mean(static_cast<size_t>(d));
  std::vector<int64_t> cnt(static_cast<size_t>(d));
  nanmean_axis0(X, n, d, mean.data(), cnt.data());
  for (int64_t j = 0; j < d; ++j) out[j] = 0.0;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < d; ++j) {
      const double v = X[i * d + j];
      if (!std::isnan(v)) {
        const double dev = v - mean[j];
        out[j] += dev * dev;
      }
    }
  for (int64_t j = 0; j < d; ++j) {
    const int64_t den = cnt[static_cast<size_t>(j)] - ddof;
    // den <= 0 (all-NaN column, or single value with ddof=1) is NaN in numpy.
    out[j] = den > 0 ? std::sqrt(out[j] / static_cast<double>(den))
                     : std::numeric_limits<double>::quiet_NaN();
  }
}

// --- Column-pairwise variants -----------------------------------------------
// In the Python pipeline, everything downstream of UniqueFeatureFilter is
// F-contiguous (boolean-mask column indexing returns F-order), so numpy's
// axis-0 reductions there are contiguous per-column PAIRWISE sums. The
// nan-variants replace NaN with 0.0 in place (numpy's _replace_nan), keeping
// the zero entries inside the pairwise tree.

inline void mean_axis0_colpw(const double* X, int64_t n, int64_t d, double* out) {
  std::vector<double> col(static_cast<size_t>(n));
  for (int64_t j = 0; j < d; ++j) {
    for (int64_t i = 0; i < n; ++i) col[static_cast<size_t>(i)] = X[i * d + j];
    out[j] = pairwise_sum(col.data(), n) / static_cast<double>(n);
  }
}

inline void std_axis0_colpw(const double* X, int64_t n, int64_t d, int ddof, double* out) {
  std::vector<double> col(static_cast<size_t>(n));
  for (int64_t j = 0; j < d; ++j) {
    for (int64_t i = 0; i < n; ++i) col[static_cast<size_t>(i)] = X[i * d + j];
    const double mean = pairwise_sum(col.data(), n) / static_cast<double>(n);
    for (int64_t i = 0; i < n; ++i) {
      const double dev = col[static_cast<size_t>(i)] - mean;
      col[static_cast<size_t>(i)] = dev * dev;
    }
    out[j] = std::sqrt(pairwise_sum(col.data(), n) / static_cast<double>(n - ddof));
  }
}

inline void nanmean_axis0_colpw(const double* X, int64_t n, int64_t d, double* out,
                                int64_t* counts = nullptr) {
  std::vector<double> col(static_cast<size_t>(n));
  for (int64_t j = 0; j < d; ++j) {
    int64_t cnt = 0;
    for (int64_t i = 0; i < n; ++i) {
      const double v = X[i * d + j];
      const bool ok = !std::isnan(v);
      col[static_cast<size_t>(i)] = ok ? v : 0.0;
      cnt += ok ? 1 : 0;
    }
    out[j] = cnt > 0 ? pairwise_sum(col.data(), n) / static_cast<double>(cnt)
                     : std::numeric_limits<double>::quiet_NaN();
    if (counts) counts[j] = cnt;
  }
}

inline void nanstd_axis0_colpw(const double* X, int64_t n, int64_t d, int ddof,
                               double* out) {
  std::vector<double> col(static_cast<size_t>(n));
  for (int64_t j = 0; j < d; ++j) {
    int64_t cnt = 0;
    for (int64_t i = 0; i < n; ++i) {
      const double v = X[i * d + j];
      const bool ok = !std::isnan(v);
      col[static_cast<size_t>(i)] = ok ? v : 0.0;
      cnt += ok ? 1 : 0;
    }
    if (cnt == 0) {
      out[j] = std::numeric_limits<double>::quiet_NaN();
      continue;
    }
    const double mean = pairwise_sum(col.data(), n) / static_cast<double>(cnt);
    for (int64_t i = 0; i < n; ++i) {
      const double v = X[i * d + j];
      const double dev = std::isnan(v) ? 0.0 : v - mean;
      col[static_cast<size_t>(i)] = dev * dev;
    }
    const int64_t den = cnt - ddof;
    out[j] = den > 0 ? std::sqrt(pairwise_sum(col.data(), n) / static_cast<double>(den))
                     : std::numeric_limits<double>::quiet_NaN();
  }
}

}  // namespace tabicl
