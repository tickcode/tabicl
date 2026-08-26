// NumPy-faithful percentile / median / interp primitives (linear method),
// including numpy's _lerp t>=0.5 formula and np.interp's NaN fallback chain.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace tabicl {

// numpy _lerp(a, b, t): a + (b-a)*t, but b - (b-a)*(1-t) when t >= 0.5.
inline double np_lerp(double a, double b, double t) {
  const double diff = b - a;
  double r = a + diff * t;
  if (t >= 0.5) r = b - diff * (1.0 - t);
  return r;
}

// np.percentile(sorted_data, q100, method="linear") for pre-sorted data.
// q100 is the percentile in [0, 100]; internally divided by 100 (replicating
// numpy's true_divide round trip).
inline double np_percentile_sorted(const double* sorted, int64_t n, double q100) {
  const double q = q100 / 100.0;
  const double virt = q * static_cast<double>(n - 1);
  double prev_idx = std::floor(virt);
  if (prev_idx >= static_cast<double>(n - 1)) return sorted[n - 1];
  if (prev_idx < 0) prev_idx = 0;  // unreachable for q >= 0
  const int64_t j = static_cast<int64_t>(prev_idx);
  const double gamma = virt - prev_idx;
  return np_lerp(sorted[j], sorted[j + 1], gamma);
}

// np.median of pre-sorted data: middle element, or mean of the two middles.
inline double np_median_sorted(const double* sorted, int64_t n) {
  if (n == 0) return std::numeric_limits<double>::quiet_NaN();
  if (n % 2 == 1) return sorted[n / 2];
  return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;
}

// Extract non-NaN values of column j from row-major (n, d) data, sorted.
inline std::vector<double> sorted_finite_column(const double* X, int64_t n,
                                                int64_t d, int64_t j) {
  std::vector<double> col;
  col.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const double v = X[i * d + j];
    if (!std::isnan(v)) col.push_back(v);
  }
  std::sort(col.begin(), col.end());
  return col;
}

// np.interp(v, xp, fp) for ascending xp: numpy C semantics (arr_interp),
// including the NaN fallback chain. Assumes len >= 1.
inline double np_interp(double v, const double* xp, const double* fp, int64_t len) {
  if (std::isnan(v)) return v;
  if (len == 1) return fp[0];
  if (v > xp[len - 1]) return fp[len - 1];
  if (v < xp[0]) return fp[0];
  // j: largest index with xp[j] <= v
  const double* it = std::upper_bound(xp, xp + len, v);
  int64_t j = static_cast<int64_t>(it - xp) - 1;
  if (j < 0) j = 0;  // v == xp[0]
  if (j == len - 1) return fp[len - 1];
  if (v == xp[j]) return fp[j];
  const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
  double res = slope * (v - xp[j]) + fp[j];
  if (std::isnan(res)) {
    res = slope * (v - xp[j + 1]) + fp[j + 1];
    if (std::isnan(res) && fp[j] == fp[j + 1]) res = fp[j];
  }
  return res;
}

// np.linspace(0, 1, num, endpoint=True): arange(num) * step with the last
// element pinned to 1.0 (numpy's exact construction).
inline std::vector<double> np_linspace01(int64_t num) {
  std::vector<double> out(static_cast<size_t>(num));
  if (num == 1) {
    out[0] = 0.0;
    return out;
  }
  const double step = 1.0 / static_cast<double>(num - 1);
  for (int64_t i = 0; i < num; ++i) out[static_cast<size_t>(i)] = static_cast<double>(i) * step;
  out[static_cast<size_t>(num - 1)] = 1.0;
  return out;
}

}  // namespace tabicl
