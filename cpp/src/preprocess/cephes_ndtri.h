// scipy.stats.norm.ppf (Cephes ndtri), ULP-exact vs scipy.
#pragma once

namespace tabicl {

// Inverse of the standard normal CDF. ndtri(0) = -inf, ndtri(1) = +inf,
// outside [0, 1] = NaN.
double ndtri(double y0);

}  // namespace tabicl
