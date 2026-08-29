// Port of tabicl's QuantileDistribution (_model/quantile_dist.py): the knot
// grid, the piecewise-linear spline between knots, and the tail extrapolation
// used outside it.
//
// The head emits `num_quantiles` values on a fixed alpha grid
// (0.001 .. 0.999 for the 999-quantile v2 head). Probability levels outside
// that grid are extrapolated from the `TAIL_QUANTILES_FOR_ESTIMATION` most
// extreme knots, either exponentially (the model default) or with a
// Generalized Pareto tail.
//
// Everything is computed in fp32 to match the torch dtype ladder; reductions
// accumulate in double and round once, as elsewhere in the regressor path.
#pragma once

#include <cstdint>
#include <vector>

namespace tabicl {

enum class TailType {
  Exp,  // Q(a) = beta*log(a) + c        — QuantileToDistribution's default
  Gpd,  // Generalized Pareto, shape eta estimated Pickands-style
};

// torch.linspace(0, 1, num_quantiles + 2)[1:-1] in fp32.
std::vector<float> quantile_alpha_levels(int64_t num_quantiles);

// Tail parameters fitted from one row of ASCENDING quantiles.
struct QuantileTails {
  TailType type = TailType::Exp;
  float alpha_l = 0.0f, alpha_r = 0.0f;  // grid endpoints
  float q_l = 0.0f, q_r = 0.0f;          // quantiles at those endpoints
  // Exp: Q(a) = a_l*log(a) + b_l (left), a_r*log(1-a) + b_r (right).
  float a_l = 0.0f, b_l = 0.0f, a_r = 0.0f, b_r = 0.0f;
  // Gpd: shape eta and scale mu per side.
  float eta_l = 0.0f, mu_l = 0.0f, eta_r = 0.0f, mu_r = 0.0f;
};

// `q_sorted` must be ascending with `alpha.size()` entries (the caller sorts,
// mirroring crossing_method="sort"). Throws if the grid is too short to
// estimate tails from.
QuantileTails fit_quantile_tails(const float* q_sorted,
                                 const std::vector<float>& alpha, TailType type);

// QuantileDistribution.icdf at one probability level. `a` outside (0, 1) is
// floored/capped exactly as the Python clamps do, so no level is rejected.
float quantile_icdf(const float* q_sorted, const std::vector<float>& alpha,
                    const QuantileTails& tails, float a);

}  // namespace tabicl
