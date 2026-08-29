// Fixed seed + fixed context must produce bitwise-identical regressor output,
// independent of thread count and of how many times the estimator is refit.
// The classifier equivalent lives in tests/test_cpp_parity.py; this asserts the
// same property for the regressor, which downstream consumers rely on for
// deterministic replay.
//
// Scope note: this deliberately does not assert bitwise equality ACROSS cache
// modes. Cached and uncached paths reorder fp32 accumulation, so test_cache.cpp
// compares those at 1e-4 -- the guarantee here is per-configuration.
#include <cstring>
#include <vector>

#include "doctest.h"
#include "tabicl/model.h"
#include "tabicl/options.h"
#include "tabicl/regressor.h"
#include "test_helpers.h"

using tabicl::EstimatorOptions;
using tabicl::test::fixture_exists;
using tabicl::test::fixture_path;
using tabicl::test::load_fixture;

namespace {

int64_t bitwise_mismatches(const std::vector<double>& a,
                           const std::vector<double>& b) {
  REQUIRE(a.size() == b.size());
  int64_t bad = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0) bad++;
  return bad;
}

}  // namespace

TEST_CASE("determinism: regressor is bitwise thread-invariant and repeatable") {
  if (!fixture_exists("e2e/reg_small_Xtr.npy") ||
      !fixture_exists("tabicl-regressor-v2.gguf")) {
    MESSAGE("fixtures missing");
    return;
  }
  auto model = tabicl::Model::load(fixture_path("tabicl-regressor-v2.gguf"));
  auto Xtr = load_fixture("e2e/reg_small_Xtr.npy");
  auto ytr = load_fixture("e2e/reg_small_ytr.npy");
  auto Xte = load_fixture("e2e/reg_small_Xte.npy");
  const int64_t n_test = Xte.shape[0];
  // Spans the spline interior and both extrapolated tails.
  const double alphas[5] = {1e-4, 0.1, 0.5, 0.9, 0.9999};
  const int64_t n_alphas = 5;

  std::vector<double> mean_ref, q_ref;
  // [1, 8, 8]: the third pass repeats the second, covering thread-count
  // invariance and run-to-run repeatability in one loop.
  for (int n_threads : {1, 8, 8}) {
    CAPTURE(n_threads);
    EstimatorOptions opts;
    opts.n_threads = n_threads;
    opts.random_state = 42;
    tabicl::TabICLRegressor reg(model, opts);
    reg.fit(Xtr.f8(), ytr.f8(), Xtr.shape[0], Xtr.shape[1]);

    std::vector<double> mean(static_cast<size_t>(n_test));
    std::vector<double> q(static_cast<size_t>(n_test * n_alphas));
    reg.predict(Xte.f8(), n_test, mean.data());
    reg.predict_quantiles(Xte.f8(), n_test, alphas, n_alphas, q.data());

    if (mean_ref.empty()) {
      mean_ref = std::move(mean);
      q_ref = std::move(q);
    } else {
      CHECK_MESSAGE(bitwise_mismatches(mean, mean_ref) == 0,
                    "predict() diverged at n_threads=" << n_threads);
      CHECK_MESSAGE(bitwise_mismatches(q, q_ref) == 0,
                    "predict_quantiles() diverged at n_threads=" << n_threads);
    }
  }
}
