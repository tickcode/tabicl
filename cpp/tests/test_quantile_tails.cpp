// Tail-extrapolation parity: QuantileDistribution.icdf goldens for both the
// exponential model (what TabICLRegressor uses) and the GPD model, across
// well-behaved, crossing, heavy-tailed and near-constant quantile rows.
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "doctest.h"
#include "quantile_dist.h"
#include "test_helpers.h"

using tabicl::TailType;
using tabicl::test::fixture_exists;
using tabicl::test::load_fixture;

TEST_CASE("quantile tails: icdf matches QuantileDistribution (exp and gpd)") {
  if (!fixture_exists("quantile_tails/alphas.npy")) {
    MESSAGE("quantile_tails fixtures missing; run tools/generate_fixtures.py");
    return;
  }
  auto alphas = load_fixture("quantile_tails/alphas.npy");
  const int64_t na = alphas.numel();

  const std::pair<const char*, TailType> kTails[] = {{"exp", TailType::Exp},
                                                     {"gpd", TailType::Gpd}};

  for (const std::string name : {"plain", "raw", "heavy", "flat"}) {
    CAPTURE(name);
    auto q = load_fixture("quantile_tails/" + name + "_q.npy");
    auto sorted_ref = load_fixture("quantile_tails/" + name + "_sorted.npy");
    const int64_t rows = q.shape[0], nq = q.shape[1];
    const auto alpha = tabicl::quantile_alpha_levels(nq);

    for (const auto& entry : kTails) {
      const std::string tail_name = entry.first;
      const TailType type = entry.second;
      auto ref = load_fixture("quantile_tails/" + name + "_" + tail_name + "_icdf.npy");
      REQUIRE(ref.shape[0] == rows);
      REQUIRE(ref.shape[1] == na);

      for (int64_t r = 0; r < rows; ++r) {
        std::vector<float> qs(q.f4() + r * nq, q.f4() + (r + 1) * nq);
        std::sort(qs.begin(), qs.end());
        // crossing_method="sort" must agree before icdf can be compared.
        for (int64_t i = 0; i < nq; ++i)
          REQUIRE_MESSAGE(qs[static_cast<size_t>(i)] == sorted_ref.f4()[r * nq + i],
                          name << " row " << r << ": sort diverged at " << i);

        const auto tails = tabicl::fit_quantile_tails(qs.data(), alpha, type);
        for (int64_t a = 0; a < na; ++a) {
          const float got = tabicl::quantile_icdf(
              qs.data(), alpha, tails, static_cast<float>(alphas.f8()[a]));
          const float want = ref.f4()[r * na + a];
          REQUIRE_MESSAGE(std::abs(got - want) <= 1e-4f + 1e-4f * std::abs(want),
                          name << "/" << tail_name << " row " << r << " alpha "
                               << alphas.f8()[a] << ": " << got << " vs " << want);
        }
      }
    }
  }
}

TEST_CASE("quantile tails: a grid too short to fit tails is rejected") {
  const auto alpha = tabicl::quantile_alpha_levels(4);
  const std::vector<float> q = {0.0f, 1.0f, 2.0f, 3.0f};
  CHECK_THROWS(tabicl::fit_quantile_tails(q.data(), alpha, TailType::Exp));
}
