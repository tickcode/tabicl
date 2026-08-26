// Exact validation of the Shuffler and EnsembleGenerator config ports
// against golden fixtures (sections shuffler / ensemble_configs).
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "doctest.h"
#include "preprocess/ensemble.h"
#include "test_helpers.h"

using tabicl::EnsembleMember;
using tabicl::generate_ensemble_configs;
using tabicl::parse_shuffle_method;
using tabicl::shuffler_patterns;
using tabicl::test::fixture_exists;
using tabicl::test::load_fixture;

TEST_CASE("shuffler: latin/random/shift patterns match goldens") {
  if (!fixture_exists("shuffler/latin_n5_s0.npy")) {
    MESSAGE("shuffler fixtures missing; run tools/generate_fixtures.py");
    return;
  }
  const std::vector<uint64_t> seeds = {0, 42, 12345};
  for (size_t si = 0; si < seeds.size(); ++si) {
    for (int n : {1, 2, 3, 5, 8, 50}) {
      for (const std::string method : {"latin", "random", "shift"}) {
        auto g = load_fixture("shuffler/" + method + "_n" + std::to_string(n) + "_s" +
                              std::to_string(si) + ".npy");
        auto got = shuffler_patterns(n, parse_shuffle_method(method), 8, seeds[si]);
        REQUIRE_MESSAGE(static_cast<int64_t>(got.size()) == g.shape[0],
                        method << " n=" << n << " seed " << seeds[si]
                               << ": pattern count mismatch");
        for (int64_t r = 0; r < g.shape[0]; ++r)
          for (int64_t c = 0; c < g.shape[1]; ++c)
            REQUIRE_MESSAGE(
                got[static_cast<size_t>(r)][static_cast<size_t>(c)] ==
                    g.i8()[r * g.shape[1] + c],
                method << " n=" << n << " seed " << seeds[si] << " diverged at ("
                       << r << "," << c << ")");
      }
    }
    {
      auto g = load_fixture("shuffler/latin_fallback_n4001_s" + std::to_string(si) + ".npy");
      auto got = shuffler_patterns(4001, tabicl::ShuffleMethod::Latin, 3, seeds[si]);
      REQUIRE(static_cast<int64_t>(got.size()) == g.shape[0]);
      for (int64_t r = 0; r < g.shape[0]; ++r)
        for (int64_t c = 0; c < g.shape[1]; ++c)
          REQUIRE(got[static_cast<size_t>(r)][static_cast<size_t>(c)] ==
                  g.i8()[r * g.shape[1] + c]);
    }
  }
}

namespace {

struct Case {
  std::string name;
  int64_t d;
  int64_t n_classes;  // 0 = regression
  int n_estimators;
  std::vector<std::string> norm_methods;
  uint64_t random_state;
};

// Rebuild Python's per-method member lists from fixture rows + method indices,
// then compare with the C++ groups (order within a method must be identical;
// group order is canonicalized differently by design — see ensemble.h).
void check_case(const Case& tc) {
  auto feats = load_fixture("ensemble_configs/" + tc.name + "_feat_shuffles.npy");
  auto midx = load_fixture("ensemble_configs/" + tc.name + "_member_method_idx.npy");
  const bool classification = tc.n_classes > 0;
  tabicl::npy::Array classes;
  if (classification)
    classes = load_fixture("ensemble_configs/" + tc.name + "_class_shuffles.npy");

  auto cfg = generate_ensemble_configs(tc.d, tc.n_classes, tc.n_estimators,
                                       tc.norm_methods, "latin", "shift",
                                       tc.random_state);
  REQUIRE(cfg.total_members() == feats.shape[0]);

  // Expected per-method lists in fixture row order.
  std::map<std::string, std::vector<EnsembleMember>> expected;
  for (int64_t r = 0; r < feats.shape[0]; ++r) {
    const auto& method = tc.norm_methods[static_cast<size_t>(midx.i8()[r])];
    EnsembleMember m;
    for (int64_t c = 0; c < feats.shape[1]; ++c)
      m.feature_shuffle.push_back(static_cast<int32_t>(feats.i8()[r * feats.shape[1] + c]));
    if (classification)
      for (int64_t c = 0; c < classes.shape[1]; ++c)
        m.class_shuffle.push_back(
            static_cast<int32_t>(classes.i8()[r * classes.shape[1] + c]));
    expected[method].push_back(std::move(m));
  }

  REQUIRE(cfg.method_order.size() == expected.size());
  for (size_t gi = 0; gi < cfg.method_order.size(); ++gi) {
    const auto& method = cfg.method_order[gi];
    REQUIRE_MESSAGE(expected.count(method), tc.name << ": unexpected method " << method);
    const auto& exp = expected[method];
    const auto& got = cfg.members[gi];
    REQUIRE_MESSAGE(got.size() == exp.size(), tc.name << "/" << method << ": member count");
    for (size_t i = 0; i < got.size(); ++i) {
      REQUIRE_MESSAGE(got[i].feature_shuffle == exp[i].feature_shuffle,
                      tc.name << "/" << method << " member " << i << ": feat perm");
      REQUIRE_MESSAGE(got[i].class_shuffle == exp[i].class_shuffle,
                      tc.name << "/" << method << " member " << i << ": class perm");
    }
  }
}

}  // namespace

TEST_CASE("ensemble: member configs match goldens per normalization method") {
  if (!fixture_exists("ensemble_configs/defaults_feat_shuffles.npy")) {
    MESSAGE("ensemble_configs fixtures missing; run tools/generate_fixtures.py");
    return;
  }
  check_case({"defaults", 5, 3, 8, {"none", "power"}, 42});
  check_case({"one_est", 5, 3, 1, {"none", "power"}, 42});
  check_case({"reg", 7, 0, 8, {"none", "power"}, 0});
  check_case({"all_norms", 6, 4, 16,
              {"none", "power", "robust", "quantile", "quantile_rtdl"}, 7});
}
