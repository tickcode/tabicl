// KV-cache correctness: cached (kv and repr modes) and uncached predictions
// must agree — test rows never enter any K/V set, so the math is identical
// up to fp32 op-reordering (repo precedent tolerance 1e-4; observed ~1e-6).
#include <cmath>
#include <vector>

#include "doctest.h"
#include "estimator_core.h"
#include "tabicl/classifier.h"
#include "tabicl/regressor.h"
#include "test_helpers.h"

using tabicl::CacheMode;
using tabicl::EstimatorOptions;
using tabicl::test::fixture_exists;
using tabicl::test::fixture_path;
using tabicl::test::load_fixture;

TEST_CASE("cache: classifier cached == uncached (kv and repr modes)") {
  if (!fixture_exists("e2e/clf_small_Xtr.npy") ||
      !fixture_exists("tabicl-classifier-v2.gguf")) {
    MESSAGE("fixtures missing");
    return;
  }
  auto model = tabicl::Model::load(fixture_path("tabicl-classifier-v2.gguf"));
  auto Xtr = load_fixture("e2e/clf_small_Xtr.npy");
  auto ytr = load_fixture("e2e/clf_small_ytr.npy");
  auto Xte = load_fixture("e2e/clf_small_Xte.npy");
  const int64_t n_test = Xte.shape[0];

  std::vector<std::vector<float>> probas;
  for (CacheMode mode : {CacheMode::None, CacheMode::KV, CacheMode::Repr}) {
    EstimatorOptions opts;
    opts.cache = mode;
    opts.n_threads = 8;
    tabicl::TabICLClassifier clf(model, opts);
    clf.fit(Xtr.f8(), ytr.f8(), Xtr.shape[0], Xtr.shape[1]);
    std::vector<float> p(static_cast<size_t>(n_test * clf.n_classes()));
    clf.predict_proba(Xte.f8(), n_test, p.data());
    probas.push_back(std::move(p));
  }
  for (size_t m = 1; m < probas.size(); ++m) {
    double max_err = 0;
    for (size_t i = 0; i < probas[0].size(); ++i)
      max_err = std::max(max_err,
                         std::abs(static_cast<double>(probas[m][i]) - probas[0][i]));
    CHECK_MESSAGE(max_err <= 1e-4,
                  "cache mode " << m << " diverges from uncached: " << max_err);
    MESSAGE("cache mode " << m << " max |diff| vs uncached: " << max_err);
  }
}

TEST_CASE("cache: regressor cached == uncached (kv and repr modes)") {
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

  std::vector<std::vector<double>> means;
  double y_scale = 0;
  for (CacheMode mode : {CacheMode::None, CacheMode::KV, CacheMode::Repr}) {
    EstimatorOptions opts;
    opts.cache = mode;
    opts.n_threads = 8;
    tabicl::TabICLRegressor reg(model, opts);
    reg.fit(Xtr.f8(), ytr.f8(), Xtr.shape[0], Xtr.shape[1]);
    std::vector<double> m(static_cast<size_t>(n_test));
    reg.predict(Xte.f8(), n_test, m.data());
    for (double v : m) y_scale = std::max(y_scale, std::abs(v));
    means.push_back(std::move(m));
  }
  for (size_t m = 1; m < means.size(); ++m) {
    double max_err = 0;
    for (size_t i = 0; i < means[0].size(); ++i)
      max_err = std::max(max_err, std::abs(means[m][i] - means[0][i]));
    CHECK_MESSAGE(max_err <= 1e-4 * y_scale,
                  "cache mode " << m << " diverges from uncached: " << max_err);
  }
}

TEST_CASE("cache: cached classifier still matches the Python reference") {
  if (!fixture_exists("e2e/clf_small_Xtr.npy")) return;
  auto model = tabicl::Model::load(fixture_path("tabicl-classifier-v2.gguf"));
  auto Xtr = load_fixture("e2e/clf_small_Xtr.npy");
  auto ytr = load_fixture("e2e/clf_small_ytr.npy");
  auto Xte = load_fixture("e2e/clf_small_Xte.npy");
  auto proba_ref = load_fixture("e2e/clf_small_proba.npy");
  const int64_t n_test = Xte.shape[0];

  EstimatorOptions opts;
  opts.cache = CacheMode::KV;
  opts.n_threads = 8;
  tabicl::TabICLClassifier clf(model, opts);
  clf.fit(Xtr.f8(), ytr.f8(), Xtr.shape[0], Xtr.shape[1]);
  std::vector<float> p(static_cast<size_t>(n_test * clf.n_classes()));
  clf.predict_proba(Xte.f8(), n_test, p.data());
  int64_t bad = 0;
  for (int64_t i = 0; i < proba_ref.numel(); ++i)
    if (std::abs(static_cast<double>(p[static_cast<size_t>(i)]) - proba_ref.f4()[i]) >
        1e-4 + 1e-4 * std::abs(proba_ref.f4()[i]))
      bad++;
  CHECK_MESSAGE(bad == 0, bad << " cached proba values out of tolerance vs Python");
}
