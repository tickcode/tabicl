// End-to-end estimator parity: C++ TabICLClassifier/TabICLRegressor
// fit+predict vs the Python estimators on identical data (section e2e).
// Tolerance rtol=1e-4, atol=1e-4 (repo precedent; Python's own norm-method
// group order is PYTHONHASHSEED-dependent, which perturbs fp32 accumulation).
#include <cmath>
#include <string>
#include <vector>

#include "doctest.h"
#include "estimator_core.h"
#include "tabicl/classifier.h"
#include "tabicl/regressor.h"
#include "test_helpers.h"

using tabicl::test::fixture_exists;
using tabicl::test::fixture_path;
using tabicl::test::load_fixture;

namespace {

void check_close(const double* got, const double* ref, int64_t n, double atol,
                 double rtol, const std::string& what) {
  for (int64_t i = 0; i < n; ++i) {
    REQUIRE_MESSAGE(std::abs(got[i] - ref[i]) <= atol + rtol * std::abs(ref[i]),
                    what << "[" << i << "]: " << got[i] << " vs " << ref[i]);
  }
}

void run_clf(const std::string& name, int n_estimators,
             const std::vector<std::string>& methods) {
  auto model = tabicl::Model::load(fixture_path("tabicl-classifier-v2.gguf"));
  auto Xtr = load_fixture("e2e/" + name + "_Xtr.npy");
  auto ytr = load_fixture("e2e/" + name + "_ytr.npy");
  auto Xte = load_fixture("e2e/" + name + "_Xte.npy");
  auto proba_ref = load_fixture("e2e/" + name + "_proba.npy");
  auto pred_ref = load_fixture("e2e/" + name + "_pred.npy");

  tabicl::EstimatorOptions opts;
  opts.n_estimators = n_estimators;
  if (!methods.empty()) opts.norm_methods = methods;
  opts.n_threads = 8;
  tabicl::TabICLClassifier clf(model, opts);
  clf.fit(Xtr.f8(), ytr.f8(), Xtr.shape[0], Xtr.shape[1]);

  const int64_t n_test = Xte.shape[0];
  const int64_t C = clf.n_classes();
  REQUIRE(C == proba_ref.shape[1]);
  std::vector<float> proba(static_cast<size_t>(n_test * C));
  clf.predict_proba(Xte.f8(), n_test, proba.data());
  int64_t bad = 0;
  double max_err = 0;
  for (int64_t i = 0; i < n_test * C; ++i) {
    const double diff = std::abs(static_cast<double>(proba[i]) - proba_ref.f4()[i]);
    max_err = std::max(max_err, diff);
    if (diff > 1e-4 + 1e-4 * std::abs(proba_ref.f4()[i])) bad++;
  }
  REQUIRE_MESSAGE(bad == 0, name << ": " << bad << " proba values out of tolerance"
                                 << " (max err " << max_err << ")");

  std::vector<double> pred(static_cast<size_t>(n_test));
  clf.predict(Xte.f8(), n_test, pred.data());
  int64_t label_mismatch = 0;
  for (int64_t i = 0; i < n_test; ++i)
    if (pred[static_cast<size_t>(i)] != pred_ref.f8()[i]) {
      // A label flip is only acceptable when the top two probabilities are
      // within 2e-4 of each other (documented tie tolerance).
      float top1 = -1.0f, top2 = -1.0f;
      for (int64_t k = 0; k < C; ++k) {
        const float p = proba[static_cast<size_t>(i * C + k)];
        if (p > top1) {
          top2 = top1;
          top1 = p;
        } else if (p > top2) {
          top2 = p;
        }
      }
      if (top1 - top2 > 2e-4f) label_mismatch++;
    }
  CHECK_MESSAGE(label_mismatch == 0, name << ": label mismatches beyond ties");
}

}  // namespace

TEST_CASE("e2e: classifier matches Python (defaults)") {
  if (!fixture_exists("e2e/clf_small_Xtr.npy") ||
      !fixture_exists("tabicl-classifier-v2.gguf")) {
    MESSAGE("e2e fixtures missing; run tools/generate_fixtures.py --only e2e");
    return;
  }
  run_clf("clf_small", 8, {});
}

TEST_CASE("e2e: classifier matches Python (all norm methods)") {
  if (!fixture_exists("e2e/clf_allnorm_Xtr.npy")) return;
  run_clf("clf_allnorm", 10,
          {"none", "power", "robust", "quantile", "quantile_rtdl"});
}

TEST_CASE("e2e: classifier matches Python (binary + NaNs)") {
  if (!fixture_exists("e2e/clf_binary_nan_Xtr.npy")) return;
  run_clf("clf_binary_nan", 4, {});
}

TEST_CASE("e2e: classifier matches Python (14 classes, hierarchical)") {
  if (!fixture_exists("e2e/clf_many_Xtr.npy")) return;
  run_clf("clf_many", 4, {});
}

TEST_CASE("e2e: regressor matches Python (mean + quantiles)") {
  if (!fixture_exists("e2e/reg_small_Xtr.npy") ||
      !fixture_exists("tabicl-regressor-v2.gguf")) {
    MESSAGE("e2e fixtures missing; run tools/generate_fixtures.py --only e2e");
    return;
  }
  auto model = tabicl::Model::load(fixture_path("tabicl-regressor-v2.gguf"));
  auto Xtr = load_fixture("e2e/reg_small_Xtr.npy");
  auto ytr = load_fixture("e2e/reg_small_ytr.npy");
  auto Xte = load_fixture("e2e/reg_small_Xte.npy");
  auto mean_ref = load_fixture("e2e/reg_small_mean.npy");
  auto q_ref = load_fixture("e2e/reg_small_quantiles.npy");
  auto alphas = load_fixture("e2e/reg_small_alphas.npy");

  tabicl::EstimatorOptions opts;
  opts.n_threads = 8;
  tabicl::TabICLRegressor reg(model, opts);
  reg.fit(Xtr.f8(), ytr.f8(), Xtr.shape[0], Xtr.shape[1]);

  const int64_t n_test = Xte.shape[0];
  // Scale-aware tolerance: targets are O(5).
  double y_scale = 0;
  for (int64_t i = 0; i < mean_ref.numel(); ++i)
    y_scale = std::max(y_scale, std::abs(mean_ref.f8()[i]));

  std::vector<double> mean(static_cast<size_t>(n_test));
  reg.predict(Xte.f8(), n_test, mean.data());
  check_close(mean.data(), mean_ref.f8(), n_test, 1e-4 * y_scale, 1e-4,
              "reg_small/mean");

  const int64_t na = alphas.numel();
  std::vector<double> q(static_cast<size_t>(n_test * na));
  reg.predict_quantiles(Xte.f8(), n_test, alphas.f8(), na, q.data());
  REQUIRE(q_ref.shape[0] * q_ref.shape[1] == n_test * na);
  check_close(q.data(), q_ref.f8(), n_test * na, 1e-4 * y_scale, 1e-4,
              "reg_small/quantiles");
}
