// Fitted-state serialization: save -> load must reproduce predictions
// BITWISE (all state is stored exactly), reject wrong models/tasks, and fail
// cleanly on corrupt files.
#include <cstdio>
#include <cstring>
#include <fstream>
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

namespace {

int64_t bitwise_mismatches(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  int64_t bad = 0;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) bad++;
  return bad;
}

struct TempFile {
  std::string path;
  explicit TempFile(std::string p) : path(std::move(p)) {}
  ~TempFile() { std::remove(path.c_str()); }
};

}  // namespace

TEST_CASE("fitted: classifier roundtrip is bitwise across all cache modes") {
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

  for (CacheMode mode : {CacheMode::None, CacheMode::KV, CacheMode::Repr}) {
    TempFile f("fitted_clf_" + std::to_string(static_cast<int>(mode)) + ".gguf");
    EstimatorOptions opts;
    opts.cache = mode;
    opts.n_threads = 8;
    tabicl::TabICLClassifier clf(model, opts);
    clf.fit(Xtr.f8(), ytr.f8(), Xtr.shape[0], Xtr.shape[1]);
    std::vector<float> before(static_cast<size_t>(n_test * clf.n_classes()));
    clf.predict_proba(Xte.f8(), n_test, before.data());
    clf.save(f.path);

    auto loaded = tabicl::TabICLClassifier::load(f.path, model);
    REQUIRE(loaded.classes() == clf.classes());
    std::vector<float> after(before.size());
    loaded.predict_proba(Xte.f8(), n_test, after.data());
    CHECK_MESSAGE(bitwise_mismatches(before, after) == 0,
                  "cache mode " << static_cast<int>(mode));

    // Thread-count override must not change results (thread invariance).
    auto loaded2 = tabicl::TabICLClassifier::load(f.path, model, 2);
    std::vector<float> after2(before.size());
    loaded2.predict_proba(Xte.f8(), n_test, after2.data());
    CHECK(bitwise_mismatches(before, after2) == 0);
  }
}

TEST_CASE("fitted: all-norm-method pipelines survive the roundtrip") {
  if (!fixture_exists("e2e/clf_allnorm_Xtr.npy")) return;
  auto model = tabicl::Model::load(fixture_path("tabicl-classifier-v2.gguf"));
  auto Xtr = load_fixture("e2e/clf_allnorm_Xtr.npy");
  auto ytr = load_fixture("e2e/clf_allnorm_ytr.npy");
  auto Xte = load_fixture("e2e/clf_allnorm_Xte.npy");
  const int64_t n_test = Xte.shape[0];

  TempFile f("fitted_clf_allnorm.gguf");
  EstimatorOptions opts;
  opts.n_estimators = 10;
  opts.norm_methods = {"none", "power", "robust", "quantile", "quantile_rtdl"};
  opts.n_threads = 8;
  tabicl::TabICLClassifier clf(model, opts);
  clf.fit(Xtr.f8(), ytr.f8(), Xtr.shape[0], Xtr.shape[1]);
  std::vector<float> before(static_cast<size_t>(n_test * clf.n_classes()));
  clf.predict_proba(Xte.f8(), n_test, before.data());
  clf.save(f.path);

  auto loaded = tabicl::TabICLClassifier::load(f.path, model);
  std::vector<float> after(before.size());
  loaded.predict_proba(Xte.f8(), n_test, after.data());
  CHECK(bitwise_mismatches(before, after) == 0);
}

TEST_CASE("fitted: hierarchical (>10 classes) roundtrip is bitwise") {
  if (!fixture_exists("e2e/clf_many_Xtr.npy")) return;
  auto model = tabicl::Model::load(fixture_path("tabicl-classifier-v2.gguf"));
  auto Xtr = load_fixture("e2e/clf_many_Xtr.npy");
  auto ytr = load_fixture("e2e/clf_many_ytr.npy");
  auto Xte = load_fixture("e2e/clf_many_Xte.npy");
  const int64_t n_test = Xte.shape[0];

  TempFile f("fitted_clf_many.gguf");
  EstimatorOptions opts;
  opts.n_estimators = 4;
  opts.n_threads = 8;
  tabicl::TabICLClassifier clf(model, opts);
  clf.fit(Xtr.f8(), ytr.f8(), Xtr.shape[0], Xtr.shape[1]);
  std::vector<float> before(static_cast<size_t>(n_test * clf.n_classes()));
  clf.predict_proba(Xte.f8(), n_test, before.data());
  clf.save(f.path);

  auto loaded = tabicl::TabICLClassifier::load(f.path, model);
  std::vector<float> after(before.size());
  loaded.predict_proba(Xte.f8(), n_test, after.data());
  CHECK(bitwise_mismatches(before, after) == 0);
}

TEST_CASE("fitted: regressor roundtrip is bitwise (mean + quantiles)") {
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
  const double alphas[3] = {0.1, 0.5, 0.9};

  TempFile f("fitted_reg.gguf");
  EstimatorOptions opts;
  opts.cache = CacheMode::KV;
  opts.n_threads = 8;
  tabicl::TabICLRegressor reg(model, opts);
  reg.fit(Xtr.f8(), ytr.f8(), Xtr.shape[0], Xtr.shape[1]);
  std::vector<double> mean_before(static_cast<size_t>(n_test));
  std::vector<double> q_before(static_cast<size_t>(n_test * 3));
  reg.predict(Xte.f8(), n_test, mean_before.data());
  reg.predict_quantiles(Xte.f8(), n_test, alphas, 3, q_before.data());
  reg.save(f.path);

  auto loaded = tabicl::TabICLRegressor::load(f.path, model);
  std::vector<double> mean_after(mean_before.size());
  std::vector<double> q_after(q_before.size());
  loaded.predict(Xte.f8(), n_test, mean_after.data());
  loaded.predict_quantiles(Xte.f8(), n_test, alphas, 3, q_after.data());
  CHECK(std::memcmp(mean_before.data(), mean_after.data(),
                    mean_before.size() * sizeof(double)) == 0);
  CHECK(std::memcmp(q_before.data(), q_after.data(),
                    q_before.size() * sizeof(double)) == 0);
}

TEST_CASE("fitted: wrong model, wrong task, and corrupt files are rejected") {
  if (!fixture_exists("e2e/clf_small_Xtr.npy") ||
      !fixture_exists("tabicl-regressor-v2.gguf")) {
    MESSAGE("fixtures missing");
    return;
  }
  auto clf_model = tabicl::Model::load(fixture_path("tabicl-classifier-v2.gguf"));
  auto reg_model = tabicl::Model::load(fixture_path("tabicl-regressor-v2.gguf"));
  auto Xtr = load_fixture("e2e/clf_small_Xtr.npy");
  auto ytr = load_fixture("e2e/clf_small_ytr.npy");

  TempFile f("fitted_reject.gguf");
  tabicl::EstimatorOptions opts;
  opts.n_threads = 8;
  tabicl::TabICLClassifier clf(clf_model, opts);
  clf.fit(Xtr.f8(), ytr.f8(), Xtr.shape[0], Xtr.shape[1]);
  clf.save(f.path);

  // Wrong checkpoint (regressor model) -> fingerprint mismatch.
  CHECK_THROWS(tabicl::TabICLClassifier::load(f.path, reg_model));
  // Wrong task loader.
  CHECK_THROWS(tabicl::TabICLRegressor::load(f.path, reg_model));
  // Truncated file.
  TempFile g("fitted_trunc.gguf");
  {
    std::ifstream in(f.path, std::ios::binary);
    std::vector<char> buf(4096);
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    std::ofstream out(g.path, std::ios::binary);
    out.write(buf.data(), in.gcount() / 2);
  }
  CHECK_THROWS(tabicl::TabICLClassifier::load(g.path, clf_model));
  // Nonexistent file.
  CHECK_THROWS(tabicl::TabICLClassifier::load("/nonexistent/x.gguf", clf_model));
}
