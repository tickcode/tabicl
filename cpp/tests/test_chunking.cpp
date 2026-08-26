// Scratch-budget chunking: results must be BITWISE identical for any
// max_scratch_bytes (chunks split only embarrassingly-parallel axes, never a
// softmax/LayerNorm reduction axis).
#include <cmath>
#include <cstring>
#include <vector>

#include "doctest.h"
#include "estimator_core.h"
#include "model_forward.h"
#include "tabicl/classifier.h"
#include "test_helpers.h"

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

constexpr int64_t kTiny = int64_t(1) << 18;  // 256 KiB: forces minimal chunks
constexpr int64_t kHuge = int64_t(1) << 33;  // 8 GiB: single chunk everywhere

}  // namespace

TEST_CASE("chunking: stage outputs are bitwise budget-invariant") {
  if (!fixture_exists("model_small/clf_X.npy") ||
      !fixture_exists("tabicl-classifier-v2.gguf")) {
    MESSAGE("fixtures missing");
    return;
  }
  auto model = tabicl::Model::load(fixture_path("tabicl-classifier-v2.gguf"));
  tabicl::GraphRunner runner(8);
  auto X = load_fixture("model_small/clf_X.npy");
  auto y = load_fixture("model_small/clf_y.npy");
  const int64_t B = X.shape[0], T = X.shape[1], H = X.shape[2];
  const int64_t train = y.shape[1];

  const auto col_a = tabicl::col_embedding_forward(*model, runner, X.f4(), y.f4(),
                                                   B, T, H, train, 3, kTiny);
  const auto col_b = tabicl::col_embedding_forward(*model, runner, X.f4(), y.f4(),
                                                   B, T, H, train, 3, kHuge);
  CHECK(bitwise_mismatches(col_a, col_b) == 0);

  const auto row_a =
      tabicl::row_interaction_forward(*model, runner, col_a.data(), B, T, H, kTiny);
  const auto row_b =
      tabicl::row_interaction_forward(*model, runner, col_a.data(), B, T, H, kHuge);
  CHECK(bitwise_mismatches(row_a, row_b) == 0);

  const auto icl_a = tabicl::icl_forward(*model, runner, row_a.data(), y.f4(), B, T,
                                         train, 3, kTiny);
  const auto icl_b = tabicl::icl_forward(*model, runner, row_a.data(), y.f4(), B, T,
                                         train, 3, kHuge);
  CHECK(bitwise_mismatches(icl_a, icl_b) == 0);
}

TEST_CASE("chunking: full estimator predictions are bitwise budget-invariant") {
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

  for (tabicl::CacheMode mode :
       {tabicl::CacheMode::None, tabicl::CacheMode::KV, tabicl::CacheMode::Repr}) {
    std::vector<std::vector<float>> probas;
    for (int64_t budget : {kTiny, kHuge}) {
      tabicl::EstimatorOptions opts;
      opts.cache = mode;
      opts.n_threads = 8;
      opts.max_scratch_bytes = budget;
      tabicl::TabICLClassifier clf(model, opts);
      clf.fit(Xtr.f8(), ytr.f8(), Xtr.shape[0], Xtr.shape[1]);
      std::vector<float> p(static_cast<size_t>(n_test * clf.n_classes()));
      clf.predict_proba(Xte.f8(), n_test, p.data());
      probas.push_back(std::move(p));
    }
    CHECK_MESSAGE(bitwise_mismatches(probas[0], probas[1]) == 0,
                  "cache mode " << static_cast<int>(mode));
  }
}

TEST_CASE("chunking: hierarchical (>10 classes) is bitwise budget-invariant") {
  if (!fixture_exists("e2e/clf_many_Xtr.npy")) return;
  auto model = tabicl::Model::load(fixture_path("tabicl-classifier-v2.gguf"));
  auto Xtr = load_fixture("e2e/clf_many_Xtr.npy");
  auto ytr = load_fixture("e2e/clf_many_ytr.npy");
  auto Xte = load_fixture("e2e/clf_many_Xte.npy");
  const int64_t n_test = Xte.shape[0];

  std::vector<std::vector<float>> probas;
  for (int64_t budget : {kTiny, kHuge}) {
    tabicl::EstimatorOptions opts;
    opts.n_estimators = 4;
    opts.n_threads = 8;
    opts.max_scratch_bytes = budget;
    tabicl::TabICLClassifier clf(model, opts);
    clf.fit(Xtr.f8(), ytr.f8(), Xtr.shape[0], Xtr.shape[1]);
    std::vector<float> p(static_cast<size_t>(n_test * clf.n_classes()));
    clf.predict_proba(Xte.f8(), n_test, p.data());
    probas.push_back(std::move(p));
  }
  CHECK(bitwise_mismatches(probas[0], probas[1]) == 0);
}
