// Stage-level and end-to-end parity of the C++ model forward against
// golden tensors dumped from the real PyTorch model (section model_small).
// Each stage is tested in isolation on the PYTHON reference input, then the
// full chain is compared end-to-end.
#include <cmath>
#include <string>
#include <vector>

#include "doctest.h"
#include "model_forward.h"
#include "tabicl/model.h"
#include "test_helpers.h"

using tabicl::test::fixture_exists;
using tabicl::test::fixture_path;
using tabicl::test::load_fixture;

namespace {

struct MaxErr {
  double abs = 0, rel = 0;
  int64_t count = 0;
};

MaxErr compare(const float* got, const float* ref, int64_t n, double atol,
               double rtol, const std::string& what) {
  MaxErr e;
  for (int64_t i = 0; i < n; ++i) {
    const double a = got[i], r = ref[i];
    const double diff = std::abs(a - r);
    e.abs = std::max(e.abs, diff);
    if (r != 0) e.rel = std::max(e.rel, diff / std::abs(r));
    if (diff > atol + rtol * std::abs(r)) e.count++;
  }
  REQUIRE_MESSAGE(e.count == 0, what << ": " << e.count << "/" << n
                                     << " values out of tolerance (max abs err "
                                     << e.abs << ", max rel err " << e.rel << ")");
  return e;
}

void run_task(const std::string& task, const std::string& gguf) {
  auto model = tabicl::Model::load(fixture_path(gguf));
  tabicl::GraphRunner runner(8);

  auto X = load_fixture("model_small/" + task + "_X.npy");
  auto y = load_fixture("model_small/" + task + "_y.npy");
  const int64_t B = X.shape[0], T = X.shape[1], H = X.shape[2];
  const int64_t train = y.shape[1];
  const int64_t n_classes = task == "clf" ? 3 : 0;

  // Stage 1: ColEmbedding
  auto col_ref = load_fixture("model_small/" + task + "_col_out.npy");
  auto col = tabicl::col_embedding_forward(*model, runner, X.f4(), y.f4(), B, T, H,
                                           train, n_classes);
  compare(col.data(), col_ref.f4(), col_ref.numel(), 1e-5, 1e-5, task + "/col_out");

  // Stage 2: RowInteraction on the Python col output
  auto row_ref = load_fixture("model_small/" + task + "_row_out.npy");
  auto row = tabicl::row_interaction_forward(*model, runner, col_ref.f4(), B, T, H);
  compare(row.data(), row_ref.f4(), row_ref.numel(), 5e-5, 5e-5, task + "/row_out");

  // Stage 3: ICL + decoder on the Python row output
  auto logits_ref = load_fixture("model_small/" + task + "_logits.npy");
  auto raw = tabicl::icl_forward(*model, runner, row_ref.f4(), y.f4(), B, T, train,
                                 n_classes);
  const int64_t full_out = task == "clf" ? model->config().max_classes
                                         : model->config().num_quantiles;
  const int64_t out_dim = logits_ref.shape[2];
  const int64_t test = T - train;
  // icl_forward returns the test rows only; slice classes.
  std::vector<float> sliced(static_cast<size_t>(B * test * out_dim));
  for (int64_t b = 0; b < B; ++b)
    for (int64_t t = 0; t < test; ++t)
      for (int64_t k = 0; k < out_dim; ++k)
        sliced[static_cast<size_t>((b * test + t) * out_dim + k)] =
            raw[static_cast<size_t>((b * test + t) * full_out + k)];
  compare(sliced.data(), logits_ref.f4(), logits_ref.numel(), 5e-4, 1e-4,
          task + "/icl_logits");

  // Full chain end-to-end
  tabicl::ForwardOptions opts;
  opts.train_size = train;
  opts.num_classes = n_classes;
  opts.return_logits = true;
  auto e2e = tabicl_forward(*model, runner, X.f4(), y.f4(), B, T, H, opts);
  compare(e2e.data(), logits_ref.f4(), logits_ref.numel(), 5e-4, 1e-4,
          task + "/e2e_logits");

  if (task == "clf") {
    auto probs_ref = load_fixture("model_small/clf_probs.npy");
    opts.return_logits = false;
    auto probs = tabicl_forward(*model, runner, X.f4(), y.f4(), B, T, H, opts);
    compare(probs.data(), probs_ref.f4(), probs_ref.numel(), 1e-5, 1e-4,
            "clf/e2e_probs");
  }
}

}  // namespace

TEST_CASE("model: classifier stages match PyTorch goldens") {
  if (!fixture_exists("model_small/clf_X.npy") ||
      !fixture_exists("tabicl-classifier-v2.gguf")) {
    MESSAGE("model_small fixtures missing; run tools/generate_fixtures.py + export_gguf");
    return;
  }
  run_task("clf", "tabicl-classifier-v2.gguf");
}

TEST_CASE("model: regressor stages match PyTorch goldens") {
  if (!fixture_exists("model_small/reg_X.npy") ||
      !fixture_exists("tabicl-regressor-v2.gguf")) {
    MESSAGE("model_small fixtures missing; run tools/generate_fixtures.py + export_gguf");
    return;
  }
  run_task("reg", "tabicl-regressor-v2.gguf");
}
