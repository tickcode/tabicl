// TabICL C++ quick start: fit a classifier on numeric data, predict, and
// round-trip the fitted state through a file. No Python anywhere.
//
//   ./tabicl_quickstart path/to/tabicl-classifier-v2.gguf
//
// (Get the model file from the repo's GitHub releases, or convert a
// checkpoint once with scripts/export_gguf.py.)
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <tabicl/classifier.h>
#include <tabicl/model.h>
#include <tabicl/options.h>

namespace {

double now_s() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <tabicl-classifier-v2.gguf>\n", argv[0]);
    return 1;
  }

  // Synthetic 3-class dataset: three Gaussian blobs in 6 dimensions.
  const int64_t n_train = 400, n_test = 50, d = 6, n_classes = 3;
  std::mt19937 rng(7);
  std::normal_distribution<double> noise(0.0, 1.0);
  std::uniform_int_distribution<int> cls(0, static_cast<int>(n_classes) - 1);
  std::vector<double> Xtr(n_train * d), ytr(n_train), Xte(n_test * d), yte(n_test);
  auto fill = [&](std::vector<double>& X, std::vector<double>& y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
      const int c = cls(rng);
      y[static_cast<size_t>(i)] = c;
      for (int64_t j = 0; j < d; ++j)
        X[static_cast<size_t>(i * d + j)] = noise(rng) + 2.5 * c * ((j % 2) ? 1 : -1);
    }
  };
  fill(Xtr, ytr, n_train);
  fill(Xte, yte, n_test);

  // Load the model once; it is immutable and shareable between estimators.
  auto model = tabicl::Model::load(argv[1]);

  tabicl::EstimatorOptions opts;
  opts.cache = tabicl::CacheMode::KV;  // cache training K/V during fit
  opts.n_threads = 0;                  // all cores

  tabicl::TabICLClassifier clf(model, opts);
  double t0 = now_s();
  clf.fit(Xtr.data(), ytr.data(), n_train, d);
  std::printf("fit (incl. KV-cache build): %.2fs\n", now_s() - t0);

  std::vector<float> proba(n_test * clf.n_classes());
  std::vector<double> pred(n_test);
  t0 = now_s();
  clf.predict_proba(Xte.data(), n_test, proba.data());
  clf.predict(Xte.data(), n_test, pred.data());
  std::printf("predict: %.2fs\n", now_s() - t0);

  int64_t correct = 0;
  for (int64_t i = 0; i < n_test; ++i)
    if (pred[static_cast<size_t>(i)] == yte[static_cast<size_t>(i)]) correct++;
  std::printf("accuracy on held-out blobs: %lld/%lld\n",
              static_cast<long long>(correct), static_cast<long long>(n_test));

  // Persist the fitted state (preprocessing + ensemble + KV cache) and load
  // it back; predictions from the loaded estimator are bitwise identical.
  clf.save("quickstart_fitted.gguf");
  auto clf2 = tabicl::TabICLClassifier::load("quickstart_fitted.gguf", model);
  std::vector<float> proba2(proba.size());
  clf2.predict_proba(Xte.data(), n_test, proba2.data());
  bool identical = true;
  for (size_t i = 0; i < proba.size(); ++i)
    if (proba[i] != proba2[i]) identical = false;
  std::printf("save -> load round trip: %s\n",
              identical ? "bitwise identical" : "MISMATCH (bug!)");
  std::remove("quickstart_fitted.gguf");

  std::printf("first test row probabilities:");
  for (int64_t k = 0; k < clf.n_classes(); ++k)
    std::printf(" %.4f", proba[static_cast<size_t>(k)]);
  std::printf("\n");
  return identical ? 0 : 1;
}
