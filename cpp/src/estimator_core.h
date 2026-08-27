// Shared fit/predict machinery for TabICLClassifier / TabICLRegressor:
// numeric-input imputation, unique-feature filtering, ensemble configs,
// per-method preprocessing pipelines, member assembly and batched forwards.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ggml_utils.h"
#include "io/fitted_state.h"
#include "model_forward.h"
#include "preprocess/ensemble.h"
#include "preprocess/pipeline.h"
#include "preprocess/transforms.h"
#include "tabicl/model.h"
#include "tabicl/options.h"

namespace tabicl {

class EstimatorCore {
 public:
  // y_model: per-train-row value fed to the model (encoded class index for
  // classification, scaled target for regression). n_classes = 0 for
  // regression. `model` is required when opts.cache != None (cache build).
  void fit(const double* X, int64_t n, int64_t d, std::vector<float> y_model,
           int64_t n_classes, const EstimatorOptions& opts,
           const Model* model = nullptr);

  // Runs the ensemble forward for test data. Returns per-member outputs,
  // each (n_test, out_dim) fp32, in the canonical member order, along with
  // the flattened class shuffles (empty vectors for regression).
  struct EnsembleOutputs {
    std::vector<std::vector<float>> member_outputs;
    std::vector<std::vector<int32_t>> class_shuffles;
    int64_t out_dim = 0;
  };
  EnsembleOutputs predict_outputs(const Model& model, const double* X_test,
                                  int64_t n_test) const;

  int64_t n_features_in() const { return d_in_; }
  int64_t train_size() const { return static_cast<int64_t>(y_model_.size()); }
  const EstimatorOptions& options() const { return opts_; }

  // Fitted-state (de)serialization; the schema is versioned by the wrapper.
  void save(FittedWriter& w) const;
  void load(const FittedReader& r, int n_threads_override = -1);

 private:
  EstimatorOptions opts_;
  int64_t d_in_ = 0;
  int64_t n_classes_ = 0;
  std::vector<float> y_model_;
  MeanImputer imputer_;
  UniqueFeatureFilter filter_;
  EnsembleConfigs configs_;
  // Pipelines per configs_.method_order entry.
  std::vector<std::unique_ptr<PreprocessingPipeline>> pipelines_;
  // KV caches per method group, one per predict-time batch (built at fit
  // when opts_.cache != None; empty otherwise).
  std::vector<std::vector<TabICLCache>> caches_;
  mutable std::unique_ptr<GraphRunner> runner_;
};

// np.array_split sizes: k parts of n, first n%k parts one element larger.
std::vector<int64_t> array_split_sizes(int64_t n, int64_t k);

// Cheap checkpoint identity (SHA256 of the decoder output projection); stored
// in fitted-state files and verified at load.
std::string model_fingerprint(const Model& model);

}  // namespace tabicl
