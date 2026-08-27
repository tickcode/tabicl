// sklearn.preprocessing.QuantileTransformer(output_distribution="normal"),
// dense path, plus the RTDL wrapper (fit-time PCG64 noise + n_quantiles rule
// + trailing StandardScaler) used by tabicl's "quantile_rtdl" norm method.
#pragma once

#include <cstdint>
#include <vector>

#include "preprocess/sklearn_scaler.h"
#include "preprocess/transforms.h"

namespace tabicl {

class QuantileTransformerNormal {
 public:
  // sklearn defaults: n_quantiles=1000, subsample=10000. TabICL's plain
  // "quantile" method uses these; RTDL passes its own values.
  void fit(const Matrix& X, int64_t n_quantiles, int64_t subsample,
           uint32_t random_state);
  Matrix transform(const Matrix& X) const;

  const std::vector<double>& references() const { return references_; }
  // quantiles_[q * d + j] — row-major (n_quantiles_, d), like numpy.
  const std::vector<double>& quantiles() const { return quantiles_; }
  int64_t n_quantiles_fitted() const { return n_quantiles_; }
  int64_t d() const { return d_; }
  void restore(int64_t n_quantiles, int64_t d, std::vector<double> references,
               std::vector<double> quantiles) {
    n_quantiles_ = n_quantiles;
    d_ = d;
    references_ = std::move(references);
    quantiles_ = std::move(quantiles);
  }

 private:
  int64_t n_quantiles_ = 0;
  int64_t d_ = 0;
  std::vector<double> references_;
  std::vector<double> quantiles_;
};

// tabicl RTDLQuantileTransformer + trailing sklearn StandardScaler
// (the "quantile_rtdl" pipeline stage).
class RTDLQuantileTransformer {
 public:
  void fit(const Matrix& X, uint64_t random_state);
  Matrix transform(const Matrix& X) const;

  const QuantileTransformerNormal& qt() const { return qt_; }
  const SkStandardScaler& scaler() const { return scaler_; }
  void restore(QuantileTransformerNormal qt, SkStandardScaler scaler) {
    qt_ = std::move(qt);
    scaler_ = std::move(scaler);
  }

 private:
  QuantileTransformerNormal qt_;
  SkStandardScaler scaler_;
};

}  // namespace tabicl
