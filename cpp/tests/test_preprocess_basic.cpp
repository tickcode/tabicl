// Validation of MeanImputer / UniqueFeatureFilter / CustomStandardScaler /
// OutlierRemover against Python-generated fixtures (section preprocess_basic).
#include <cmath>
#include <string>
#include <vector>

#include "doctest.h"
#include "preprocess/transforms.h"
#include "test_helpers.h"

using tabicl::Matrix;
using tabicl::test::fixture_exists;
using tabicl::test::load_fixture;

namespace {

Matrix to_matrix(const tabicl::npy::Array& a) {
  REQUIRE(a.shape.size() == 2);
  Matrix m;
  m.n = a.shape[0];
  m.d = a.shape[1];
  m.data.assign(a.f8(), a.f8() + a.numel());
  return m;
}

void check_close(const std::vector<double>& got, const tabicl::npy::Array& ref,
                 double atol, double rtol, const std::string& what) {
  REQUIRE_MESSAGE(static_cast<int64_t>(got.size()) == ref.numel(), what << ": size");
  for (size_t i = 0; i < got.size(); ++i) {
    const double r = ref.f8()[i];
    if (std::isnan(r)) {
      REQUIRE_MESSAGE(std::isnan(got[i]), what << "[" << i << "]: expected NaN");
    } else {
      REQUIRE_MESSAGE(std::abs(got[i] - r) <= atol + rtol * std::abs(r),
                      what << "[" << i << "]: " << got[i] << " vs " << r);
    }
  }
}

void check_matrix(const Matrix& got, const tabicl::npy::Array& ref, double atol,
                  double rtol, const std::string& what) {
  REQUIRE_MESSAGE(got.n == ref.shape[0], what << ": rows");
  REQUIRE_MESSAGE(got.d == ref.shape[1], what << ": cols");
  check_close(got.data, ref, atol, rtol, what);
}

}  // namespace

TEST_CASE("preprocess: imputer/filter/scaler/outlier match Python fixtures") {
  if (!fixture_exists("preprocess_basic/plain_X.npy")) {
    MESSAGE("preprocess_basic fixtures missing; run tools/generate_fixtures.py");
    return;
  }
  const double kAtol = 1e-12, kRtol = 1e-13;
  for (const std::string name : {"plain", "edge", "single_col", "tiny", "ties", "large"}) {
    CAPTURE(name);
    Matrix X = to_matrix(load_fixture("preprocess_basic/" + name + "_X.npy"));

    tabicl::MeanImputer imp;
    imp.fit(X);
    Matrix Xi = imp.transform(X);
    check_matrix(Xi, load_fixture("preprocess_basic/" + name + "_imputed.npy"),
                 kAtol, kRtol, name + "/imputed");
    check_close(imp.statistics(),
                load_fixture("preprocess_basic/" + name + "_imputer_stats.npy"),
                kAtol, kRtol, name + "/imputer_stats");

    tabicl::UniqueFeatureFilter uf;
    uf.fit(Xi);
    {
      auto ref = load_fixture("preprocess_basic/" + name + "_filter_keep.npy");
      REQUIRE(static_cast<int64_t>(uf.features_to_keep().size()) == ref.numel());
      for (int64_t j = 0; j < ref.numel(); ++j)
        REQUIRE_MESSAGE(uf.features_to_keep()[static_cast<size_t>(j)] ==
                            (ref.i8()[j] != 0),
                        name << "/filter_keep[" << j << "]");
    }
    Matrix Xu = uf.transform(Xi);

    tabicl::CustomStandardScaler ss;
    ss.fit(Xu);
    check_close(ss.mean(), load_fixture("preprocess_basic/" + name + "_scaler_mean.npy"),
                kAtol, kRtol, name + "/scaler_mean");
    check_close(ss.scale(), load_fixture("preprocess_basic/" + name + "_scaler_scale.npy"),
                kAtol, kRtol, name + "/scaler_scale");
    Matrix Xs = ss.transform(Xu);
    check_matrix(Xs, load_fixture("preprocess_basic/" + name + "_scaled.npy"),
                 kAtol, kRtol, name + "/scaled");

    tabicl::OutlierRemover outl;
    outl.fit(Xs, 4.0);
    check_close(outl.lower_bounds(),
                load_fixture("preprocess_basic/" + name + "_outlier_lower.npy"),
                kAtol, kRtol, name + "/outlier_lower");
    check_close(outl.upper_bounds(),
                load_fixture("preprocess_basic/" + name + "_outlier_upper.npy"),
                kAtol, kRtol, name + "/outlier_upper");
    Matrix Xo = outl.transform(Xs);
    check_matrix(Xo, load_fixture("preprocess_basic/" + name + "_outlier_out.npy"),
                 kAtol, kRtol, name + "/outlier_out");
  }
}
