// Validation of ndtri, SkStandardScaler, PowerTransformerYJ, RobustScalerUV,
// and QuantileTransformer (plain + RTDL) against Python fixtures.
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "doctest.h"
#include "preprocess/cephes_ndtri.h"
#include "preprocess/power_transform.h"
#include "preprocess/quantile_transform.h"
#include "preprocess/robust_scaler.h"
#include "preprocess/sklearn_scaler.h"
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

void check_vals(const double* got, const tabicl::npy::Array& ref, double atol,
                double rtol, const std::string& what) {
  for (int64_t i = 0; i < ref.numel(); ++i) {
    const double r = ref.f8()[i];
    if (std::isnan(r)) {
      REQUIRE_MESSAGE(std::isnan(got[i]), what << "[" << i << "]: expected NaN");
    } else if (std::isinf(r)) {
      REQUIRE_MESSAGE(got[i] == r, what << "[" << i << "]: expected " << r);
    } else {
      REQUIRE_MESSAGE(std::abs(got[i] - r) <= atol + rtol * std::abs(r),
                      what << "[" << i << "]: " << got[i] << " vs " << r
                           << " (diff " << std::abs(got[i] - r) << ")");
    }
  }
}

const char* kDatasets[] = {"plain", "single_col", "tiny", "ties", "large"};

}  // namespace

TEST_CASE("normalizers: ndtri matches scipy norm.ppf to the ULP") {
  if (!fixture_exists("normalizers/ndtri_p.npy")) {
    MESSAGE("normalizers fixtures missing; run tools/generate_fixtures.py");
    return;
  }
  auto p = load_fixture("normalizers/ndtri_p.npy");
  auto ref = load_fixture("normalizers/ndtri_ref.npy");
  int64_t exact = 0;
  for (int64_t i = 0; i < p.numel(); ++i) {
    const double got = tabicl::ndtri(p.f8()[i]);
    const double r = ref.f8()[i];
    if (std::memcmp(&got, &r, 8) == 0) exact++;
    REQUIRE_MESSAGE(std::abs(got - r) <= 1e-14 * std::max(1.0, std::abs(r)),
                    "ndtri(" << p.f8()[i] << ") = " << got << " vs " << r);
  }
  // Expect the overwhelming majority bitwise-identical.
  CHECK(static_cast<double>(exact) / static_cast<double>(p.numel()) > 0.999);
}

TEST_CASE("normalizers: StandardScaler matches sklearn") {
  if (!fixture_exists("normalizers/plain_X.npy")) return;
  for (const std::string name : kDatasets) {
    CAPTURE(name);
    Matrix X = to_matrix(load_fixture("normalizers/" + name + "_X.npy"));
    tabicl::SkStandardScaler ss;
    ss.fit(X);
    check_vals(ss.mean().data(), load_fixture("normalizers/" + name + "_std_mean.npy"),
               1e-12, 1e-13, name + "/std_mean");
    check_vals(ss.scale().data(), load_fixture("normalizers/" + name + "_std_scale.npy"),
               1e-12, 1e-13, name + "/std_scale");
    Matrix out = ss.transform(X);
    check_vals(out.data.data(), load_fixture("normalizers/" + name + "_std_out.npy"),
               1e-12, 1e-12, name + "/std_out");
  }
}

TEST_CASE("normalizers: Yeo-Johnson lambdas and transforms match sklearn") {
  if (!fixture_exists("normalizers/plain_X.npy")) return;
  for (const std::string name : kDatasets) {
    CAPTURE(name);
    Matrix X = to_matrix(load_fixture("normalizers/" + name + "_X.npy"));
    tabicl::PowerTransformerYJ pt;
    Matrix out = pt.fit_transform(X);
    // Lambda tolerance: fminbound xatol is 1.48e-8; libm pow ULP differences
    // can flip a Brent comparison, so iterates agree only to optimizer
    // tolerance, not bitwise. Output diffs scale with the lambda diff
    // (~1e-8 relative) — far below the fp32 cast at the model boundary.
    check_vals(pt.lambdas().data(),
               load_fixture("normalizers/" + name + "_yj_lambdas.npy"), 1e-7, 1e-7,
               name + "/yj_lambdas");
    check_vals(out.data.data(), load_fixture("normalizers/" + name + "_yj_out.npy"),
               2e-7, 2e-7, name + "/yj_out");
    Matrix out2 = pt.transform(X);
    check_vals(out2.data.data(), load_fixture("normalizers/" + name + "_yj_out2.npy"),
               2e-7, 2e-7, name + "/yj_out2");
    Matrix Xt = to_matrix(load_fixture("normalizers/" + name + "_Xtest.npy"));
    Matrix outt = pt.transform(Xt);
    check_vals(outt.data.data(),
               load_fixture("normalizers/" + name + "_yj_test_out.npy"), 2e-7, 2e-7,
               name + "/yj_test_out");
  }
}

TEST_CASE("normalizers: RobustScaler matches sklearn") {
  if (!fixture_exists("normalizers/plain_X.npy")) return;
  for (const std::string name : kDatasets) {
    CAPTURE(name);
    Matrix X = to_matrix(load_fixture("normalizers/" + name + "_X.npy"));
    tabicl::RobustScalerUV rs;
    rs.fit(X);
    check_vals(rs.center().data(),
               load_fixture("normalizers/" + name + "_robust_center.npy"), 1e-13, 1e-13,
               name + "/robust_center");
    check_vals(rs.scale().data(),
               load_fixture("normalizers/" + name + "_robust_scale.npy"), 1e-13, 1e-13,
               name + "/robust_scale");
    Matrix out = rs.transform(X);
    check_vals(out.data.data(), load_fixture("normalizers/" + name + "_robust_out.npy"),
               1e-12, 1e-12, name + "/robust_out");
  }
}

TEST_CASE("normalizers: QuantileTransformer (plain + RTDL) matches sklearn/tabicl") {
  if (!fixture_exists("normalizers/plain_X.npy")) return;
  for (const std::string name : kDatasets) {
    CAPTURE(name);
    Matrix X = to_matrix(load_fixture("normalizers/" + name + "_X.npy"));
    Matrix Xt = to_matrix(load_fixture("normalizers/" + name + "_Xtest.npy"));

    tabicl::QuantileTransformerNormal qt;
    qt.fit(X, /*n_quantiles=*/1000, /*subsample=*/10000, /*random_state=*/42);
    {
      auto ref = load_fixture("normalizers/" + name + "_qt_quantiles.npy");
      REQUIRE(qt.n_quantiles_fitted() == ref.shape[0]);
      check_vals(qt.quantiles().data(), ref, 1e-12, 1e-13, name + "/qt_quantiles");
    }
    Matrix out = qt.transform(X);
    check_vals(out.data.data(), load_fixture("normalizers/" + name + "_qt_out.npy"),
               1e-9, 1e-9, name + "/qt_out");
    Matrix outt = qt.transform(Xt);
    check_vals(outt.data.data(),
               load_fixture("normalizers/" + name + "_qt_test_out.npy"), 1e-9, 1e-9,
               name + "/qt_test_out");

    tabicl::RTDLQuantileTransformer rtdl;
    rtdl.fit(X, 42);
    Matrix rout = rtdl.transform(X);
    check_vals(rout.data.data(), load_fixture("normalizers/" + name + "_rtdl_out.npy"),
               1e-9, 1e-9, name + "/rtdl_out");
    Matrix routt = rtdl.transform(Xt);
    check_vals(routt.data.data(),
               load_fixture("normalizers/" + name + "_rtdl_test_out.npy"), 1e-9, 1e-9,
               name + "/rtdl_test_out");
  }
}
