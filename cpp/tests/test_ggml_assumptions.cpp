// M0 assumption checks: every ggml behavior the TabICL port relies on.
// If any of these fail after a ggml submodule bump, the port's parity
// guarantees are void — fix the port or pin ggml back.
#include <cmath>
#include <cstring>
#include <vector>

#include "doctest.h"
#include "ggml-cpu.h"
#include "ggml.h"

namespace {

struct Ctx {
  ggml_context* ctx;
  explicit Ctx(size_t mb = 256) {
    ggml_init_params p{mb * 1024 * 1024, nullptr, false};
    ctx = ggml_init(p);
    REQUIRE(ctx != nullptr);
  }
  ~Ctx() { ggml_free(ctx); }
  operator ggml_context*() const { return ctx; }
};

void compute(ggml_context* ctx, ggml_tensor* out, int n_threads = 4) {
  ggml_cgraph* gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, out);
  REQUIRE(ggml_graph_compute_with_ctx(ctx, gf, n_threads) == GGML_STATUS_SUCCESS);
}

ggml_tensor* new_f32(ggml_context* ctx, std::vector<float> const& v,
                     int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1) {
  ggml_tensor* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
  REQUIRE(static_cast<int64_t>(v.size()) == ggml_nelements(t));
  std::memcpy(t->data, v.data(), v.size() * sizeof(float));
  return t;
}

std::vector<float> to_vec(const ggml_tensor* t) {
  std::vector<float> v(ggml_nelements(t));
  std::memcpy(v.data(), t->data, v.size() * sizeof(float));
  return v;
}

}  // namespace

TEST_CASE("ggml: GGML_MAX_NAME is 64") {
  // Export script's tensor-name shortening map assumes this limit.
  CHECK(GGML_MAX_NAME == 64);
}

TEST_CASE("ggml: gelu_erf matches exact erf formula") {
  std::vector<float> xs;
  for (float x = -8.0f; x <= 8.0f; x += 0.037f) xs.push_back(x);
  Ctx ctx;
  ggml_tensor* x = new_f32(ctx, xs, static_cast<int64_t>(xs.size()));
  ggml_tensor* y = ggml_gelu_erf(ctx, x);
  compute(ctx, y);
  auto out = to_vec(y);
  for (size_t i = 0; i < xs.size(); ++i) {
    const double ref = 0.5 * xs[i] * (1.0 + std::erf(xs[i] / std::sqrt(2.0)));
    CHECK(std::abs(out[i] - ref) <= 3e-7 + 2e-7 * std::abs(ref));
  }
}

TEST_CASE("ggml: tanh is LUT-free precise") {
  std::vector<float> xs;
  for (float x = -6.0f; x <= 6.0f; x += 0.041f) xs.push_back(x);
  Ctx ctx;
  ggml_tensor* x = new_f32(ctx, xs, static_cast<int64_t>(xs.size()));
  ggml_tensor* y = ggml_tanh(ctx, x);
  compute(ctx, y);
  auto out = to_vec(y);
  for (size_t i = 0; i < xs.size(); ++i) {
    const double ref = std::tanh(static_cast<double>(xs[i]));
    CHECK(std::abs(out[i] - ref) <= 2e-7);
  }
}

TEST_CASE("ggml: add broadcasts bias (d,1,1,1) over (d,n,b2,b3)") {
  const int64_t d = 3, n = 2, b2 = 2, b3 = 2;
  std::vector<float> xs(d * n * b2 * b3);
  for (size_t i = 0; i < xs.size(); ++i) xs[i] = static_cast<float>(i);
  std::vector<float> bias = {10.f, 20.f, 30.f};
  Ctx ctx;
  ggml_tensor* x = new_f32(ctx, xs, d, n, b2, b3);
  ggml_tensor* b = new_f32(ctx, bias, d);
  ggml_tensor* y = ggml_add(ctx, x, b);
  compute(ctx, y);
  auto out = to_vec(y);
  for (size_t i = 0; i < xs.size(); ++i) CHECK(out[i] == xs[i] + bias[i % d]);
}

TEST_CASE("ggml: mul_mat convention C[i,j] = dot(A_row_i, B_row_j)") {
  // A (k=2, m=3), B (k=2, n=2) -> C (m=3, n=2)
  std::vector<float> A = {1, 2,   3, 4,   5, 6};   // rows: (1,2),(3,4),(5,6)
  std::vector<float> B = {10, 100,   20, 200};     // rows: (10,100),(20,200)
  Ctx ctx;
  ggml_tensor* a = new_f32(ctx, A, 2, 3);
  ggml_tensor* b = new_f32(ctx, B, 2, 2);
  ggml_tensor* c = ggml_mul_mat(ctx, a, b);
  compute(ctx, c);
  CHECK(c->ne[0] == 3);
  CHECK(c->ne[1] == 2);
  auto out = to_vec(c);
  const float ref[6] = {210, 430, 650, 420, 860, 1300};
  for (int i = 0; i < 6; ++i) CHECK(out[i] == ref[i]);
}

TEST_CASE("ggml: mul_mat broadcasts A over batch dims ne2/ne3") {
  // Shared weight A (k=2, m=2, 1, 1) applied to B (k=2, n=1, b2=2, b3=3)
  std::vector<float> A = {1, 0, 0, 1};  // identity rows
  std::vector<float> B(2 * 1 * 2 * 3);
  for (size_t i = 0; i < B.size(); ++i) B[i] = static_cast<float>(i + 1);
  Ctx ctx;
  ggml_tensor* a = new_f32(ctx, A, 2, 2);
  ggml_tensor* b = new_f32(ctx, B, 2, 1, 2, 3);
  ggml_tensor* c = ggml_mul_mat(ctx, a, b);
  compute(ctx, c);
  CHECK(c->ne[0] == 2);
  CHECK(c->ne[1] == 1);
  CHECK(c->ne[2] == 2);
  CHECK(c->ne[3] == 3);
  auto out = to_vec(c);
  for (size_t i = 0; i < B.size(); ++i) CHECK(out[i] == B[i]);  // identity per batch
}

TEST_CASE("ggml: soft_max_ext(scale) == softmax(scale*x) along ne0, max-subtracted") {
  const float scale = 0.25f;
  std::vector<float> xs = {1.f, 2.f, 3.f, 4.f,  -1.f, 0.f, 100.f, 100.f};
  Ctx ctx;
  ggml_tensor* x = new_f32(ctx, xs, 4, 2);
  ggml_tensor* y = ggml_soft_max_ext(ctx, x, nullptr, scale, 0.0f);
  compute(ctx, y);
  auto out = to_vec(y);
  for (int r = 0; r < 2; ++r) {
    double m = -1e30;
    for (int i = 0; i < 4; ++i) m = std::max(m, static_cast<double>(scale * xs[r * 4 + i]));
    double denom = 0;
    std::vector<double> e(4);
    for (int i = 0; i < 4; ++i) { e[i] = std::exp(scale * xs[r * 4 + i] - m); denom += e[i]; }
    double sum = 0;
    for (int i = 0; i < 4; ++i) {
      CHECK(std::abs(out[r * 4 + i] - e[i] / denom) <= 1e-6);
      sum += out[r * 4 + i];
    }
    CHECK(std::abs(sum - 1.0) <= 1e-6);
  }
}

TEST_CASE("ggml: view/neg/concat implement rotate_half on ne0") {
  // rot(x) = concat(-x[hi], x[lo]) along ne0, per row: the RoPE building block.
  const int64_t d = 4, n = 3;
  std::vector<float> xs(d * n);
  for (size_t i = 0; i < xs.size(); ++i) xs[i] = static_cast<float>(i + 1);
  Ctx ctx;
  ggml_tensor* x = new_f32(ctx, xs, d, n);
  ggml_tensor* lo = ggml_view_2d(ctx, x, d / 2, n, x->nb[1], 0);
  ggml_tensor* hi = ggml_view_2d(ctx, x, d / 2, n, x->nb[1], (d / 2) * sizeof(float));
  ggml_tensor* rot = ggml_concat(ctx, ggml_neg(ctx, ggml_cont(ctx, hi)),
                                 ggml_cont(ctx, lo), 0);
  compute(ctx, rot);
  auto out = to_vec(rot);
  for (int64_t r = 0; r < n; ++r) {
    const float* row = xs.data() + r * d;
    CHECK(out[r * d + 0] == -row[2]);
    CHECK(out[r * d + 1] == -row[3]);
    CHECK(out[r * d + 2] == row[0]);
    CHECK(out[r * d + 3] == row[1]);
  }
}

TEST_CASE("ggml: get_rows gathers rows by i32 index") {
  std::vector<float> xs = {1, 2,  3, 4,  5, 6};  // 3 rows of 2
  Ctx ctx;
  ggml_tensor* x = new_f32(ctx, xs, 2, 3);
  ggml_tensor* idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
  const int32_t iv[4] = {2, 0, 2, 1};
  std::memcpy(idx->data, iv, sizeof(iv));
  ggml_tensor* y = ggml_get_rows(ctx, x, idx);
  compute(ctx, y);
  auto out = to_vec(y);
  const float ref[8] = {5, 6, 1, 2, 5, 6, 3, 4};
  for (int i = 0; i < 8; ++i) CHECK(out[i] == ref[i]);
}

TEST_CASE("ggml: norm(eps) == (x-mean)/sqrt(var+eps), biased var, over ne0") {
  const float eps = 1e-5f;
  std::vector<float> xs = {1.f, 2.f, 3.f, 4.f, -2.f, 0.f, 5.f, 1.f};
  Ctx ctx;
  ggml_tensor* x = new_f32(ctx, xs, 4, 2);
  ggml_tensor* y = ggml_norm(ctx, x, eps);
  compute(ctx, y);
  auto out = to_vec(y);
  for (int r = 0; r < 2; ++r) {
    double mean = 0, var = 0;
    for (int i = 0; i < 4; ++i) mean += xs[r * 4 + i];
    mean /= 4;
    for (int i = 0; i < 4; ++i) var += (xs[r * 4 + i] - mean) * (xs[r * 4 + i] - mean);
    var /= 4;  // biased, matching torch LayerNorm
    for (int i = 0; i < 4; ++i) {
      const double ref = (xs[r * 4 + i] - mean) / std::sqrt(var + eps);
      CHECK(std::abs(out[i + r * 4] - ref) <= 1e-6);
    }
  }
}

TEST_CASE("ggml: mul_mat is bitwise deterministic across thread counts") {
  const int64_t k = 96, m = 64, n = 512;
  std::vector<float> A(k * m), B(k * n);
  // Deterministic pseudo-random fill (no libc rand dependency)
  uint32_t s = 123456789;
  auto next = [&s]() {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return static_cast<float>(static_cast<int32_t>(s % 2000) - 1000) / 997.0f;
  };
  for (auto& v : A) v = next();
  for (auto& v : B) v = next();

  std::vector<float> ref;
  for (int n_threads : {1, 4, 8}) {
    Ctx ctx;
    ggml_tensor* a = new_f32(ctx, A, k, m);
    ggml_tensor* b = new_f32(ctx, B, k, n);
    ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    compute(ctx, c, n_threads);
    auto out = to_vec(c);
    if (ref.empty()) {
      ref = out;
    } else {
      REQUIRE(ref.size() == out.size());
      size_t mismatches = 0;
      for (size_t i = 0; i < out.size(); ++i)
        if (std::memcmp(&ref[i], &out[i], sizeof(float)) != 0) mismatches++;
      CHECK(mismatches == 0);
    }
  }
}

TEST_CASE("ggml: soft_max and norm are bitwise deterministic across thread counts") {
  const int64_t d = 517, rows = 64;
  std::vector<float> X(d * rows);
  uint32_t s = 42424242;
  for (auto& v : X) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    v = static_cast<float>(static_cast<int32_t>(s % 4000) - 2000) / 61.0f;
  }
  for (bool use_norm : {false, true}) {
    std::vector<float> ref;
    for (int n_threads : {1, 8}) {
      Ctx ctx;
      ggml_tensor* x = new_f32(ctx, X, d, rows);
      ggml_tensor* y = use_norm ? ggml_norm(ctx, x, 1e-5f)
                                : ggml_soft_max_ext(ctx, x, nullptr, 0.125f, 0.0f);
      compute(ctx, y, n_threads);
      auto out = to_vec(y);
      if (ref.empty()) {
        ref = out;
      } else {
        size_t mismatches = 0;
        for (size_t i = 0; i < out.size(); ++i)
          if (std::memcmp(&ref[i], &out[i], sizeof(float)) != 0) mismatches++;
        CHECK(mismatches == 0);
      }
    }
  }
}
