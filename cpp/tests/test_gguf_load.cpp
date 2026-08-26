#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "doctest.h"
#include "ggml.h"
#include "io/sha256.h"
#include "tabicl/model.h"

#ifndef TABICL_FIXTURES_DIR
#define TABICL_FIXTURES_DIR "fixtures"
#endif

namespace {

std::string fixture(const std::string& name) {
  const char* env = std::getenv("TABICL_FIXTURES_DIR");
  return (env ? std::string(env) : std::string(TABICL_FIXTURES_DIR)) + "/" + name;
}

bool exists(const std::string& path) {
  return std::ifstream(path).good();
}

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE(f.good());
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Extract manifest["tensors"][name]["sha256"] with plain string scanning
// (the manifest is machine-written by scripts/export_gguf.py).
std::string manifest_sha(const std::string& manifest, const std::string& name) {
  const std::string anchor = "\"" + name + "\": {";
  size_t p = manifest.find(anchor);
  REQUIRE_MESSAGE(p != std::string::npos, "manifest missing tensor " << name);
  const std::string key = "\"sha256\": \"";
  size_t q = manifest.find(key, p);
  REQUIRE(q != std::string::npos);
  q += key.size();
  return manifest.substr(q, 64);
}

}  // namespace

TEST_CASE("gguf: classifier loads, hashes match manifest, config correct") {
  const std::string path = fixture("tabicl-classifier-v2.gguf");
  if (!exists(path)) {
    MESSAGE("fixture missing, skipping (run scripts/export_gguf.py): " << path);
    return;
  }
  auto model = tabicl::Model::load(path);
  const tabicl::ModelConfig& c = model->config();
  CHECK(c.task == tabicl::Task::Classification);
  CHECK(c.max_classes == 10);
  CHECK(c.embed_dim == 128);
  CHECK(c.col_num_blocks == 3);
  CHECK(c.row_num_cls == 4);
  CHECK(c.icl_num_blocks == 12);
  CHECK(c.icl_dim() == 512);
  CHECK(c.bias_free_ln == false);
  CHECK(model->n_tensors() == 391);

  const std::string manifest = slurp(fixture("tabicl-classifier-v2.manifest.json"));
  // Hash every tensor's raw bytes against the export manifest.
  size_t checked = 0;
  for (const char* name :
       {"col.in_linear.weight", "col.blk.0.ind_vectors", "col.blk.2.attn1.in_proj_weight",
        "col.blk.1.attn1.ssmax.base_mlp.2.weight", "row.cls_tokens", "row.rope.freqs",
        "row.blk.2.attn.out_proj.weight", "icl.blk.0.attn.in_proj_weight",
        "icl.blk.11.linear2.weight", "icl.decoder.2.weight", "icl.y_encoder.weight",
        "icl.ln.weight"}) {
    ggml_tensor* t = model->tensor(name);
    CHECK(tabicl::sha256_hex(t->data, ggml_nbytes(t)) == manifest_sha(manifest, name));
    checked++;
  }
  CHECK(checked == 12);

  // RoPE freqs: informational check against the closed-form 1/theta^(2i/d).
  // The checkpoint stores fp32 values, so agreement is only to fp32 rounding —
  // which is why the port always builds its cos/sin tables from the tensor,
  // never from the formula.
  {
    ggml_tensor* f = model->tensor("row.rope.freqs");
    REQUIRE(f->ne[0] == 8);
    const float* fd = static_cast<const float*>(f->data);
    for (int i = 0; i < 8; ++i) {
      const double ref = 1.0 / std::pow(100000.0, (2.0 * i) / 16.0);
      CHECK(std::abs(fd[i] - ref) <= 1e-7 * std::abs(ref) + 1e-12);
    }
  }
}

TEST_CASE("gguf: regressor loads with bias-free LN and quantile head") {
  const std::string path = fixture("tabicl-regressor-v2.gguf");
  if (!exists(path)) {
    MESSAGE("fixture missing, skipping (run scripts/export_gguf.py): " << path);
    return;
  }
  auto model = tabicl::Model::load(path);
  const tabicl::ModelConfig& c = model->config();
  CHECK(c.task == tabicl::Task::Regression);
  CHECK(c.max_classes == 0);
  CHECK(c.num_quantiles == 999);
  CHECK(c.bias_free_ln == true);
  CHECK(model->n_tensors() == 347);
  CHECK(model->tensor_or_null("icl.ln.bias") == nullptr);
  CHECK(model->tensor_or_null("row.blk.0.norm1.bias") == nullptr);
  ggml_tensor* dec = model->tensor("icl.decoder.2.weight");
  CHECK(dec->ne[0] == 1024);
  CHECK(dec->ne[1] == 999);
  ggml_tensor* ye = model->tensor("icl.y_encoder.weight");
  CHECK(ye->ne[0] == 1);
  CHECK(ye->ne[1] == 512);

  const std::string manifest = slurp(fixture("tabicl-regressor-v2.manifest.json"));
  for (const char* name : {"icl.decoder.2.weight", "icl.y_encoder.weight", "col.in_linear.weight"}) {
    ggml_tensor* t = model->tensor(name);
    CHECK(tabicl::sha256_hex(t->data, ggml_nbytes(t)) == manifest_sha(manifest, name));
  }
}

TEST_CASE("gguf: loading a non-gguf file fails cleanly") {
  CHECK_THROWS(tabicl::Model::load("/nonexistent/path/model.gguf"));
}
