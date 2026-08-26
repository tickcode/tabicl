#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "tabicl/model.h"

namespace tabicl {

namespace {

uint32_t get_u32(const gguf_context* g, const char* key) {
  int64_t id = gguf_find_key(g, key);
  if (id < 0) throw std::runtime_error(std::string("gguf: missing key ") + key);
  return gguf_get_val_u32(g, id);
}

float get_f32(const gguf_context* g, const char* key) {
  int64_t id = gguf_find_key(g, key);
  if (id < 0) throw std::runtime_error(std::string("gguf: missing key ") + key);
  return gguf_get_val_f32(g, id);
}

bool get_bool(const gguf_context* g, const char* key) {
  int64_t id = gguf_find_key(g, key);
  if (id < 0) throw std::runtime_error(std::string("gguf: missing key ") + key);
  return gguf_get_val_bool(g, id);
}

std::string get_str(const gguf_context* g, const char* key) {
  int64_t id = gguf_find_key(g, key);
  if (id < 0) throw std::runtime_error(std::string("gguf: missing key ") + key);
  return gguf_get_val_str(g, id);
}

// The tensors every checkpoint must provide, parameterized by config.
// (Presence + fp32 dtype is validated; shapes are checked where they gate
// pointer arithmetic in the stages.)
std::vector<std::string> expected_tensors(const ModelConfig& c) {
  std::vector<std::string> names;
  auto block_common = [&](const std::string& p, bool ssmax) {
    names.push_back(p + "in_proj_weight");
    names.push_back(p + "in_proj_bias");
    names.push_back(p + "out_proj.weight");
    names.push_back(p + "out_proj.bias");
    if (ssmax) {
      for (const char* m : {"base_mlp", "query_mlp"}) {
        for (const char* i : {"0", "2"}) {
          names.push_back(p + "ssmax." + m + "." + i + ".weight");
          names.push_back(p + "ssmax." + m + "." + i + ".bias");
        }
      }
    }
  };
  names.push_back("col.in_linear.weight");
  names.push_back("col.in_linear.bias");
  names.push_back("col.y_encoder.weight");
  names.push_back("col.y_encoder.bias");
  for (int b = 0; b < c.col_num_blocks; ++b) {
    std::string blk = "col.blk." + std::to_string(b) + ".";
    names.push_back(blk + "ind_vectors");
    for (int a = 1; a <= 2; ++a) {
      std::string ap = blk + "attn" + std::to_string(a) + ".";
      block_common(ap, a == 1);
      for (const char* mod : {"linear1", "linear2"}) {
        names.push_back(ap + mod + ".weight");
        names.push_back(ap + mod + ".bias");
      }
      for (const char* mod : {"norm1", "norm2"}) {
        names.push_back(ap + mod + ".weight");
        if (!c.bias_free_ln) names.push_back(ap + mod + ".bias");
      }
    }
  }

  names.push_back("row.cls_tokens");
  names.push_back("row.out_ln.weight");
  if (!c.bias_free_ln) names.push_back("row.out_ln.bias");
  names.push_back("row.rope.freqs");
  for (int b = 0; b < c.row_num_blocks; ++b) {
    std::string blk = "row.blk." + std::to_string(b) + ".attn.";
    block_common(blk, false);
    std::string p = "row.blk." + std::to_string(b) + ".";
    for (const char* mod : {"linear1", "linear2"}) {
      names.push_back(p + mod + ".weight");
      names.push_back(p + mod + ".bias");
    }
    for (const char* mod : {"norm1", "norm2"}) {
      names.push_back(p + mod + ".weight");
      if (!c.bias_free_ln) names.push_back(p + mod + ".bias");
    }
  }

  names.push_back("icl.y_encoder.weight");
  names.push_back("icl.y_encoder.bias");
  names.push_back("icl.ln.weight");
  if (!c.bias_free_ln) names.push_back("icl.ln.bias");
  names.push_back("icl.decoder.0.weight");
  names.push_back("icl.decoder.0.bias");
  names.push_back("icl.decoder.2.weight");
  names.push_back("icl.decoder.2.bias");
  for (int b = 0; b < c.icl_num_blocks; ++b) {
    std::string blk = "icl.blk." + std::to_string(b) + ".attn.";
    block_common(blk, true);
    std::string p = "icl.blk." + std::to_string(b) + ".";
    for (const char* mod : {"linear1", "linear2"}) {
      names.push_back(p + mod + ".weight");
      names.push_back(p + mod + ".bias");
    }
    for (const char* mod : {"norm1", "norm2"}) {
      names.push_back(p + mod + ".weight");
      if (!c.bias_free_ln) names.push_back(p + mod + ".bias");
    }
  }
  return names;
}

}  // namespace

Model::~Model() {
  if (weights_ctx_) ggml_free(weights_ctx_);
}

ggml_tensor* Model::tensor(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end())
    throw std::runtime_error("model: missing tensor " + name);
  return it->second;
}

ggml_tensor* Model::tensor_or_null(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : it->second;
}

std::shared_ptr<Model> Model::load(const std::string& gguf_path) {
  auto model = std::shared_ptr<Model>(new Model());

  ggml_context* data_ctx = nullptr;
  gguf_init_params params{/*no_alloc=*/false, /*ctx=*/&data_ctx};
  gguf_context* g = gguf_init_from_file(gguf_path.c_str(), params);
  if (!g) throw std::runtime_error("gguf: failed to load " + gguf_path);

  try {
    if (get_str(g, "general.architecture") != "tabicl")
      throw std::runtime_error("gguf: not a tabicl model");
    uint32_t fmt = get_u32(g, "tabicl.format_version");
    if (fmt != 1)
      throw std::runtime_error("gguf: unsupported tabicl.format_version " + std::to_string(fmt));

    ModelConfig& c = model->config_;
    const std::string task = get_str(g, "tabicl.task");
    if (task == "classification") c.task = Task::Classification;
    else if (task == "regression") c.task = Task::Regression;
    else throw std::runtime_error("gguf: unknown task " + task);
    c.max_classes = static_cast<int>(get_u32(g, "tabicl.max_classes"));
    c.num_quantiles = static_cast<int>(get_u32(g, "tabicl.num_quantiles"));
    c.embed_dim = static_cast<int>(get_u32(g, "tabicl.embed_dim"));
    c.col_num_blocks = static_cast<int>(get_u32(g, "tabicl.col.num_blocks"));
    c.col_num_heads = static_cast<int>(get_u32(g, "tabicl.col.num_heads"));
    c.col_num_inds = static_cast<int>(get_u32(g, "tabicl.col.num_inds"));
    c.col_feature_group_size = static_cast<int>(get_u32(g, "tabicl.col.feature_group_size"));
    c.row_num_blocks = static_cast<int>(get_u32(g, "tabicl.row.num_blocks"));
    c.row_num_heads = static_cast<int>(get_u32(g, "tabicl.row.num_heads"));
    c.row_num_cls = static_cast<int>(get_u32(g, "tabicl.row.num_cls"));
    c.row_rope_base = get_f32(g, "tabicl.row.rope_base");
    c.icl_num_blocks = static_cast<int>(get_u32(g, "tabicl.icl.num_blocks"));
    c.icl_num_heads = static_cast<int>(get_u32(g, "tabicl.icl.num_heads"));
    c.ff_factor = static_cast<int>(get_u32(g, "tabicl.ff_factor"));
    c.bias_free_ln = get_bool(g, "tabicl.bias_free_ln");
    c.norm_eps = get_f32(g, "tabicl.norm_eps");
    c.softmax_temperature = get_f32(g, "tabicl.softmax_temperature");

    if ((c.task == Task::Regression) != (c.max_classes == 0))
      throw std::runtime_error("gguf: task/max_classes mismatch");

    model->weights_ctx_ = data_ctx;
    // Iterate the gguf tensor list (the context also holds an internal
    // "binary blob" buffer tensor that must not be treated as a weight).
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
      const char* name = gguf_get_tensor_name(g, i);
      ggml_tensor* t = ggml_get_tensor(data_ctx, name);
      if (!t) throw std::runtime_error(std::string("gguf: tensor not in ctx: ") + name);
      if (t->type != GGML_TYPE_F32)
        throw std::runtime_error(std::string("gguf: non-f32 tensor ") + name);
      model->tensors_.emplace(name, t);
    }

    for (const std::string& name : expected_tensors(c)) {
      if (model->tensors_.find(name) == model->tensors_.end())
        throw std::runtime_error("gguf: checkpoint missing expected tensor " + name);
    }

    // Spot-check shapes that the stage code relies on for pointer math.
    const auto dim_is = [&](const char* name, int64_t ne0, int64_t ne1) {
      ggml_tensor* t = model->tensor(name);
      if (t->ne[0] != ne0 || (ne1 >= 0 && t->ne[1] != ne1))
        throw std::runtime_error(std::string("gguf: unexpected shape for ") + name);
    };
    dim_is("col.in_linear.weight", c.col_feature_group_size, c.embed_dim);
    dim_is("col.blk.0.ind_vectors", c.embed_dim, c.col_num_inds);
    dim_is("row.cls_tokens", c.embed_dim, c.row_num_cls);
    dim_is("row.rope.freqs", (c.embed_dim / c.row_num_heads) / 2, -1);
    dim_is("icl.decoder.0.weight", c.icl_dim(), c.icl_dim() * c.ff_factor);
    const int out_dim = c.task == Task::Regression ? c.num_quantiles : c.max_classes;
    dim_is("icl.decoder.2.weight", c.icl_dim() * c.ff_factor, out_dim);
  } catch (...) {
    gguf_free(g);
    // weights_ctx_ freed by destructor if set; if not set yet, free data_ctx here
    if (!model->weights_ctx_ && data_ctx) ggml_free(data_ctx);
    throw;
  }
  gguf_free(g);
  return model;
}

}  // namespace tabicl
