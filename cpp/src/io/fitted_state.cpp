#include "io/fitted_state.h"

#include <cstring>
#include <stdexcept>

#include "ggml.h"
#include "gguf.h"

namespace tabicl {

// ---------------------------------------------------------------------------
// FittedWriter
// ---------------------------------------------------------------------------

FittedWriter::FittedWriter() {
  gctx_ = gguf_init_empty();
  if (!gctx_) throw std::runtime_error("fitted: gguf_init_empty failed");
}

FittedWriter::~FittedWriter() {
  if (gctx_) gguf_free(gctx_);
  for (ggml_context* c : ctxs_) ggml_free(c);
}

void FittedWriter::put_u32(const std::string& key, uint32_t v) {
  gguf_set_val_u32(gctx_, key.c_str(), v);
}
void FittedWriter::put_u64(const std::string& key, uint64_t v) {
  gguf_set_val_u64(gctx_, key.c_str(), v);
}
void FittedWriter::put_f64(const std::string& key, double v) {
  gguf_set_val_f64(gctx_, key.c_str(), v);
}
void FittedWriter::put_bool(const std::string& key, bool v) {
  gguf_set_val_bool(gctx_, key.c_str(), v);
}
void FittedWriter::put_str(const std::string& key, const std::string& v) {
  gguf_set_val_str(gctx_, key.c_str(), v.c_str());
}

void FittedWriter::add_tensor(const std::string& name, int type, const void* data,
                              int64_t count, size_t bytes) {
  ggml_init_params p{bytes + ggml_tensor_overhead() + 256, nullptr,
                     /*no_alloc=*/false};
  ggml_context* ctx = ggml_init(p);
  if (!ctx) throw std::runtime_error("fitted: tensor ctx alloc failed");
  ctxs_.push_back(ctx);
  ggml_tensor* t =
      ggml_new_tensor_1d(ctx, static_cast<ggml_type>(type), std::max<int64_t>(count, 1));
  if (count > 0) std::memcpy(t->data, data, bytes);
  names_.push_back(name);
  ggml_set_name(t, names_.back().c_str());
  gguf_add_tensor(gctx_, t);
}

void FittedWriter::put_f64_tensor(const std::string& name,
                                  const std::vector<double>& v) {
  add_tensor(name, GGML_TYPE_F64, v.data(), static_cast<int64_t>(v.size()),
             v.size() * sizeof(double));
}
void FittedWriter::put_f32_tensor(const std::string& name,
                                  const std::vector<float>& v) {
  add_tensor(name, GGML_TYPE_F32, v.data(), static_cast<int64_t>(v.size()),
             v.size() * sizeof(float));
}
void FittedWriter::put_i32_tensor(const std::string& name,
                                  const std::vector<int32_t>& v) {
  add_tensor(name, GGML_TYPE_I32, v.data(), static_cast<int64_t>(v.size()),
             v.size() * sizeof(int32_t));
}

void FittedWriter::write(const std::string& path) {
  if (!gguf_write_to_file(gctx_, path.c_str(), /*only_meta=*/false))
    throw std::runtime_error("fitted: failed to write " + path);
}

// ---------------------------------------------------------------------------
// FittedReader
// ---------------------------------------------------------------------------

FittedReader::FittedReader(const std::string& path) {
  gguf_init_params params{/*no_alloc=*/false, /*ctx=*/&data_ctx_};
  gctx_ = gguf_init_from_file(path.c_str(), params);
  if (!gctx_) throw std::runtime_error("fitted: cannot open " + path);
}

FittedReader::~FittedReader() {
  if (gctx_) gguf_free(gctx_);
  if (data_ctx_) ggml_free(data_ctx_);
}

namespace {
int64_t require_key(const gguf_context* g, const std::string& key) {
  const int64_t id = gguf_find_key(g, key.c_str());
  if (id < 0) throw std::runtime_error("fitted: missing key " + key);
  return id;
}
}  // namespace

uint32_t FittedReader::get_u32(const std::string& key) const {
  return gguf_get_val_u32(gctx_, require_key(gctx_, key));
}
uint64_t FittedReader::get_u64(const std::string& key) const {
  return gguf_get_val_u64(gctx_, require_key(gctx_, key));
}
double FittedReader::get_f64(const std::string& key) const {
  return gguf_get_val_f64(gctx_, require_key(gctx_, key));
}
bool FittedReader::get_bool(const std::string& key) const {
  return gguf_get_val_bool(gctx_, require_key(gctx_, key));
}
std::string FittedReader::get_str(const std::string& key) const {
  return gguf_get_val_str(gctx_, require_key(gctx_, key));
}
bool FittedReader::has(const std::string& key) const {
  return gguf_find_key(gctx_, key.c_str()) >= 0;
}

bool FittedReader::has_tensor(const std::string& name) const {
  return data_ctx_ && ggml_get_tensor(data_ctx_, name.c_str()) != nullptr;
}

const void* FittedReader::find(const std::string& name, int type, int64_t& n) const {
  ggml_tensor* t = data_ctx_ ? ggml_get_tensor(data_ctx_, name.c_str()) : nullptr;
  if (!t) throw std::runtime_error("fitted: missing tensor " + name);
  if (t->type != type)
    throw std::runtime_error("fitted: tensor type mismatch for " + name);
  n = ggml_nelements(t);
  return t->data;
}

std::vector<double> FittedReader::get_f64_tensor(const std::string& name) const {
  int64_t n = 0;
  const auto* p = static_cast<const double*>(find(name, GGML_TYPE_F64, n));
  return std::vector<double>(p, p + n);
}
std::vector<float> FittedReader::get_f32_tensor(const std::string& name) const {
  int64_t n = 0;
  const auto* p = static_cast<const float*>(find(name, GGML_TYPE_F32, n));
  return std::vector<float>(p, p + n);
}
std::vector<int32_t> FittedReader::get_i32_tensor(const std::string& name) const {
  int64_t n = 0;
  const auto* p = static_cast<const int32_t*>(find(name, GGML_TYPE_I32, n));
  return std::vector<int32_t>(p, p + n);
}

}  // namespace tabicl
