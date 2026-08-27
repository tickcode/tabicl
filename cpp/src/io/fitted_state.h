// GGUF-backed serialization for fitted estimator state (C++-only format,
// versioned; not interoperable with Python pickles).
//
// FittedWriter collects key/value metadata and named tensors, then writes a
// single .gguf file. FittedReader opens one and hands back typed values;
// every getter throws on a missing key/tensor or a type mismatch, so loaders
// fail loudly on schema drift or corruption.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;

namespace tabicl {

class FittedWriter {
 public:
  FittedWriter();
  ~FittedWriter();
  FittedWriter(const FittedWriter&) = delete;
  FittedWriter& operator=(const FittedWriter&) = delete;

  void put_u32(const std::string& key, uint32_t v);
  void put_u64(const std::string& key, uint64_t v);
  void put_f64(const std::string& key, double v);
  void put_bool(const std::string& key, bool v);
  void put_str(const std::string& key, const std::string& v);

  void put_f64_tensor(const std::string& name, const std::vector<double>& v);
  void put_f32_tensor(const std::string& name, const std::vector<float>& v);
  void put_i32_tensor(const std::string& name, const std::vector<int32_t>& v);

  void write(const std::string& path);

 private:
  void add_tensor(const std::string& name, int type, const void* data,
                  int64_t count, size_t bytes);
  gguf_context* gctx_ = nullptr;
  // One small data-owning context per tensor; data must outlive write().
  std::vector<ggml_context*> ctxs_;
  std::vector<std::string> names_;  // stable storage for tensor names
};

class FittedReader {
 public:
  explicit FittedReader(const std::string& path);
  ~FittedReader();
  FittedReader(const FittedReader&) = delete;
  FittedReader& operator=(const FittedReader&) = delete;

  uint32_t get_u32(const std::string& key) const;
  uint64_t get_u64(const std::string& key) const;
  double get_f64(const std::string& key) const;
  bool get_bool(const std::string& key) const;
  std::string get_str(const std::string& key) const;
  bool has(const std::string& key) const;

  std::vector<double> get_f64_tensor(const std::string& name) const;
  std::vector<float> get_f32_tensor(const std::string& name) const;
  std::vector<int32_t> get_i32_tensor(const std::string& name) const;
  bool has_tensor(const std::string& name) const;

 private:
  const void* find(const std::string& name, int type, int64_t& n) const;
  gguf_context* gctx_ = nullptr;
  ggml_context* data_ctx_ = nullptr;
};

}  // namespace tabicl
