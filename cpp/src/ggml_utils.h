// Graph-execution utility for the TabICL stages: build a ggml graph in a
// no-alloc context, allocate activations with a reused gallocr, fill inputs,
// compute on the CPU backend with a configurable thread count.
//
// Usage:
//   GraphRunner runner(n_threads);
//   GraphExec ge(runner, /*graph_size=*/8192);
//   ggml_tensor* x = ge.input(GGML_TYPE_F32, dim, seq);
//   ggml_tensor* out = ...build ops...(ge.ctx(), x, weights...);
//   ge.finalize(out);              // allocates all activations
//   ge.set(x, host_ptr);           // memcpy into x->data
//   ge.compute();                  // out->data now valid until ge destructs
#pragma once

#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
typedef struct ggml_gallocr* ggml_gallocr_t;

namespace tabicl {

class GraphRunner {
 public:
  explicit GraphRunner(int n_threads);
  ~GraphRunner();
  GraphRunner(const GraphRunner&) = delete;
  GraphRunner& operator=(const GraphRunner&) = delete;

  int n_threads() const { return n_threads_; }
  ggml_gallocr_t galloc() { return galloc_; }

 private:
  int n_threads_;
  ggml_gallocr_t galloc_;
};

class GraphExec {
 public:
  // graph_size: max node count for this graph.
  GraphExec(GraphRunner& runner, size_t graph_size = 8192);
  ~GraphExec();
  GraphExec(const GraphExec&) = delete;
  GraphExec& operator=(const GraphExec&) = delete;

  ggml_context* ctx() { return ctx_; }

  // New input tensor (fp32 unless type given); data becomes valid after
  // finalize(). ne given in ggml order (ne0 fastest).
  ggml_tensor* input(int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1);
  ggml_tensor* input_i32(int64_t ne0);

  // Mark outputs and allocate the graph. May be called once.
  void finalize(ggml_tensor* out);
  void finalize(const std::vector<ggml_tensor*>& outs);

  // Copy host fp32/i32 data into an input tensor (after finalize()).
  void set(ggml_tensor* t, const void* host_data);

  void compute();

 private:
  GraphRunner& runner_;
  ggml_context* ctx_ = nullptr;
  ggml_cgraph* gf_ = nullptr;
  bool finalized_ = false;
};

}  // namespace tabicl
