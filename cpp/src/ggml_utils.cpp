#include "ggml_utils.h"

#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

namespace tabicl {

GraphRunner::GraphRunner(int n_threads) {
  n_threads_ = n_threads > 0
                   ? n_threads
                   : static_cast<int>(std::thread::hardware_concurrency());
  if (n_threads_ <= 0) n_threads_ = 4;
  galloc_ = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
  if (!galloc_) throw std::runtime_error("ggml: gallocr init failed");
}

GraphRunner::~GraphRunner() { ggml_gallocr_free(galloc_); }

GraphExec::GraphExec(GraphRunner& runner, size_t graph_size) : runner_(runner) {
  // Metadata-only context: tensors + graph structure, no data.
  const size_t mem =
      ggml_tensor_overhead() * (graph_size + 64) +
      ggml_graph_overhead_custom(graph_size, /*grads=*/false) + (1u << 16);
  ggml_init_params params{mem, nullptr, /*no_alloc=*/true};
  ctx_ = ggml_init(params);
  if (!ctx_) throw std::runtime_error("ggml: ctx init failed");
  gf_ = ggml_new_graph_custom(ctx_, graph_size, /*grads=*/false);
}

GraphExec::~GraphExec() {
  if (ctx_) ggml_free(ctx_);
}

ggml_tensor* GraphExec::input(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
  ggml_tensor* t = ggml_new_tensor_4d(ctx_, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
  ggml_set_input(t);
  return t;
}

ggml_tensor* GraphExec::input_i32(int64_t ne0) {
  ggml_tensor* t = ggml_new_tensor_1d(ctx_, GGML_TYPE_I32, ne0);
  ggml_set_input(t);
  return t;
}

void GraphExec::finalize(ggml_tensor* out) {
  finalize(std::vector<ggml_tensor*>{out});
}

void GraphExec::finalize(const std::vector<ggml_tensor*>& outs) {
  if (finalized_) throw std::runtime_error("GraphExec: already finalized");
  for (ggml_tensor* o : outs) {
    ggml_set_output(o);
    ggml_build_forward_expand(gf_, o);
  }
  if (!ggml_gallocr_alloc_graph(runner_.galloc(), gf_))
    throw std::runtime_error("ggml: graph allocation failed");
  finalized_ = true;
}

void GraphExec::set(ggml_tensor* t, const void* host_data) {
  if (!finalized_) throw std::runtime_error("GraphExec: set before finalize");
  std::memcpy(t->data, host_data, ggml_nbytes(t));
}

void GraphExec::compute() {
  if (!finalized_) throw std::runtime_error("GraphExec: compute before finalize");
  ggml_cplan plan = ggml_graph_plan(gf_, runner_.n_threads(), nullptr);
  std::vector<uint8_t> work(plan.work_size);
  plan.work_data = work.data();
  if (ggml_graph_compute(gf_, &plan) != GGML_STATUS_SUCCESS)
    throw std::runtime_error("ggml: graph compute failed");
}

}  // namespace tabicl
