#include "hierarchical.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

#include "model_forward.h"

namespace tabicl {

namespace {

struct ClassNode {
  bool is_leaf = false;
  std::vector<int32_t> classes;        // sorted unique classes at this node
  std::vector<float> R;                // (n, D) rows assigned to this node
  std::vector<int32_t> y;              // class of each row
  std::vector<int32_t> group_of_row;   // internal: group index per row
  std::vector<std::unique_ptr<ClassNode>> children;
};

// learning.py _grouping: balanced contiguous split of the node's classes.
std::vector<int32_t> grouping(int64_t num_classes, int64_t max_classes,
                              int64_t& num_groups) {
  num_groups = std::min((num_classes + max_classes - 1) / max_classes, max_classes);
  std::vector<int32_t> assign(static_cast<size_t>(num_classes));
  int64_t pos = 0, remaining = num_classes, groups_left = num_groups;
  for (int64_t g = 0; g < num_groups; ++g) {
    const int64_t size = (remaining + groups_left - 1) / groups_left;
    for (int64_t i = 0; i < size; ++i) assign[static_cast<size_t>(pos + i)] = static_cast<int32_t>(g);
    pos += size;
    remaining -= size;
    groups_left -= 1;
  }
  return assign;
}

void fit_node(ClassNode& node, std::vector<float> R, std::vector<int32_t> y,
              int64_t D, int64_t max_classes) {
  node.classes = y;
  std::sort(node.classes.begin(), node.classes.end());
  node.classes.erase(std::unique(node.classes.begin(), node.classes.end()),
                     node.classes.end());
  node.R = std::move(R);
  node.y = std::move(y);
  const int64_t k = static_cast<int64_t>(node.classes.size());
  if (k <= max_classes) {
    node.is_leaf = true;
    return;
  }
  int64_t num_groups = 0;
  const auto assign = grouping(k, max_classes, num_groups);
  node.group_of_row.resize(node.y.size());
  for (size_t i = 0; i < node.y.size(); ++i) {
    const auto it = std::lower_bound(node.classes.begin(), node.classes.end(), node.y[i]);
    node.group_of_row[i] = assign[static_cast<size_t>(it - node.classes.begin())];
  }
  const int64_t n = static_cast<int64_t>(node.y.size());
  for (int64_t g = 0; g < num_groups; ++g) {
    std::vector<float> Rg;
    std::vector<int32_t> yg;
    for (int64_t i = 0; i < n; ++i)
      if (node.group_of_row[static_cast<size_t>(i)] == g) {
        Rg.insert(Rg.end(), node.R.begin() + i * D, node.R.begin() + (i + 1) * D);
        yg.push_back(node.y[static_cast<size_t>(i)]);
      }
    auto child = std::make_unique<ClassNode>();
    fit_node(*child, std::move(Rg), std::move(yg), D, max_classes);
    node.children.push_back(std::move(child));
  }
}

// _predict_standard on [node rows ++ test rows] with the given per-row train
// labels; returns softmax(logits/tau) probs (n_test, k).
std::vector<float> node_probs(const Model& model, GraphRunner& runner,
                              const ClassNode& node, const std::vector<int32_t>& yenc,
                              int64_t k, const float* R_test, int64_t n_test,
                              int64_t D, float tau) {
  const int64_t n_node = static_cast<int64_t>(node.y.size());
  const int64_t T = n_node + n_test;
  std::vector<float> R_full(static_cast<size_t>(T * D));
  std::copy(node.R.begin(), node.R.end(), R_full.begin());
  std::copy(R_test, R_test + n_test * D, R_full.begin() + n_node * D);
  std::vector<float> yf(yenc.begin(), yenc.end());

  const auto raw = icl_forward(model, runner, R_full.data(), yf.data(), 1, T,
                               n_node, k);
  const int64_t full_out = model.config().max_classes;
  std::vector<float> probs(static_cast<size_t>(n_test * k));
  for (int64_t t = 0; t < n_test; ++t) {
    const float* row = &raw[static_cast<size_t>((n_node + t) * full_out)];
    float mx = -std::numeric_limits<float>::infinity();
    for (int64_t j = 0; j < k; ++j) mx = std::max(mx, row[j] / tau);
    float denom = 0.0f;
    for (int64_t j = 0; j < k; ++j) {
      probs[static_cast<size_t>(t * k + j)] = std::exp(row[j] / tau - mx);
      denom += probs[static_cast<size_t>(t * k + j)];
    }
    for (int64_t j = 0; j < k; ++j) probs[static_cast<size_t>(t * k + j)] /= denom;
  }
  return probs;
}

std::vector<float> process_node(const Model& model, GraphRunner& runner,
                                const ClassNode& node, const float* R_test,
                                int64_t n_test, int64_t D, int64_t C, float tau) {
  std::vector<float> out(static_cast<size_t>(n_test * C), 0.0f);
  if (node.is_leaf) {
    // Label-encode node.y to contiguous ids in sorted-class order.
    std::vector<int32_t> yenc(node.y.size());
    for (size_t i = 0; i < node.y.size(); ++i) {
      const auto it =
          std::lower_bound(node.classes.begin(), node.classes.end(), node.y[i]);
      yenc[i] = static_cast<int32_t>(it - node.classes.begin());
    }
    const int64_t k = static_cast<int64_t>(node.classes.size());
    const auto probs = node_probs(model, runner, node, yenc, k, R_test, n_test, D, tau);
    for (int64_t t = 0; t < n_test; ++t)
      for (int64_t j = 0; j < k; ++j)
        out[static_cast<size_t>(t * C + node.classes[static_cast<size_t>(j)])] =
            probs[static_cast<size_t>(t * k + j)];
    return out;
  }
  const int64_t num_groups = static_cast<int64_t>(node.children.size());
  const auto group_probs = node_probs(model, runner, node, node.group_of_row,
                                      num_groups, R_test, n_test, D, tau);
  for (int64_t g = 0; g < num_groups; ++g) {
    const auto child = process_node(model, runner, *node.children[static_cast<size_t>(g)],
                                    R_test, n_test, D, C, tau);
    for (int64_t t = 0; t < n_test; ++t)
      for (int64_t j = 0; j < C; ++j)
        out[static_cast<size_t>(t * C + j)] +=
            child[static_cast<size_t>(t * C + j)] *
            group_probs[static_cast<size_t>(t * num_groups + g)];
  }
  return out;
}

}  // namespace

std::vector<float> hierarchical_member_logits(const Model& model,
                                              GraphRunner& runner, const float* R,
                                              const float* y, int64_t T,
                                              int64_t train, int64_t n_classes) {
  const ModelConfig& c = model.config();
  const int64_t D = c.icl_dim();
  const float tau = c.softmax_temperature;
  const int64_t n_test = T - train;

  ClassNode root;
  std::vector<float> R_train(R, R + train * D);
  std::vector<int32_t> yi(static_cast<size_t>(train));
  for (int64_t i = 0; i < train; ++i) yi[static_cast<size_t>(i)] = static_cast<int32_t>(y[i]);
  fit_node(root, std::move(R_train), std::move(yi), D, c.max_classes);

  auto probs = process_node(model, runner, root, R + train * D, n_test, D,
                            n_classes, tau);
  // Pseudo-logits: tau * log(p + 1e-6) (learning.py:483).
  for (float& p : probs) p = tau * std::log(p + 1e-6f);
  return probs;
}

}  // namespace tabicl
