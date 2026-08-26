// Port of tabicl EnsembleGenerator's configuration machinery (Shuffler +
// _generate_ensemble): which feature/class permutations and normalization
// method each ensemble member uses. Permutations are bit-exact vs CPython
// (validated against shuffler/ensemble_configs golden fixtures).
//
// One deliberate deviation: Python groups members per normalization method
// via `list(set(...))`, whose order depends on PYTHONHASHSEED (i.e. varies
// per process). The C++ port uses the deterministic first-appearance order
// of methods in the truncated (member, method) product. Within-group member
// lists are identical to Python; only the group order can differ, which
// perturbs float32 ensemble accumulation at last-bit level only.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tabicl {

enum class ShuffleMethod { None, Shift, Random, Latin };

ShuffleMethod parse_shuffle_method(const std::string& s);

// Shuffler.shuffle(n_estimators): list of permutations of 0..n_elements-1.
std::vector<std::vector<int32_t>> shuffler_patterns(int64_t n_elements,
                                                    ShuffleMethod method,
                                                    int n_estimators,
                                                    uint64_t random_state,
                                                    int64_t max_elements_for_latin = 4000);

struct EnsembleMember {
  std::vector<int32_t> feature_shuffle;  // permutation over kept features
  std::vector<int32_t> class_shuffle;    // empty for regression
};

struct EnsembleConfigs {
  // Normalization methods in deterministic first-appearance order.
  std::vector<std::string> method_order;
  // members[i] corresponds to method_order[i].
  std::vector<std::vector<EnsembleMember>> members;

  int64_t total_members() const {
    int64_t n = 0;
    for (const auto& m : members) n += static_cast<int64_t>(m.size());
    return n;
  }
};

// EnsembleGenerator._generate_ensemble. `n_features` is the count AFTER the
// unique-feature filter; `n_classes` is 0 for regression.
EnsembleConfigs generate_ensemble_configs(int64_t n_features, int64_t n_classes,
                                          int n_estimators,
                                          const std::vector<std::string>& norm_methods,
                                          const std::string& feat_shuffle_method,
                                          const std::string& class_shuffle_method,
                                          uint64_t random_state);

}  // namespace tabicl
