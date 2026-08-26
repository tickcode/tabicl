#include "preprocess/ensemble.h"

#include <algorithm>
#include <stdexcept>

#include "preprocess/rng_cpython.h"

namespace tabicl {

ShuffleMethod parse_shuffle_method(const std::string& s) {
  if (s == "none") return ShuffleMethod::None;
  if (s == "shift") return ShuffleMethod::Shift;
  if (s == "random") return ShuffleMethod::Random;
  if (s == "latin") return ShuffleMethod::Latin;
  throw std::runtime_error("unknown shuffle method: " + s);
}

namespace {

using Perm = std::vector<int32_t>;

Perm identity(int64_t n) {
  Perm p(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) p[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  return p;
}

// Shuffler._latin_squares: _rls + shuffle/transpose/shuffle.
std::vector<Perm> latin_squares(int64_t n, PyRandom& rng) {
  // _rls(symbols): recursive Latin square build; mirrors rng call order.
  std::vector<int32_t> symbols(identity(n));
  // Recursive lambda via explicit stack of chosen symbols (recursion depth can
  // reach n; Python raises the recursion limit to 100k, we iterate instead).
  // _rls removes a chosen symbol at each level until one remains, then
  // rebuilds the square on the way back up.
  std::vector<int32_t> chosen;  // symbols chosen at levels n, n-1, ..., 2
  chosen.reserve(static_cast<size_t>(n));
  while (symbols.size() > 1) {
    const size_t idx = static_cast<size_t>(rng.choice_index(symbols.size()));
    const int32_t sym = symbols[idx];
    // Python list.remove(sym): removes first occurrence by value (values unique).
    symbols.erase(std::find(symbols.begin(), symbols.end(), sym));
    chosen.push_back(sym);
  }
  // Base case: square = [symbols] (single row, single element).
  std::vector<Perm> square;
  square.push_back(symbols);
  // Unwind: for each level (last chosen first), append copy of row 0 and
  // insert the level's symbol at position i of row i.
  for (auto it = chosen.rbegin(); it != chosen.rend(); ++it) {
    const int32_t sym = *it;
    square.push_back(square[0]);  // square.append(square[0].copy())
    const size_t rows = square.size();
    for (size_t i = 0; i < rows; ++i)
      square[i].insert(square[i].begin() + static_cast<int64_t>(i), sym);
  }
  // _shuffle_transpose_shuffle
  rng.shuffle(square);
  std::vector<Perm> trans(square[0].size(), Perm(square.size()));
  for (size_t r = 0; r < square.size(); ++r)
    for (size_t c = 0; c < square[r].size(); ++c) trans[c][r] = square[r][c];
  rng.shuffle(trans);
  return trans;
}

// itertools.permutations(range(n)) in lexicographic order.
std::vector<Perm> all_permutations(int64_t n) {
  std::vector<Perm> out;
  Perm p = identity(n);
  do {
    out.push_back(p);
  } while (std::next_permutation(p.begin(), p.end()));
  return out;
}

}  // namespace

std::vector<Perm> shuffler_patterns(int64_t n_elements, ShuffleMethod method,
                                    int n_estimators, uint64_t random_state,
                                    int64_t max_elements_for_latin) {
  if (n_elements <= 0) throw std::runtime_error("shuffler: n_elements must be positive");
  PyRandom rng(random_state);

  if (n_elements > max_elements_for_latin && method == ShuffleMethod::Latin)
    method = ShuffleMethod::Random;

  if (method == ShuffleMethod::None || n_estimators == 1) return {identity(n_elements)};

  switch (method) {
    case ShuffleMethod::Shift: {
      std::vector<Perm> out;
      out.reserve(static_cast<size_t>(n_elements));
      for (int64_t i = 0; i < n_elements; ++i) {
        // indices[-i:] + indices[:-i]  (i = 0 yields identity)
        Perm p;
        p.reserve(static_cast<size_t>(n_elements));
        for (int64_t j = n_elements - i; j < n_elements; ++j)
          p.push_back(static_cast<int32_t>(j));
        for (int64_t j = 0; j < n_elements - i; ++j) p.push_back(static_cast<int32_t>(j));
        out.push_back(std::move(p));
      }
      return out;
    }
    case ShuffleMethod::Random: {
      if (n_elements <= 5) {
        const auto perms = all_permutations(n_elements);
        const int64_t k = std::min<int64_t>(n_estimators,
                                            static_cast<int64_t>(perms.size()));
        const auto idxs = rng.sample_indices(static_cast<int64_t>(perms.size()), k);
        std::vector<Perm> out;
        out.reserve(idxs.size());
        for (int64_t i : idxs) out.push_back(perms[static_cast<size_t>(i)]);
        return out;
      }
      std::vector<Perm> out;
      out.reserve(static_cast<size_t>(n_estimators));
      for (int i = 0; i < n_estimators; ++i) {
        const auto idxs = rng.sample_indices(n_elements, n_elements);
        Perm p(idxs.size());
        for (size_t j = 0; j < idxs.size(); ++j) p[j] = static_cast<int32_t>(idxs[j]);
        out.push_back(std::move(p));
      }
      return out;
    }
    case ShuffleMethod::Latin:
      return latin_squares(n_elements, rng);
    case ShuffleMethod::None:
      break;  // handled above
  }
  throw std::runtime_error("shuffler: unreachable");
}

EnsembleConfigs generate_ensemble_configs(int64_t n_features, int64_t n_classes,
                                          int n_estimators,
                                          const std::vector<std::string>& norm_methods,
                                          const std::string& feat_shuffle_method,
                                          const std::string& class_shuffle_method,
                                          uint64_t random_state) {
  const auto x_shuffles = shuffler_patterns(
      n_features, parse_shuffle_method(feat_shuffle_method), n_estimators, random_state);

  std::vector<Perm> y_patterns;
  const bool classification = n_classes > 0;
  if (classification) {
    y_patterns = shuffler_patterns(n_classes, parse_shuffle_method(class_shuffle_method),
                                   n_estimators, random_state);
  } else {
    y_patterns.push_back({});  // Python: [None]
  }

  // itertools.product(X_shuffles, y_patterns), then rng_.shuffle(...)
  std::vector<EnsembleMember> shuffle_configs;
  shuffle_configs.reserve(x_shuffles.size() * y_patterns.size());
  for (const auto& xs : x_shuffles)
    for (const auto& ys : y_patterns) shuffle_configs.push_back({xs, ys});
  PyRandom rng(random_state);
  rng.shuffle(shuffle_configs);

  // itertools.product(shuffle_configs, norm_methods)[:n_estimators],
  // grouped by method in first-appearance order (see header for the
  // deliberate deviation from Python's hash-dependent set order).
  EnsembleConfigs out;
  int taken = 0;
  for (size_t ci = 0; ci < shuffle_configs.size() && taken < n_estimators; ++ci) {
    for (size_t mi = 0; mi < norm_methods.size() && taken < n_estimators; ++mi, ++taken) {
      const std::string& method = norm_methods[mi];
      auto it = std::find(out.method_order.begin(), out.method_order.end(), method);
      size_t gi;
      if (it == out.method_order.end()) {
        out.method_order.push_back(method);
        out.members.emplace_back();
        gi = out.method_order.size() - 1;
      } else {
        gi = static_cast<size_t>(it - out.method_order.begin());
      }
      out.members[gi].push_back(shuffle_configs[ci]);
    }
  }
  return out;
}

}  // namespace tabicl
