// Bit-exact port of CPython's random.Random (MT19937 + the Python-level
// algorithms the TabICL ensemble generator uses: random(), getrandbits,
// shuffle, sample, choice). Validated against golden sequences generated
// by tools/generate_fixtures.py (section rng_cpython).
#pragma once

#include <cstdint>
#include <vector>

namespace tabicl {

class PyRandom {
 public:
  // Equivalent to random.Random(seed) for a non-negative integer seed given
  // as little-endian 32-bit words (CPython seeds MT19937 with init_by_array
  // over the absolute value's digits). For seed 0 pass {0}.
  explicit PyRandom(uint64_t seed);
  explicit PyRandom(const std::vector<uint32_t>& seed_words);

  // random.Random.random(): 53-bit double in [0, 1).
  double random();

  // getrandbits(k) for k <= 64.
  uint64_t getrandbits64(int k);

  // getrandbits(k) for k <= 128, returned as {lo, hi} 64-bit limbs.
  struct U128 { uint64_t lo, hi; };
  U128 getrandbits128(int k);

  // _randbelow_with_getrandbits(n): uniform int in [0, n) for n >= 1.
  uint64_t randbelow(uint64_t n);

  // random.Random.shuffle (reverse Fisher-Yates), in place.
  template <typename T>
  void shuffle(std::vector<T>& x) {
    if (x.size() < 2) return;
    for (size_t i = x.size() - 1; i >= 1; --i) {
      const size_t j = static_cast<size_t>(randbelow(i + 1));
      std::swap(x[i], x[j]);
    }
  }

  // random.Random.sample(range(n), k): index positions in selection order.
  // Replicates both CPython branches (pool vs selection-set).
  std::vector<int64_t> sample_indices(int64_t n, int64_t k);

  // random.Random.choice(seq) for a sequence of length n: the chosen index.
  uint64_t choice_index(uint64_t n) { return randbelow(n); }

 private:
  uint32_t genrand_uint32();
  void init_genrand(uint32_t s);
  void init_by_array(const std::vector<uint32_t>& key);

  static constexpr int N = 624;
  uint32_t mt_[N];
  int mti_ = N + 1;
};

}  // namespace tabicl
