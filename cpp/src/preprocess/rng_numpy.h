// Bit-exact ports of the NumPy RNG machinery TabICL's preprocessing uses:
//  - NpRandomState: legacy np.random.RandomState (MT19937) `shuffle`
//    (used by sklearn's QuantileTransformer subsampling via utils.resample)
//  - SeedSequence + PCG64 + ziggurat standard_normal: np.random.default_rng
//    (used by RTDLQuantileTransformer fit-time noise)
// Validated bitwise against goldens from tools/generate_fixtures.py.
#pragma once

#include <cstdint>
#include <vector>

namespace tabicl {

// Legacy RandomState seeded with a scalar int (init_genrand path).
class NpRandomState {
 public:
  explicit NpRandomState(uint32_t seed);

  uint32_t next_u32();

  // RandomState.shuffle: reverse loop, bounded draw via mask rejection.
  template <typename T>
  void shuffle(std::vector<T>& x) {
    if (x.size() < 2) return;
    for (size_t i = x.size() - 1; i >= 1; --i) {
      const size_t j = static_cast<size_t>(bounded(static_cast<uint64_t>(i)));
      std::swap(x[i], x[j]);
    }
  }

  // Uniform draw in [0, max] via mask rejection (legacy rk_interval).
  uint64_t bounded(uint64_t max);

 private:
  static constexpr int N = 624;
  uint32_t mt_[N];
  int mti_ = N + 1;
};

// numpy SeedSequence (pool_size 4) for integer entropy.
class NpSeedSequence {
 public:
  explicit NpSeedSequence(const std::vector<uint32_t>& entropy_words);
  explicit NpSeedSequence(uint64_t entropy);

  // generate_state(n_words, np.uint64)
  std::vector<uint64_t> generate_state64(int n_words) const;

 private:
  std::vector<uint32_t> pool_;
};

// numpy PCG64 (xsl-rr 128/64 variant) + Generator methods we need.
class NpPCG64 {
 public:
  explicit NpPCG64(uint64_t seed);
  explicit NpPCG64(const std::vector<uint32_t>& seed_words);

  uint64_t next64();
  // Generator.random(): 53-bit double in [0, 1).
  double next_double();
  // Generator.standard_normal(): double-precision ziggurat.
  double standard_normal();

 private:
  void seed_from(const NpSeedSequence& ss);
  unsigned __int128 state_;
  unsigned __int128 inc_;
};

}  // namespace tabicl
