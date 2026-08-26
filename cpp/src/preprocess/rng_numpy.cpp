#include "preprocess/rng_numpy.h"

#include <cmath>
#include <stdexcept>

#include "preprocess/ziggurat_constants.h"

namespace tabicl {

// ---------------------------------------------------------------------------
// Legacy RandomState (MT19937, init_genrand seeding)
// ---------------------------------------------------------------------------

NpRandomState::NpRandomState(uint32_t seed) {
  mt_[0] = seed;
  for (mti_ = 1; mti_ < N; mti_++) {
    mt_[mti_] = 1812433253u * (mt_[mti_ - 1] ^ (mt_[mti_ - 1] >> 30)) +
                static_cast<uint32_t>(mti_);
  }
}

uint32_t NpRandomState::next_u32() {
  constexpr uint32_t MATRIX_A = 0x9908b0df;
  constexpr uint32_t UPPER_MASK = 0x80000000;
  constexpr uint32_t LOWER_MASK = 0x7fffffff;
  uint32_t y;
  if (mti_ >= N) {
    static const uint32_t mag01[2] = {0u, MATRIX_A};
    int kk;
    for (kk = 0; kk < N - 397; kk++) {
      y = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
      mt_[kk] = mt_[kk + 397] ^ (y >> 1) ^ mag01[y & 0x1u];
    }
    for (; kk < N - 1; kk++) {
      y = (mt_[kk] & UPPER_MASK) | (mt_[kk + 1] & LOWER_MASK);
      mt_[kk] = mt_[kk + (397 - N)] ^ (y >> 1) ^ mag01[y & 0x1u];
    }
    y = (mt_[N - 1] & UPPER_MASK) | (mt_[0] & LOWER_MASK);
    mt_[N - 1] = mt_[396] ^ (y >> 1) ^ mag01[y & 0x1u];
    mti_ = 0;
  }
  y = mt_[mti_++];
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680u;
  y ^= (y << 15) & 0xefc60000u;
  y ^= (y >> 18);
  return y;
}

uint64_t NpRandomState::bounded(uint64_t max) {
  // legacy rk_interval: smallest all-ones mask >= max, rejection sample.
  uint64_t mask = max;
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
  mask |= mask >> 32;
  if (max <= 0xFFFFFFFFull) {
    while (true) {
      const uint64_t v = next_u32() & mask;
      if (v <= max) return v;
    }
  }
  while (true) {
    const uint64_t hi = next_u32();
    const uint64_t lo = next_u32();
    const uint64_t v = ((hi << 32) | lo) & mask;
    if (v <= max) return v;
  }
}

// ---------------------------------------------------------------------------
// SeedSequence
// ---------------------------------------------------------------------------

namespace {
constexpr uint32_t INIT_A = 0x43b0d7e5;
constexpr uint32_t MULT_A = 0x931e8875;
constexpr uint32_t INIT_B = 0x8b51f9dd;
constexpr uint32_t MULT_B = 0x58f38ded;
constexpr uint32_t MIX_MULT_L = 0xca01f9dd;
constexpr uint32_t MIX_MULT_R = 0x4973f715;
constexpr uint32_t XSHIFT = 16;

uint32_t hashmix(uint32_t value, uint32_t& hash_const) {
  value ^= hash_const;
  hash_const *= MULT_A;
  value *= hash_const;
  value ^= value >> XSHIFT;
  return value;
}

uint32_t mix(uint32_t x, uint32_t y) {
  uint32_t result = MIX_MULT_L * x - MIX_MULT_R * y;
  result ^= result >> XSHIFT;
  return result;
}
}  // namespace

NpSeedSequence::NpSeedSequence(uint64_t entropy) {
  std::vector<uint32_t> words;
  do {
    words.push_back(static_cast<uint32_t>(entropy & 0xFFFFFFFFu));
    entropy >>= 32;
  } while (entropy != 0);
  *this = NpSeedSequence(words);
}

NpSeedSequence::NpSeedSequence(const std::vector<uint32_t>& entropy_words) {
  if (entropy_words.empty()) throw std::runtime_error("SeedSequence: empty entropy");
  constexpr size_t POOL = 4;
  pool_.assign(POOL, 0);
  uint32_t hc = INIT_A;
  for (size_t i = 0; i < POOL; ++i)
    pool_[i] = hashmix(i < entropy_words.size() ? entropy_words[i] : 0u, hc);
  for (size_t src = 0; src < POOL; ++src)
    for (size_t dst = 0; dst < POOL; ++dst)
      if (src != dst) pool_[dst] = mix(pool_[dst], hashmix(pool_[src], hc));
  for (size_t src = POOL; src < entropy_words.size(); ++src)
    for (size_t dst = 0; dst < POOL; ++dst)
      pool_[dst] = mix(pool_[dst], hashmix(entropy_words[src], hc));
}

std::vector<uint64_t> NpSeedSequence::generate_state64(int n_words) const {
  uint32_t hc = INIT_B;
  std::vector<uint32_t> out32(static_cast<size_t>(n_words) * 2);
  for (size_t i = 0; i < out32.size(); ++i) {
    uint32_t v = pool_[i % pool_.size()];
    v ^= hc;
    hc *= MULT_B;
    v *= hc;
    v ^= v >> XSHIFT;
    out32[i] = v;
  }
  std::vector<uint64_t> out(static_cast<size_t>(n_words));
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = static_cast<uint64_t>(out32[2 * i]) |
             (static_cast<uint64_t>(out32[2 * i + 1]) << 32);
  return out;
}

// ---------------------------------------------------------------------------
// PCG64 (setseq 128, XSL-RR output) + ziggurat standard normal
// ---------------------------------------------------------------------------

namespace {
using u128 = unsigned __int128;
constexpr uint64_t PCG_MULT_HI = 2549297995355413924ull;
constexpr uint64_t PCG_MULT_LO = 4865540595714422341ull;
inline u128 pcg_mult() { return (static_cast<u128>(PCG_MULT_HI) << 64) | PCG_MULT_LO; }

inline uint64_t rotr64(uint64_t value, unsigned rot) {
  return (value >> rot) | (value << ((-rot) & 63u));
}
}  // namespace

NpPCG64::NpPCG64(uint64_t seed) { seed_from(NpSeedSequence(seed)); }

NpPCG64::NpPCG64(const std::vector<uint32_t>& seed_words) {
  seed_from(NpSeedSequence(seed_words));
}

void NpPCG64::seed_from(const NpSeedSequence& ss) {
  const auto v = ss.generate_state64(4);
  const u128 initstate = (static_cast<u128>(v[0]) << 64) | v[1];
  const u128 initseq = (static_cast<u128>(v[2]) << 64) | v[3];
  state_ = 0;
  inc_ = (initseq << 1) | 1;
  state_ = state_ * pcg_mult() + inc_;
  state_ += initstate;
  state_ = state_ * pcg_mult() + inc_;
}

uint64_t NpPCG64::next64() {
  state_ = state_ * pcg_mult() + inc_;  // step, then output
  const uint64_t hi = static_cast<uint64_t>(state_ >> 64);
  const uint64_t lo = static_cast<uint64_t>(state_);
  return rotr64(hi ^ lo, static_cast<unsigned>(hi >> 58));
}

double NpPCG64::next_double() {
  return static_cast<double>(next64() >> 11) * (1.0 / 9007199254740992.0);
}

double NpPCG64::standard_normal() {
  // Port of numpy random_standard_normal (distributions.c, v2.4.0).
  while (true) {
    uint64_t r = next64();
    const int idx = static_cast<int>(r & 0xff);
    r >>= 8;
    const int sign = static_cast<int>(r & 0x1);
    const uint64_t rabs = (r >> 1) & 0x000fffffffffffffull;
    double x = static_cast<double>(rabs) * ziggurat::wi_double[idx];
    if (sign & 0x1) x = -x;
    if (rabs < ziggurat::ki_double[idx]) return x;
    if (idx == 0) {
      while (true) {
        const double xx = -ziggurat::nor_inv_r * std::log1p(-next_double());
        const double yy = -std::log1p(-next_double());
        if (yy + yy > xx * xx)
          return ((rabs >> 8) & 0x1) ? -(ziggurat::nor_r + xx)
                                     : ziggurat::nor_r + xx;
      }
    } else {
      if ((ziggurat::fi_double[idx - 1] - ziggurat::fi_double[idx]) * next_double() +
              ziggurat::fi_double[idx] <
          std::exp(-0.5 * x * x))
        return x;
    }
  }
}

}  // namespace tabicl
