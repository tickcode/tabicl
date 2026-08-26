#include "preprocess/rng_cpython.h"

#include <bit>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace tabicl {

namespace {
constexpr uint32_t MATRIX_A = 0x9908b0df;
constexpr uint32_t UPPER_MASK = 0x80000000;
constexpr uint32_t LOWER_MASK = 0x7fffffff;

int bit_length(uint64_t n) { return 64 - std::countl_zero(n); }
}  // namespace

PyRandom::PyRandom(uint64_t seed) {
  // CPython random_seed: absolute value split into little-endian 32-bit words
  // (at least one word, so seed 0 becomes key = [0]).
  std::vector<uint32_t> words;
  do {
    words.push_back(static_cast<uint32_t>(seed & 0xFFFFFFFFu));
    seed >>= 32;
  } while (seed != 0);
  init_by_array(words);
}

PyRandom::PyRandom(const std::vector<uint32_t>& seed_words) {
  if (seed_words.empty()) throw std::runtime_error("PyRandom: empty seed");
  init_by_array(seed_words);
}

void PyRandom::init_genrand(uint32_t s) {
  mt_[0] = s;
  for (mti_ = 1; mti_ < N; mti_++) {
    mt_[mti_] = 1812433253u * (mt_[mti_ - 1] ^ (mt_[mti_ - 1] >> 30)) +
                static_cast<uint32_t>(mti_);
  }
}

void PyRandom::init_by_array(const std::vector<uint32_t>& key) {
  init_genrand(19650218u);
  size_t i = 1, j = 0;
  size_t k = std::max<size_t>(N, key.size());
  for (; k; k--) {
    mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525u)) +
             key[j] + static_cast<uint32_t>(j);
    i++; j++;
    if (i >= N) { mt_[0] = mt_[N - 1]; i = 1; }
    if (j >= key.size()) j = 0;
  }
  for (k = N - 1; k; k--) {
    mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941u)) -
             static_cast<uint32_t>(i);
    i++;
    if (i >= N) { mt_[0] = mt_[N - 1]; i = 1; }
  }
  mt_[0] = 0x80000000u;
}

uint32_t PyRandom::genrand_uint32() {
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

double PyRandom::random() {
  const uint32_t a = genrand_uint32() >> 5;  // 27 bits
  const uint32_t b = genrand_uint32() >> 6;  // 26 bits
  return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

uint64_t PyRandom::getrandbits64(int k) {
  if (k <= 0 || k > 64) throw std::runtime_error("getrandbits64: bad k");
  if (k <= 32) return genrand_uint32() >> (32 - k);
  // CPython accumulates 32-bit words little-endian, truncating the last.
  uint64_t result = 0;
  int shift = 0;
  while (k > 0) {
    uint32_t r = genrand_uint32();
    if (k < 32) r >>= (32 - k);
    result |= static_cast<uint64_t>(r) << shift;
    shift += 32;
    k -= 32;
  }
  return result;
}

PyRandom::U128 PyRandom::getrandbits128(int k) {
  if (k <= 0 || k > 128) throw std::runtime_error("getrandbits128: bad k");
  U128 out{0, 0};
  int shift = 0;
  while (k > 0) {
    uint32_t r = genrand_uint32();
    if (k < 32) r >>= (32 - k);
    if (shift < 64) out.lo |= static_cast<uint64_t>(r) << shift;
    else out.hi |= static_cast<uint64_t>(r) << (shift - 64);
    shift += 32;
    k -= 32;
  }
  return out;
}

uint64_t PyRandom::randbelow(uint64_t n) {
  if (n == 0) throw std::runtime_error("randbelow: n == 0");
  const int k = bit_length(n);
  uint64_t r = getrandbits64(k);
  while (r >= n) r = getrandbits64(k);
  return r;
}

std::vector<int64_t> PyRandom::sample_indices(int64_t n, int64_t k) {
  if (k < 0 || k > n) throw std::runtime_error("sample: k out of range");
  std::vector<int64_t> result(static_cast<size_t>(k));
  // CPython: setsize = 21; if k > 5: setsize += 4 ** ceil(log(k*3, 4))
  double setsize = 21.0;
  if (k > 5)
    setsize += std::pow(4.0, std::ceil(std::log(static_cast<double>(k) * 3.0) /
                                       std::log(4.0)));
  if (static_cast<double>(n) <= setsize) {
    // pool branch (element values are their indices here)
    std::vector<int64_t> pool(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) pool[static_cast<size_t>(i)] = i;
    for (int64_t i = 0; i < k; ++i) {
      const auto j = static_cast<size_t>(randbelow(static_cast<uint64_t>(n - i)));
      result[static_cast<size_t>(i)] = pool[j];
      pool[j] = pool[static_cast<size_t>(n - i - 1)];
    }
  } else {
    std::unordered_set<uint64_t> selected;
    for (int64_t i = 0; i < k; ++i) {
      uint64_t j = randbelow(static_cast<uint64_t>(n));
      while (selected.count(j)) j = randbelow(static_cast<uint64_t>(n));
      selected.insert(j);
      result[static_cast<size_t>(i)] = static_cast<int64_t>(j);
    }
  }
  return result;
}

}  // namespace tabicl
