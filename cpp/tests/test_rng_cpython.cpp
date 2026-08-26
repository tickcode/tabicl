// Bitwise validation of the CPython random.Random port against golden
// sequences from tools/generate_fixtures.py (section rng_cpython).
#include <cstdint>
#include <string>
#include <vector>

#include "doctest.h"
#include "preprocess/rng_cpython.h"
#include "test_helpers.h"

using tabicl::PyRandom;
using tabicl::test::fixture_exists;
using tabicl::test::load_fixture;

namespace {

// Must match RNG_SEEDS in tools/generate_fixtures.py (little-endian 32-bit words).
const std::vector<std::vector<uint32_t>> kSeeds = {
    {0u},
    {1u},
    {42u},
    {12345u},
    {0x7FFFFFFFu},            // 2^31 - 1
    {0xFFFFFFFFu},            // 2^32 - 1
    {12345u, 0u, 1u},         // 2^64 + 12345
};

}  // namespace

TEST_CASE("rng_cpython: random(), getrandbits, shuffle, sample, choice match goldens") {
  if (!fixture_exists("rng_cpython/random_s0.npy")) {
    MESSAGE("rng_cpython fixtures missing; run tools/generate_fixtures.py");
    return;
  }
  for (size_t si = 0; si < kSeeds.size(); ++si) {
    const std::string s = "_s" + std::to_string(si);

    {
      auto g = load_fixture("rng_cpython/random" + s + ".npy");
      PyRandom r(kSeeds[si]);
      for (int64_t i = 0; i < g.numel(); ++i) {
        const double v = r.random();
        REQUIRE_MESSAGE(v == g.f8()[i], "random() diverged at draw " << i << " seed " << si);
      }
    }

    for (int k : {1, 8, 31, 32, 33, 64, 128}) {
      auto g = load_fixture("rng_cpython/getrandbits_k" + std::to_string(k) + s + ".npy");
      REQUIRE(g.shape[0] == 2);
      const int64_t n = g.shape[1];
      const auto* lo = reinterpret_cast<const uint64_t*>(g.i8());
      const auto* hi = lo + n;
      PyRandom r(kSeeds[si]);
      for (int64_t i = 0; i < n; ++i) {
        if (k <= 64) {
          REQUIRE_MESSAGE(r.getrandbits64(k) == lo[i],
                          "getrandbits(" << k << ") diverged at " << i << " seed " << si);
        } else {
          const auto v = r.getrandbits128(k);
          REQUIRE(v.lo == lo[i]);
          REQUIRE(v.hi == hi[i]);
        }
      }
    }

    for (int n : {1, 2, 3, 10, 57, 1000}) {
      auto g = load_fixture("rng_cpython/shuffle_n" + std::to_string(n) + s + ".npy");
      PyRandom r(kSeeds[si]);
      std::vector<int64_t> x(static_cast<size_t>(n));
      for (int i = 0; i < n; ++i) x[static_cast<size_t>(i)] = i;
      r.shuffle(x);
      for (int i = 0; i < n; ++i)
        REQUIRE_MESSAGE(x[static_cast<size_t>(i)] == g.i8()[i],
                        "shuffle n=" << n << " diverged at " << i << " seed " << si);
    }

    {
      const std::vector<std::pair<int, int>> cases = {
          {10, 3}, {1000, 5}, {24, 20}, {120, 100}, {5, 5}};
      for (auto [n, k] : cases) {
        auto g = load_fixture("rng_cpython/sample_n" + std::to_string(n) + "_k" +
                              std::to_string(k) + s + ".npy");
        PyRandom r(kSeeds[si]);
        auto got = r.sample_indices(n, k);
        REQUIRE(static_cast<int64_t>(got.size()) == g.numel());
        for (int64_t i = 0; i < g.numel(); ++i)
          REQUIRE_MESSAGE(got[static_cast<size_t>(i)] == g.i8()[i],
                          "sample(" << n << "," << k << ") diverged at " << i
                                    << " seed " << si);
      }
    }

    {
      auto g = load_fixture("rng_cpython/choice" + s + ".npy");
      PyRandom r(kSeeds[si]);
      for (int64_t i = 0; i < g.numel(); ++i)
        REQUIRE_MESSAGE(static_cast<int64_t>(r.choice_index(97)) == g.i8()[i],
                        "choice diverged at " << i << " seed " << si);
    }
  }
}
