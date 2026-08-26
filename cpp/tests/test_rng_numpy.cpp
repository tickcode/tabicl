// Bitwise validation of the NumPy RNG ports against golden sequences from
// tools/generate_fixtures.py (section rng_numpy).
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "doctest.h"
#include "preprocess/rng_numpy.h"
#include "test_helpers.h"

using tabicl::NpPCG64;
using tabicl::NpRandomState;
using tabicl::test::fixture_exists;
using tabicl::test::load_fixture;

namespace {
// Must match RNG_SEEDS in tools/generate_fixtures.py.
const std::vector<std::vector<uint32_t>> kSeeds = {
    {0u}, {1u}, {42u}, {12345u}, {0x7FFFFFFFu}, {0xFFFFFFFFu}, {12345u, 0u, 1u}};
}  // namespace

TEST_CASE("rng_numpy: RandomState shuffle matches goldens") {
  if (!fixture_exists("rng_numpy/rs_shuffle_n10_s0.npy")) {
    MESSAGE("rng_numpy fixtures missing; run tools/generate_fixtures.py");
    return;
  }
  for (size_t si = 0; si < kSeeds.size(); ++si) {
    // Fixture used seed % 2**32 (single word).
    const uint32_t seed32 = kSeeds[si][0];
    for (int n : {2, 10, 1000, 10001}) {
      auto g = load_fixture("rng_numpy/rs_shuffle_n" + std::to_string(n) + "_s" +
                            std::to_string(si) + ".npy");
      NpRandomState rs(seed32);
      std::vector<int64_t> x(static_cast<size_t>(n));
      for (int i = 0; i < n; ++i) x[static_cast<size_t>(i)] = i;
      rs.shuffle(x);
      for (int i = 0; i < n; ++i)
        REQUIRE_MESSAGE(x[static_cast<size_t>(i)] == g.i8()[i],
                        "rs.shuffle n=" << n << " diverged at " << i << " seed " << si);
    }
  }
}

TEST_CASE("rng_numpy: PCG64 raw, uniform, and ziggurat normals match goldens") {
  if (!fixture_exists("rng_numpy/pcg64_raw_s0.npy")) {
    MESSAGE("rng_numpy fixtures missing; run tools/generate_fixtures.py");
    return;
  }
  for (size_t si = 0; si < kSeeds.size(); ++si) {
    const std::string s = "_s" + std::to_string(si);
    {
      auto g = load_fixture("rng_numpy/pcg64_raw" + s + ".npy");
      NpPCG64 pcg(kSeeds[si]);
      const auto* ref = reinterpret_cast<const uint64_t*>(g.i8());
      for (int64_t i = 0; i < g.numel(); ++i)
        REQUIRE_MESSAGE(pcg.next64() == ref[i],
                        "pcg64 raw diverged at " << i << " seed " << si);
    }
    {
      auto g = load_fixture("rng_numpy/pcg64_uniform" + s + ".npy");
      NpPCG64 pcg(kSeeds[si]);
      for (int64_t i = 0; i < g.numel(); ++i) {
        const double v = pcg.next_double();
        REQUIRE_MESSAGE(std::memcmp(&v, &g.f8()[i], 8) == 0,
                        "pcg64 uniform diverged at " << i << " seed " << si);
      }
    }
    {
      auto g = load_fixture("rng_numpy/pcg64_normal" + s + ".npy");
      NpPCG64 pcg(kSeeds[si]);
      for (int64_t i = 0; i < g.numel(); ++i) {
        const double v = pcg.standard_normal();
        REQUIRE_MESSAGE(std::memcmp(&v, &g.f8()[i], 8) == 0,
                        "ziggurat normal diverged at " << i << " seed " << si);
      }
    }
  }
}
