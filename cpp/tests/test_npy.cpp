#include <cstdio>
#include <string>
#include <vector>

#include "doctest.h"
#include "io/npy.h"

using tabicl::npy::Array;
using tabicl::npy::DType;

TEST_CASE("npy: round-trip f4/f8/i8") {
  const std::string dir = "./";
  {
    std::vector<float> v = {1.5f, -2.25f, 3.0f, 0.0f, 1e-8f, -1e8f};
    Array a = tabicl::npy::make_f4({2, 3}, v.data());
    tabicl::npy::save(dir + "rt_f4.npy", a);
    Array b = tabicl::npy::load(dir + "rt_f4.npy");
    CHECK(b.dtype == DType::F4);
    REQUIRE(b.shape == std::vector<int64_t>{2, 3});
    for (int i = 0; i < 6; ++i) CHECK(b.f4()[i] == v[i]);
  }
  {
    std::vector<double> v = {1.0 / 3.0, -2e300, 5e-324};
    Array a = tabicl::npy::make_f8({3}, v.data());
    tabicl::npy::save(dir + "rt_f8.npy", a);
    Array b = tabicl::npy::load(dir + "rt_f8.npy");
    CHECK(b.dtype == DType::F8);
    for (int i = 0; i < 3; ++i) CHECK(b.f8()[i] == v[i]);
  }
  {
    std::vector<int64_t> v = {-9223372036854775807LL, 0, 42};
    Array a = tabicl::npy::make_i8({3, 1}, v.data());
    tabicl::npy::save(dir + "rt_i8.npy", a);
    Array b = tabicl::npy::load(dir + "rt_i8.npy");
    CHECK(b.dtype == DType::I8);
    for (int i = 0; i < 3; ++i) CHECK(b.i8()[i] == v[i]);
  }
  std::remove("rt_f4.npy");
  std::remove("rt_f8.npy");
  std::remove("rt_i8.npy");
}

TEST_CASE("npy: parses a numpy-1.0-format byte stream") {
  // Bytes of np.save for np.array([[1.5, 2.5]], dtype='<f4') (v1.0 header, 64-byte aligned)
  std::vector<uint8_t> bytes = {
      0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00, 0x76, 0x00};
  std::string header =
      "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 2), }";
  header += std::string(118 - 10 - header.size() - 1, ' ');
  header += '\n';
  bytes[8] = static_cast<uint8_t>(header.size() & 0xFF);
  bytes[9] = static_cast<uint8_t>(header.size() >> 8);
  for (char c : header) bytes.push_back(static_cast<uint8_t>(c));
  const float vals[2] = {1.5f, 2.5f};
  const uint8_t* vb = reinterpret_cast<const uint8_t*>(vals);
  bytes.insert(bytes.end(), vb, vb + 8);

  Array a = tabicl::npy::parse(bytes.data(), bytes.size());
  CHECK(a.dtype == DType::F4);
  REQUIRE(a.shape == std::vector<int64_t>{1, 2});
  CHECK(a.f4()[0] == 1.5f);
  CHECK(a.f4()[1] == 2.5f);
}

TEST_CASE("npy: rejects fortran order and unknown dtypes") {
  std::string h1 = "{'descr': '<f4', 'fortran_order': True, 'shape': (2,), }\n";
  std::vector<uint8_t> b1 = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00,
                             static_cast<uint8_t>(h1.size() & 0xFF),
                             static_cast<uint8_t>(h1.size() >> 8)};
  for (char c : h1) b1.push_back(static_cast<uint8_t>(c));
  b1.resize(b1.size() + 8, 0);
  CHECK_THROWS(tabicl::npy::parse(b1.data(), b1.size()));

  std::string h2 = "{'descr': '<c16', 'fortran_order': False, 'shape': (1,), }\n";
  std::vector<uint8_t> b2 = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00,
                             static_cast<uint8_t>(h2.size() & 0xFF),
                             static_cast<uint8_t>(h2.size() >> 8)};
  for (char c : h2) b2.push_back(static_cast<uint8_t>(c));
  b2.resize(b2.size() + 16, 0);
  CHECK_THROWS(tabicl::npy::parse(b2.data(), b2.size()));
}
