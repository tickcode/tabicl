// Minimal .npy reader/writer for parity fixtures.
// Supports v1.0/v2.0 headers, C-order only, little-endian dtypes:
//   f4 f8 i1 i4 i8 u1 b1
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tabicl::npy {

enum class DType { F4, F8, I1, I4, I8, U1, B1 };

struct Array {
  DType dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;  // raw little-endian bytes, C-order

  int64_t numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
  size_t itemsize() const;

  const float*   f4() const;
  const double*  f8() const;
  const int32_t* i4() const;
  const int64_t* i8() const;
  const int8_t*  i1() const;
  const uint8_t* u1() const;  // also used for b1
};

// Throws std::runtime_error on malformed input or unsupported features
// (fortran order, big-endian, object dtype, v3 header).
Array load(const std::string& path);
Array parse(const uint8_t* bytes, size_t len);

void save(const std::string& path, const Array& a);

// Convenience constructors (copy the buffer).
Array make_f4(const std::vector<int64_t>& shape, const float* p);
Array make_f8(const std::vector<int64_t>& shape, const double* p);
Array make_i8(const std::vector<int64_t>& shape, const int64_t* p);
Array make_i4(const std::vector<int64_t>& shape, const int32_t* p);

}  // namespace tabicl::npy
