#include "io/npy.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace tabicl::npy {

namespace {

constexpr uint8_t kMagic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};

size_t dtype_size(DType t) {
  switch (t) {
    case DType::F4: case DType::I4: return 4;
    case DType::F8: case DType::I8: return 8;
    case DType::I1: case DType::U1: case DType::B1: return 1;
  }
  throw std::runtime_error("npy: bad dtype");
}

const char* dtype_descr(DType t) {
  switch (t) {
    case DType::F4: return "<f4";
    case DType::F8: return "<f8";
    case DType::I4: return "<i4";
    case DType::I8: return "<i8";
    case DType::I1: return "|i1";
    case DType::U1: return "|u1";
    case DType::B1: return "|b1";
  }
  throw std::runtime_error("npy: bad dtype");
}

DType parse_descr(const std::string& d) {
  if (d == "<f4") return DType::F4;
  if (d == "<f8") return DType::F8;
  if (d == "<i4") return DType::I4;
  if (d == "<i8") return DType::I8;
  if (d == "|i1" || d == "<i1") return DType::I1;
  if (d == "|u1" || d == "<u1") return DType::U1;
  if (d == "|b1") return DType::B1;
  throw std::runtime_error("npy: unsupported descr '" + d + "'");
}

// Extract the value following `'key': ` in the header dict.
std::string dict_value(const std::string& header, const std::string& key) {
  const std::string pat = "'" + key + "':";
  size_t p = header.find(pat);
  if (p == std::string::npos) throw std::runtime_error("npy: header missing key " + key);
  p += pat.size();
  while (p < header.size() && header[p] == ' ') p++;
  return header.substr(p);
}

}  // namespace

size_t Array::itemsize() const { return dtype_size(dtype); }

#define TABICL_NPY_ACCESSOR(fn, T, DT1, DT2)                                     \
  const T* Array::fn() const {                                                   \
    if (dtype != DT1 && dtype != DT2)                                            \
      throw std::runtime_error("npy: dtype mismatch in accessor " #fn);          \
    return reinterpret_cast<const T*>(data.data());                              \
  }
TABICL_NPY_ACCESSOR(f4, float, DType::F4, DType::F4)
TABICL_NPY_ACCESSOR(f8, double, DType::F8, DType::F8)
TABICL_NPY_ACCESSOR(i4, int32_t, DType::I4, DType::I4)
TABICL_NPY_ACCESSOR(i8, int64_t, DType::I8, DType::I8)
TABICL_NPY_ACCESSOR(i1, int8_t, DType::I1, DType::I1)
TABICL_NPY_ACCESSOR(u1, uint8_t, DType::U1, DType::B1)
#undef TABICL_NPY_ACCESSOR

Array parse(const uint8_t* bytes, size_t len) {
  if (len < 10 || std::memcmp(bytes, kMagic, 6) != 0)
    throw std::runtime_error("npy: bad magic");
  const uint8_t major = bytes[6];
  size_t header_len, header_off;
  if (major == 1) {
    if (len < 10) throw std::runtime_error("npy: truncated");
    header_len = bytes[8] | (bytes[9] << 8);
    header_off = 10;
  } else if (major == 2) {
    if (len < 12) throw std::runtime_error("npy: truncated");
    header_len = static_cast<size_t>(bytes[8]) | (static_cast<size_t>(bytes[9]) << 8) |
                 (static_cast<size_t>(bytes[10]) << 16) | (static_cast<size_t>(bytes[11]) << 24);
    header_off = 12;
  } else {
    throw std::runtime_error("npy: unsupported version");
  }
  if (header_off + header_len > len) throw std::runtime_error("npy: truncated header");
  const std::string header(reinterpret_cast<const char*>(bytes + header_off), header_len);

  // fortran_order
  {
    std::string v = dict_value(header, "fortran_order");
    if (v.rfind("False", 0) != 0)
      throw std::runtime_error("npy: fortran_order unsupported");
  }
  Array a;
  // descr
  {
    std::string v = dict_value(header, "descr");
    if (v.empty() || v[0] != '\'') throw std::runtime_error("npy: bad descr");
    size_t end = v.find('\'', 1);
    a.dtype = parse_descr(v.substr(1, end - 1));
  }
  // shape
  {
    std::string v = dict_value(header, "shape");
    if (v.empty() || v[0] != '(') throw std::runtime_error("npy: bad shape");
    size_t end = v.find(')');
    std::string tup = v.substr(1, end - 1);
    size_t pos = 0;
    while (pos < tup.size()) {
      while (pos < tup.size() && (tup[pos] == ' ' || tup[pos] == ',')) pos++;
      if (pos >= tup.size()) break;
      size_t next;
      a.shape.push_back(std::stoll(tup.substr(pos), &next));
      pos += next;
    }
  }
  const size_t nbytes = static_cast<size_t>(a.numel()) * a.itemsize();
  const size_t data_off = header_off + header_len;
  if (data_off + nbytes > len) throw std::runtime_error("npy: truncated data");
  a.data.assign(bytes + data_off, bytes + data_off + nbytes);
  return a;
}

Array load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("npy: cannot open " + path);
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  return parse(buf.data(), buf.size());
}

void save(const std::string& path, const Array& a) {
  std::string shape = "(";
  for (size_t i = 0; i < a.shape.size(); ++i) shape += std::to_string(a.shape[i]) + ",";
  if (a.shape.size() > 1) shape.pop_back();
  shape += ")";
  std::string dict = std::string("{'descr': '") + dtype_descr(a.dtype) +
                     "', 'fortran_order': False, 'shape': " + shape + ", }";
  // pad so that magic(6)+ver(2)+len(2)+header is a multiple of 64, newline-terminated
  size_t unpadded = 10 + dict.size() + 1;
  size_t pad = (64 - unpadded % 64) % 64;
  dict += std::string(pad, ' ');
  dict += '\n';
  if (dict.size() > 0xFFFF) throw std::runtime_error("npy: header too large");

  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("npy: cannot write " + path);
  f.write(reinterpret_cast<const char*>(kMagic), 6);
  const uint8_t ver[2] = {1, 0};
  f.write(reinterpret_cast<const char*>(ver), 2);
  const uint16_t hlen = static_cast<uint16_t>(dict.size());
  const uint8_t lenb[2] = {static_cast<uint8_t>(hlen & 0xFF), static_cast<uint8_t>(hlen >> 8)};
  f.write(reinterpret_cast<const char*>(lenb), 2);
  f.write(dict.data(), static_cast<std::streamsize>(dict.size()));
  f.write(reinterpret_cast<const char*>(a.data.data()), static_cast<std::streamsize>(a.data.size()));
  if (!f) throw std::runtime_error("npy: write failed " + path);
}

namespace {
Array make(DType t, const std::vector<int64_t>& shape, const void* p) {
  Array a;
  a.dtype = t;
  a.shape = shape;
  const uint8_t* b = static_cast<const uint8_t*>(p);
  a.data.assign(b, b + a.numel() * a.itemsize());
  return a;
}
}  // namespace

Array make_f4(const std::vector<int64_t>& shape, const float* p) { return make(DType::F4, shape, p); }
Array make_f8(const std::vector<int64_t>& shape, const double* p) { return make(DType::F8, shape, p); }
Array make_i8(const std::vector<int64_t>& shape, const int64_t* p) { return make(DType::I8, shape, p); }
Array make_i4(const std::vector<int64_t>& shape, const int32_t* p) { return make(DType::I4, shape, p); }

}  // namespace tabicl::npy
