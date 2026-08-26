// Minimal SHA-256 (FIPS 180-4) for fixture/manifest integrity checks.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tabicl {

// Hex digest of `len` bytes at `data`.
std::string sha256_hex(const void* data, size_t len);

}  // namespace tabicl
