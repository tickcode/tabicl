// Shared helpers for fixture-driven tests.
#pragma once

#include <cstdlib>
#include <fstream>
#include <string>

#include "io/npy.h"

#ifndef TABICL_FIXTURES_DIR
#define TABICL_FIXTURES_DIR "fixtures"
#endif

namespace tabicl::test {

inline std::string fixtures_dir() {
  const char* env = std::getenv("TABICL_FIXTURES_DIR");
  return env ? std::string(env) : std::string(TABICL_FIXTURES_DIR);
}

inline std::string fixture_path(const std::string& rel) {
  return fixtures_dir() + "/" + rel;
}

inline bool fixture_exists(const std::string& rel) {
  return std::ifstream(fixture_path(rel)).good();
}

inline npy::Array load_fixture(const std::string& rel) {
  return npy::load(fixture_path(rel));
}

}  // namespace tabicl::test
