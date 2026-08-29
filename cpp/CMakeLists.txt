cmake_minimum_required(VERSION 3.20)
project(tabicl_cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(TABICL_BUILD_TESTS    "Build tabicl C++ tests"          ON)
option(TABICL_BUILD_EXAMPLES "Build example programs"          ON)
option(TABICL_BUILD_PYBIND "Build the pybind11 testing module" OFF)
option(TABICL_SANITIZE     "Build with ASan+UBSan"             OFF)
# Route large fp32 matmuls through BLAS (OpenBLAS). Changes accumulation
# order within sgemm; parity suite must stay green at unchanged tolerances
# (it does — see cpp/README). Off by default for strict determinism.
option(TABICL_BLAS         "Use BLAS for fp32 matmuls (via ggml)" OFF)
# "native" tunes for the build machine; anything else is passed to -march verbatim
# (use e.g. "x86-64-v3" for portable CI binaries).
set(TABICL_CPU_ARCH "native" CACHE STRING "Target CPU architecture")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(cmake/warnings.cmake)
include(cmake/sanitizers.cmake)

# --- ggml (vendored submodule, CPU backend only) ------------------------------
set(GGML_METAL  OFF CACHE BOOL "" FORCE)
set(GGML_CUDA   OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
if(TABICL_BLAS)
  set(GGML_BLAS ON CACHE BOOL "" FORCE)
  set(GGML_BLAS_VENDOR "OpenBLAS" CACHE STRING "" FORCE)
else()
  set(GGML_BLAS OFF CACHE BOOL "" FORCE)
endif()
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(TABICL_CPU_ARCH STREQUAL "native")
  set(GGML_NATIVE ON  CACHE BOOL "" FORCE)
else()
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  set(GGML_CPU_ARCH_FLAGS "-march=${TABICL_CPU_ARCH}" CACHE STRING "" FORCE)
endif()
add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# --- tabicl_cpp library -------------------------------------------------------
add_library(tabicl_cpp STATIC
  src/ggml_utils.cpp
  src/model_forward.cpp
  src/estimator_core.cpp
  src/hierarchical.cpp
  src/quantile_dist.cpp
  src/classifier.cpp
  src/regressor.cpp
  src/preprocess/pipeline.cpp
  src/gguf_model.cpp
  src/preprocess/rng_cpython.cpp
  src/preprocess/rng_numpy.cpp
  src/preprocess/ensemble.cpp
  src/preprocess/transforms.cpp
  src/preprocess/cephes_ndtri.cpp
  src/preprocess/sklearn_scaler.cpp
  src/preprocess/power_transform.cpp
  src/preprocess/robust_scaler.cpp
  src/preprocess/quantile_transform.cpp
  src/io/fitted_state.cpp
  src/io/npy.cpp
  src/io/sha256.cpp
)
target_include_directories(tabicl_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(tabicl_cpp PROPERTIES EXPORT_NAME tabicl)
add_library(tabicl::tabicl ALIAS tabicl_cpp)
target_link_libraries(tabicl_cpp PUBLIC ggml)
tabicl_set_warnings(tabicl_cpp)
tabicl_set_sanitizers(tabicl_cpp)

# --- tests --------------------------------------------------------------------
if(TABICL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(TABICL_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(TABICL_BUILD_PYBIND)
  add_subdirectory(bindings)
endif()

# --- install / package config -------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS tabicl_cpp ggml ggml-base ggml-cpu
        EXPORT tabiclTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/tabicl DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT tabiclTargets NAMESPACE tabicl::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tabicl)
configure_package_config_file(cmake/tabiclConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/tabiclConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tabicl)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/tabiclConfigVersion.cmake
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tabiclConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/tabiclConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tabicl)
