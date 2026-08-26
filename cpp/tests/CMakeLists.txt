add_executable(tabicl_tests
  doctest_main.cpp
  test_ggml_assumptions.cpp
  test_gguf_load.cpp
  test_npy.cpp
  test_rng_cpython.cpp
  test_rng_numpy.cpp
  test_ensemble.cpp
  test_preprocess_basic.cpp
  test_normalizers.cpp
  test_model_stages.cpp
  test_e2e.cpp
  test_cache.cpp
  test_chunking.cpp
)
target_include_directories(tabicl_tests PRIVATE ${CMAKE_SOURCE_DIR}/third_party/doctest)
target_compile_definitions(tabicl_tests PRIVATE
  TABICL_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
target_link_libraries(tabicl_tests PRIVATE tabicl_cpp)
tabicl_set_warnings(tabicl_tests)
tabicl_set_sanitizers(tabicl_tests)

add_test(NAME tabicl_tests COMMAND tabicl_tests)
