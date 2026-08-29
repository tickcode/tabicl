# tabicl-cpp

A pure C++ CPU inference library for TabICL — classifier and regressor
`fit`/`predict` end-to-end (preprocessing, ensembling, the transformer, and the
KV cache), with no Python or PyTorch at runtime. The math backend is
[ggml](https://github.com/ggml-org/ggml) (vendored submodule, CPU backend,
fp32).

Numerical parity with the Python package is enforced by an extensive test
suite (~850k assertions): bit-exact RNG and permutation ports, 1e-9..1e-12
preprocessing parity, per-stage transformer parity at 1e-5..5e-4, and
end-to-end `predict_proba`/`predict` parity at `rtol=atol=1e-4` (the repo's
established tolerance). Cached and uncached predictions are bitwise identical.

## Scope

Supported (parity-tested against the Python implementation):

- `TabICLClassifier` / `TabICLRegressor` with numeric `double*` input
  (`NaN` = missing; categorical encoding is the caller's job)
- All five normalization methods: `none`, `power` (Yeo-Johnson), `robust`,
  `quantile`, `quantile_rtdl` — including bit-exact ports of CPython's
  `random.Random`, NumPy's `RandomState` and `PCG64`/ziggurat generators, and
  scipy's Cephes `ndtri`
- \>10-class datasets (mixed-radix column embedding + hierarchical class tree)
- KV cache, both `kv` and `repr` modes: built during `fit`, reused across
  `predict` calls
- v2 checkpoints (classifier + regressor), converted to GGUF via
  `scripts/export_gguf.py`
- Memory-bounded execution: attention and activations are chunked along
  embarrassingly-parallel axes under a configurable scratch budget
  (`EstimatorOptions::max_scratch_bytes`, default 1 GiB). Results are
  **bitwise identical** for any budget (test-asserted); tight budgets degrade
  to slower execution, never failure. 10k training rows run in <700 MB RSS.
- Fitted-estimator serialization: `save(path)` / `load(path, model)` persist
  the complete fitted state (preprocessing, ensemble configs, KV cache) to a
  versioned GGUF file. Loading reproduces predictions **bitwise**; the file
  records a checkpoint fingerprint and refuses to load against a different
  model. This is a C++-only format (no Python pickle interop).
- Regressor quantile tails: `predict_quantiles` accepts any level in `(0, 1)`.
  Levels outside the head's `[0.001, 0.999]` knot grid are extrapolated by the
  `QuantileDistribution` port in `src/quantile_dist.h` — exponential tails (what
  `TabICLRegressor` builds, matching `QuantileToDistribution`'s default) and the
  GPD variant, both parity-tested against Python goldens.

Not (yet) ported: string/categorical input handling, SHAP feature masks,
fine-tuning/forecast/unsupervised, GPU, and v1 checkpoints (interleaved RoPE).

## Quick start (no Python required)

```bash
git clone --recurse-submodules https://github.com/tickcode/tabicl.git
cd tabicl

# Build the library and examples
cmake -B cpp/build -S cpp -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j

# Download a converted model (see the cpp-v0.1.0 GitHub release; verify
# against sha256sums.txt from the same release)
wget https://github.com/tickcode/tabicl/releases/download/cpp-v0.1.0/tabicl-classifier-v2.gguf

# Run the example: fit, predict, save/load the fitted state
./cpp/build/examples/tabicl_quickstart tabicl-classifier-v2.gguf
```

`cpp/examples/quickstart.cpp` is the annotated starting point.

## Using the library from your own project

Install and consume via CMake:

```bash
cmake --install cpp/build --prefix /your/prefix
```

```cmake
find_package(tabicl CONFIG REQUIRED)   # -DCMAKE_PREFIX_PATH=/your/prefix
target_link_libraries(your_app PRIVATE tabicl::tabicl)
```

Public headers: `<tabicl/model.h>`, `<tabicl/options.h>`,
`<tabicl/classifier.h>`, `<tabicl/regressor.h>`. Alternatively, vendor the
repo and `add_subdirectory(tabicl/cpp)` — the same `tabicl::tabicl` target
exists in-tree.

## Build options

- `-DTABICL_BLAS=ON` — route large fp32 matmuls through OpenBLAS. The full
  parity suite passes at unchanged tolerances; mainly speeds up the KV-cache
  build/use path. Off by default for strict run-to-run determinism.
- `-DTABICL_BUILD_PYBIND=ON -Dpybind11_DIR=$(python -m pybind11 --cmakedir)` —
  build the `tabicl_cpp_testing` module used by `tests/test_cpp_parity.py`.
- `-DTABICL_CPU_ARCH=x86-64-v3` — portable binaries (default: `native`).
- `-DTABICL_SANITIZE=ON` — ASan+UBSan build for the tests.

## Checkpoints

Pre-converted GGUF model files (classifier + regressor, with SHA256 sums) are
attached to the `cpp-v0.1.0` GitHub release — that is the recommended path
and needs no Python. To convert a checkpoint yourself (e.g. a fine-tuned
one), a one-time Python step does it:

```bash
pip install torch gguf
python scripts/export_gguf.py /path/to/tabicl-classifier-v2-20260212.ckpt \
       tabicl-classifier-v2.gguf
```

The exporter validates the config against the supported feature matrix,
shortens tensor names (ggml caps names at 64 bytes), and writes a SHA256
manifest that the loader test verifies.

## Usage

```cpp
#include <tabicl/classifier.h>

auto model = tabicl::Model::load("tabicl-classifier-v2.gguf");
tabicl::EstimatorOptions opts;        // src/estimator_core.h
opts.cache = tabicl::CacheMode::KV;   // cache training K/V during fit
opts.n_threads = 16;

tabicl::TabICLClassifier clf(model, opts);
clf.fit(X_train, y_train, n_train, n_features);        // row-major double*
clf.predict_proba(X_test, n_test, probs_out);          // fast: reuses cache

clf.save("fitted.gguf");                               // persist fitted state
auto clf2 = tabicl::TabICLClassifier::load("fitted.gguf", model);
clf2.predict_proba(X_test, n_test, probs_out);         // bitwise-identical
```

## Tests

```bash
# C++ suite (golden fixtures generated by tools/generate_fixtures.py)
python tools/generate_fixtures.py
ctest --test-dir cpp/build --output-on-failure

# Python side-by-side parity matrix
python -m pytest tests/test_cpp_parity.py -v
```

Fixture-driven tests skip cleanly when fixtures are absent. CI runs the full
pipeline (fixture generation -> cmake/ctest incl. sanitizers -> pytest parity)
in `.github/workflows/cpp.yml`.

## Performance (2000x30 train / 500 test, 5 classes, 16 threads)

| | fit | predict |
|---|---|---|
| PyTorch (CPU) | 0.3 s | 21.1 s |
| C++ uncached | 0.02 s | 28.5 s |
| C++ KV cache | 25.8 s | 4.3 s |
| C++ KV cache + BLAS | 18.5 s | 3.0 s |

The KV cache moves the training-data computation into `fit`, making repeated
`predict` calls on the same training data 5-10x faster. Further kernel-level
tuning (threadpool reuse, flash-attention-style streaming) is future work;
every optimization is gated on the parity suite staying green.

## Determinism

Results are bitwise-reproducible across runs and thread counts (asserted by
tests) in the default build. `TABICL_BLAS=ON` keeps parity tolerances but may
vary in the last bits across BLAS thread counts. Note that the *Python*
implementation's ensemble grouping order depends on `PYTHONHASHSEED`, so
Python's own outputs vary at last-bit level between processes; comparisons are
tolerance-based (1e-4) for that reason.
