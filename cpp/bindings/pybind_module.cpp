// Test-only pybind11 module exposing the C++ TabICL estimators to pytest
// for side-by-side parity testing against the Python implementation.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "estimator_core.h"
#include "tabicl/classifier.h"
#include "tabicl/model.h"
#include "tabicl/regressor.h"

namespace py = pybind11;

namespace {

tabicl::EstimatorOptions make_options(int n_estimators,
                                      std::vector<std::string> norm_methods,
                                      const std::string& cache, int batch_size,
                                      int n_threads, uint64_t random_state) {
  tabicl::EstimatorOptions o;
  o.n_estimators = n_estimators;
  if (!norm_methods.empty()) o.norm_methods = std::move(norm_methods);
  if (cache == "kv") o.cache = tabicl::CacheMode::KV;
  else if (cache == "repr") o.cache = tabicl::CacheMode::Repr;
  else if (cache == "none") o.cache = tabicl::CacheMode::None;
  else throw std::runtime_error("cache must be none|kv|repr");
  o.batch_size = batch_size;
  o.n_threads = n_threads;
  o.random_state = random_state;
  return o;
}

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

}  // namespace

PYBIND11_MODULE(tabicl_cpp_testing, m) {
  m.doc() = "TabICL C++ port test bindings";

  py::class_<tabicl::Model, std::shared_ptr<tabicl::Model>>(m, "Model")
      .def_static("load", &tabicl::Model::load)
      .def_property_readonly("n_tensors", &tabicl::Model::n_tensors);

  py::class_<tabicl::TabICLClassifier>(m, "Classifier")
      .def(py::init([](std::shared_ptr<tabicl::Model> model, int n_estimators,
                       std::vector<std::string> norm_methods, const std::string& cache,
                       int batch_size, int n_threads, uint64_t random_state) {
             return new tabicl::TabICLClassifier(
                 std::move(model),
                 make_options(n_estimators, std::move(norm_methods), cache,
                              batch_size, n_threads, random_state));
           }),
           py::arg("model"), py::arg("n_estimators") = 8,
           py::arg("norm_methods") = std::vector<std::string>{},
           py::arg("cache") = "none", py::arg("batch_size") = 8,
           py::arg("n_threads") = 0, py::arg("random_state") = 42)
      .def("fit",
           [](tabicl::TabICLClassifier& self, Arr X, Arr y) {
             if (X.ndim() != 2 || y.ndim() != 1 || X.shape(0) != y.shape(0))
               throw std::runtime_error("fit: bad shapes");
             self.fit(X.data(), y.data(), X.shape(0), X.shape(1));
           })
      .def("predict_proba",
           [](const tabicl::TabICLClassifier& self, Arr X) {
             const int64_t n = X.shape(0);
             py::array_t<float> out({n, self.n_classes()});
             self.predict_proba(X.data(), n, out.mutable_data());
             return out;
           })
      .def("predict",
           [](const tabicl::TabICLClassifier& self, Arr X) {
             const int64_t n = X.shape(0);
             py::array_t<double> out(n);
             self.predict(X.data(), n, out.mutable_data());
             return out;
           })
      .def_property_readonly("classes_", &tabicl::TabICLClassifier::classes)
      .def("save", &tabicl::TabICLClassifier::save)
      .def_static("load", &tabicl::TabICLClassifier::load, py::arg("path"),
                  py::arg("model"), py::arg("n_threads_override") = -1);

  py::class_<tabicl::TabICLRegressor>(m, "Regressor")
      .def(py::init([](std::shared_ptr<tabicl::Model> model, int n_estimators,
                       std::vector<std::string> norm_methods, const std::string& cache,
                       int batch_size, int n_threads, uint64_t random_state) {
             return new tabicl::TabICLRegressor(
                 std::move(model),
                 make_options(n_estimators, std::move(norm_methods), cache,
                              batch_size, n_threads, random_state));
           }),
           py::arg("model"), py::arg("n_estimators") = 8,
           py::arg("norm_methods") = std::vector<std::string>{},
           py::arg("cache") = "none", py::arg("batch_size") = 8,
           py::arg("n_threads") = 0, py::arg("random_state") = 42)
      .def("fit",
           [](tabicl::TabICLRegressor& self, Arr X, Arr y) {
             if (X.ndim() != 2 || y.ndim() != 1 || X.shape(0) != y.shape(0))
               throw std::runtime_error("fit: bad shapes");
             self.fit(X.data(), y.data(), X.shape(0), X.shape(1));
           })
      .def("predict",
           [](const tabicl::TabICLRegressor& self, Arr X) {
             const int64_t n = X.shape(0);
             py::array_t<double> out(n);
             self.predict(X.data(), n, out.mutable_data());
             return out;
           })
      .def("predict_quantiles",
           [](const tabicl::TabICLRegressor& self, Arr X, Arr alphas) {
             const int64_t n = X.shape(0);
             const int64_t na = alphas.shape(0);
             py::array_t<double> out({n, na});
             self.predict_quantiles(X.data(), n, alphas.data(), na,
                                    out.mutable_data());
             return out;
           })
      .def("save", &tabicl::TabICLRegressor::save)
      .def_static("load", &tabicl::TabICLRegressor::load, py::arg("path"),
                  py::arg("model"), py::arg("n_threads_override") = -1);
}
