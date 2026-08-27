"""Side-by-side parity tests: C++ TabICL port (cpp/) vs the Python package.

Requires the pybind test module and exported GGUF checkpoints:
    cmake -B cpp/build -S cpp -DTABICL_BUILD_PYBIND=ON \
          -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
    cmake --build cpp/build -j
    python scripts/export_gguf.py <ckpt> cpp/tests/fixtures/tabicl-classifier-v2.gguf

Skipped automatically when those artifacts are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "cpp" / "build" / "bindings"
FIXTURES = REPO / "cpp" / "tests" / "fixtures"
CLF_GGUF = FIXTURES / "tabicl-classifier-v2.gguf"
REG_GGUF = FIXTURES / "tabicl-regressor-v2.gguf"

sys.path.insert(0, str(BUILD))
tcpp = pytest.importorskip("tabicl_cpp_testing")

pytestmark = pytest.mark.skipif(
    not (CLF_GGUF.exists() and REG_GGUF.exists()),
    reason="GGUF fixtures missing (run scripts/export_gguf.py)",
)

RTOL = ATOL = 1e-4


@pytest.fixture(scope="module")
def clf_model():
    return tcpp.Model.load(str(CLF_GGUF))


@pytest.fixture(scope="module")
def reg_model():
    return tcpp.Model.load(str(REG_GGUF))


def make_data(seed, n_train, n_test, d, n_classes, nan_frac=0.0):
    rng = np.random.RandomState(seed)
    Xtr = rng.randn(n_train, d) * rng.uniform(0.5, 3, size=d)
    Xte = rng.randn(n_test, d) * rng.uniform(0.5, 3, size=d)
    if nan_frac > 0:
        Xtr[rng.rand(*Xtr.shape) < nan_frac] = np.nan
        Xte[rng.rand(*Xte.shape) < nan_frac] = np.nan
    if n_classes > 0:
        y = rng.randint(0, n_classes, size=n_train).astype(float)
    else:
        y = rng.randn(n_train) * 3 + 1
    return Xtr, y, Xte


def python_clf_proba(Xtr, y, Xte, **kwargs):
    from tabicl import TabICLClassifier

    est = TabICLClassifier(device="cpu", use_amp=False, use_fa3=False,
                           random_state=42, **kwargs)
    est.fit(Xtr, y)
    return est.predict_proba(Xte)


CLF_CASES = [
    # (name, seed, n_train, n_test, d, n_classes, n_estimators, norm_methods, nan)
    ("defaults", 0, 64, 16, 5, 3, 8, None, 0.0),
    ("one_estimator", 1, 48, 12, 4, 2, 1, None, 0.0),
    ("four_estimators_nan", 2, 56, 14, 6, 4, 4, None, 0.1),
    ("robust_quantile", 3, 80, 10, 5, 3, 6, ["robust", "quantile"], 0.0),
]


@pytest.mark.parametrize("case", CLF_CASES, ids=[c[0] for c in CLF_CASES])
def test_classifier_parity(clf_model, case):
    name, seed, ntr, nte, d, ncls, n_est, methods, nan = case
    Xtr, y, Xte = make_data(seed, ntr, nte, d, ncls, nan)
    ref = python_clf_proba(Xtr, y, Xte, n_estimators=n_est, norm_methods=methods)

    for cache in ["none", "kv", "repr"]:
        clf = tcpp.Classifier(clf_model, n_estimators=n_est,
                              norm_methods=methods or [], cache=cache, n_threads=8)
        clf.fit(Xtr, y)
        proba = clf.predict_proba(Xte)
        np.testing.assert_allclose(proba, ref, rtol=RTOL, atol=ATOL,
                                   err_msg=f"{name} cache={cache}")
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_classifier_batch_size_invariance(clf_model):
    Xtr, y, Xte = make_data(4, 60, 10, 5, 3)
    ref = None
    for bs in [3, 8]:
        clf = tcpp.Classifier(clf_model, batch_size=bs, n_threads=8)
        clf.fit(Xtr, y)
        proba = clf.predict_proba(Xte)
        if ref is None:
            ref = proba
        else:
            # Different batch compositions only reorder fp32 op batching.
            np.testing.assert_allclose(proba, ref, rtol=1e-4, atol=1e-4)


def test_classifier_deterministic_and_thread_invariant(clf_model):
    Xtr, y, Xte = make_data(5, 50, 10, 4, 3)
    ref = None
    for n_threads in [1, 8, 8]:
        clf = tcpp.Classifier(clf_model, n_threads=n_threads)
        clf.fit(Xtr, y)
        proba = clf.predict_proba(Xte)
        if ref is None:
            ref = proba
        else:
            np.testing.assert_array_equal(proba, ref)  # bitwise


def test_classifier_label_encoding(clf_model):
    Xtr, y, Xte = make_data(6, 40, 8, 4, 3)
    y_labels = y * 7 - 5  # non-contiguous labels
    clf = tcpp.Classifier(clf_model, n_threads=8)
    clf.fit(Xtr, y_labels)
    assert sorted(clf.classes_) == sorted(set(y_labels))
    pred = clf.predict(Xte)
    assert set(pred).issubset(set(y_labels))


def test_fitted_state_roundtrip(clf_model, tmp_path):
    """save -> load reproduces predictions bitwise (KV-cache mode)."""
    Xtr, y, Xte = make_data(8, 50, 10, 4, 3)
    clf = tcpp.Classifier(clf_model, cache="kv", n_threads=8)
    clf.fit(Xtr, y)
    before = clf.predict_proba(Xte)
    path = str(tmp_path / "fitted.gguf")
    clf.save(path)
    loaded = tcpp.Classifier.load(path, clf_model)
    after = loaded.predict_proba(Xte)
    np.testing.assert_array_equal(before, after)  # bitwise
    assert loaded.classes_ == clf.classes_


def test_regressor_parity(reg_model):
    from tabicl import TabICLRegressor

    Xtr, y, Xte = make_data(7, 70, 15, 5, 0)
    est = TabICLRegressor(device="cpu", use_amp=False, use_fa3=False, random_state=42)
    est.fit(Xtr, y)
    mean_ref = np.asarray(est.predict(Xte), dtype=float)
    alphas = [0.1, 0.5, 0.9]
    q_ref = np.asarray(est.predict(Xte, output_type="quantiles", alphas=alphas))

    scale = np.abs(mean_ref).max()
    for cache in ["none", "kv", "repr"]:
        reg = tcpp.Regressor(reg_model, cache=cache, n_threads=8)
        reg.fit(Xtr, y)
        mean = reg.predict(Xte)
        np.testing.assert_allclose(mean, mean_ref, rtol=RTOL, atol=ATOL * scale,
                                   err_msg=f"mean cache={cache}")
        q = reg.predict_quantiles(Xte, np.asarray(alphas))
        np.testing.assert_allclose(q, q_ref, rtol=RTOL, atol=ATOL * scale,
                                   err_msg=f"quantiles cache={cache}")
