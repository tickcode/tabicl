#!/usr/bin/env python3
"""Generate golden parity fixtures for the C++ port (cpp/).

Each fixture group is a directory of .npy files under cpp/tests/fixtures/.
Deterministic: fixed seeds, CPU, fp32/fp64, PYTHONHASHSEED pinned by re-exec.

Sections can be generated selectively:
    python tools/generate_fixtures.py --only rng
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
from pathlib import Path

# Norm-method ordering inside tabicl uses `list(set(...))`, which depends on
# PYTHONHASHSEED. Pin it and re-exec so fixtures are reproducible.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np

FIXTURES = Path(__file__).resolve().parent.parent / "cpp" / "tests" / "fixtures"


def save(group: str, name: str, arr: np.ndarray) -> None:
    d = FIXTURES / group
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"{name}.npy", np.ascontiguousarray(arr))


def save_manifest(group: str, info: dict) -> None:
    (FIXTURES / group).mkdir(parents=True, exist_ok=True)
    (FIXTURES / group / "manifest.json").write_text(json.dumps(info, indent=1, sort_keys=True))


# --------------------------------------------------------------------------
# RNG goldens
# --------------------------------------------------------------------------

RNG_SEEDS = [0, 1, 42, 12345, 2**31 - 1, 2**32 - 1, 2**64 + 12345]


def gen_rng_cpython() -> None:
    group = "rng_cpython"
    for si, seed in enumerate(RNG_SEEDS):
        r = random.Random(seed)
        save(group, f"random_s{si}", np.array([r.random() for _ in range(1000)], dtype=np.float64))
        for k in (1, 8, 31, 32, 33, 64, 128):
            r = random.Random(seed)
            draws = [r.getrandbits(k) for _ in range(200)]
            # store as decimal strings via two u64 limbs for k<=128
            lo = np.array([d & (2**64 - 1) for d in draws], dtype=np.uint64)
            hi = np.array([d >> 64 for d in draws], dtype=np.uint64)
            save(group, f"getrandbits_k{k}_s{si}", np.stack([lo.view(np.int64), hi.view(np.int64)]))
        for n in (1, 2, 3, 10, 57, 1000):
            r = random.Random(seed)
            x = list(range(n))
            r.shuffle(x)
            save(group, f"shuffle_n{n}_s{si}", np.array(x, dtype=np.int64))
        # sample: small-k (selection-set branch) and pool branch
        for n, k in ((10, 3), (1000, 5), (24, 20), (120, 100), (5, 5)):
            r = random.Random(seed)
            save(group, f"sample_n{n}_k{k}_s{si}",
                 np.array(r.sample(range(n), k), dtype=np.int64))
        r = random.Random(seed)
        save(group, f"choice_s{si}",
             np.array([r.choice(range(97)) for _ in range(500)], dtype=np.int64))
    save_manifest(group, {"seeds": [str(s) for s in RNG_SEEDS]})


def gen_rng_numpy() -> None:
    group = "rng_numpy"
    for si, seed in enumerate(RNG_SEEDS):
        # Legacy RandomState (MT19937): raw draws + shuffle/permutation
        rs = np.random.RandomState(seed % 2**32)
        save(group, f"rs_randint_s{si}",
             rs.randint(0, 2**32, size=500, dtype=np.uint64).astype(np.int64))
        for n in (2, 10, 1000, 10001):
            rs = np.random.RandomState(seed % 2**32)
            idx = np.arange(n)
            rs.shuffle(idx)
            save(group, f"rs_shuffle_n{n}_s{si}", idx.astype(np.int64))
        # PCG64 Generator: raw stream + standard_normal (ziggurat)
        bg = np.random.PCG64(seed)
        save(group, f"pcg64_raw_s{si}", bg.random_raw(500).view(np.int64))
        g = np.random.Generator(np.random.PCG64(seed))
        save(group, f"pcg64_normal_s{si}", g.standard_normal(2000))
        g = np.random.Generator(np.random.PCG64(seed))
        save(group, f"pcg64_uniform_s{si}", g.random(1000))
    save_manifest(group, {"seeds": [str(s) for s in RNG_SEEDS]})


def gen_shuffler() -> None:
    """EnsembleGenerator/Shuffler permutation goldens (pure CPython RNG)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from tabicl._sklearn.preprocessing import Shuffler

    group = "shuffler"
    for si, seed in enumerate([0, 42, 12345]):
        for n_elem in (1, 2, 3, 5, 8, 50):
            for method in ("latin", "random", "shift"):
                sh = Shuffler(n_elements=n_elem, method=method, random_state=seed)
                pats = sh.shuffle(8)
                save(group, f"{method}_n{n_elem}_s{si}",
                     np.array([list(p) for p in pats], dtype=np.int64))
        # latin->random fallback above 4000 elements (store first 3 perms only)
        sh = Shuffler(n_elements=4001, method="latin", random_state=seed)
        pats = sh.shuffle(3)
        save(group, f"latin_fallback_n4001_s{si}",
             np.array([list(p) for p in pats], dtype=np.int64))
    save_manifest(group, {"note": "Shuffler.shuffle(8) patterns; latin fallback uses 3"})


def gen_ensemble_configs() -> None:
    """Full EnsembleGenerator member configs for default settings."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from tabicl._sklearn.preprocessing import EnsembleGenerator

    group = "ensemble_configs"
    cases = [
        ("defaults", dict(classification=True, n_estimators=8,
                          norm_methods=["none", "power"], feat_shuffle_method="latin",
                          class_shuffle_method="shift", outlier_threshold=4.0,
                          random_state=42), 50, 5, 3),
        ("one_est", dict(classification=True, n_estimators=1,
                         norm_methods=["none", "power"], feat_shuffle_method="latin",
                         class_shuffle_method="shift", outlier_threshold=4.0,
                         random_state=42), 50, 5, 3),
        ("reg", dict(classification=False, n_estimators=8,
                     norm_methods=["none", "power"], feat_shuffle_method="latin",
                     class_shuffle_method="shift", outlier_threshold=4.0,
                     random_state=0), 60, 7, 0),
        ("all_norms", dict(classification=True, n_estimators=16,
                           norm_methods=["none", "power", "robust", "quantile", "quantile_rtdl"],
                           feat_shuffle_method="latin", class_shuffle_method="shift",
                           outlier_threshold=4.0, random_state=7), 100, 6, 4),
    ]
    rng = np.random.RandomState(123)
    meta = {}
    for name, kwargs, n, d, n_classes in cases:
        X = rng.randn(n, d)
        y = rng.randint(0, n_classes, size=n) if kwargs["classification"] else rng.randn(n)
        eg = EnsembleGenerator(**kwargs)
        eg.fit(X, y)
        # ensemble_configs_: OrderedDict norm_method -> list of (feat_shuffle, class_shuffle)
        order = list(eg.ensemble_configs_.keys())
        feats, classes, method_idx = [], [], []
        for m in order:
            for fs, cs in eg.ensemble_configs_[m]:
                feats.append(list(fs))
                classes.append(list(cs) if cs is not None else [])
                method_idx.append(kwargs["norm_methods"].index(m))
        save(group, f"{name}_feat_shuffles", np.array(feats, dtype=np.int64))
        save(group, f"{name}_member_method_idx", np.array(method_idx, dtype=np.int64))
        if kwargs["classification"]:
            save(group, f"{name}_class_shuffles", np.array(classes, dtype=np.int64))
        meta[name] = {"norm_method_order": order,
                      "n": n, "d": d, "n_classes": n_classes,
                      "kwargs": {k: v for k, v in kwargs.items()}}
    save_manifest(group, meta)


def _basic_datasets():
    """Edge-case matrices shared by preprocessing fixture sections."""
    rng = np.random.RandomState(77)
    ds = {}
    ds["plain"] = rng.randn(200, 12) * rng.uniform(0.1, 50, size=12) + rng.uniform(-30, 30, size=12)
    X = rng.randn(60, 6)
    X[:, 2] = 3.14  # constant column
    X[5:9, 0] = np.nan
    X[:, 4] = np.nan  # all-NaN column
    X[10, 1] = 1e6  # outlier
    X[11, 1] = -1e6
    ds["edge"] = X
    ds["single_col"] = rng.randn(40, 1)
    ds["tiny"] = rng.randn(1, 4)  # n <= threshold for the unique filter
    X = rng.randn(80, 5)
    X[:, 3] = np.round(X[:, 3])  # heavy ties
    ds["ties"] = X
    ds["large"] = rng.randn(12000, 4)  # n > 10000: QT subsampling territory
    return ds


def gen_preprocess_basic() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from sklearn.impute import SimpleImputer
    from tabicl._sklearn.preprocessing import (CustomStandardScaler, OutlierRemover,
                                               UniqueFeatureFilter)

    group = "preprocess_basic"
    for name, X in _basic_datasets().items():
        save(group, f"{name}_X", X)
        # SimpleImputer(mean): may drop all-NaN columns
        imp = SimpleImputer()
        Xi = imp.fit_transform(X)
        save(group, f"{name}_imputed", Xi)
        save(group, f"{name}_imputer_stats", imp.statistics_)
        # UniqueFeatureFilter on imputed data
        uf = UniqueFeatureFilter()
        Xu = uf.fit_transform(Xi)
        save(group, f"{name}_filter_keep", uf.features_to_keep_.astype(np.int64))
        # CustomStandardScaler
        ss = CustomStandardScaler()
        Xs = ss.fit_transform(Xu)
        save(group, f"{name}_scaler_mean", ss.mean_)
        save(group, f"{name}_scaler_scale", ss.scale_)
        save(group, f"{name}_scaled", Xs)
        # OutlierRemover on the scaled data (pipeline order)
        outl = OutlierRemover(threshold=4.0)
        Xo = outl.fit_transform(Xs)
        save(group, f"{name}_outlier_lower", outl.lower_bounds_)
        save(group, f"{name}_outlier_upper", outl.upper_bounds_)
        save(group, f"{name}_outlier_out", Xo)
    save_manifest(group, {"datasets": sorted(_basic_datasets().keys())})


def gen_normalizers() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from scipy.stats import norm
    from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
    from sklearn.preprocessing import QuantileTransformer
    from tabicl._sklearn.preprocessing import RTDLQuantileTransformer
    from sklearn.pipeline import Pipeline

    group = "normalizers"
    # ndtri sweep incl. tails and the QT clip constants
    ps = np.concatenate([
        np.linspace(1e-9, 1 - 1e-9, 2001),
        np.array([1e-300, 1e-30, 1e-14, 1e-7, 1e-7 - np.spacing(1.0),
                  1 - 1e-7, 0.13533528323661269189, 0.25, 0.5, 0.75]),
    ])
    save(group, "ndtri_p", ps)
    save(group, "ndtri_ref", norm.ppf(ps))

    rng = np.random.RandomState(9)
    for name, X in _basic_datasets().items():
        if name in ("edge",):  # NaNs: normalizers run post-imputation
            continue
        # Emulate pipeline entry order: F-contiguous like Python post-filter
        Xf = np.asfortranarray(X)
        save(group, f"{name}_X", X)

        ss = StandardScaler()
        Xs = ss.fit_transform(np.asfortranarray(Xf.copy()))
        save(group, f"{name}_std_mean", ss.mean_)
        save(group, f"{name}_std_scale", ss.scale_)
        save(group, f"{name}_std_out", Xs)

        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        Xp = pt.fit_transform(np.asfortranarray(Xf.copy()))
        save(group, f"{name}_yj_lambdas", pt.lambdas_)
        save(group, f"{name}_yj_out", Xp)
        # transform path (differs from fit_transform for constant cols)
        save(group, f"{name}_yj_out2", pt.transform(np.asfortranarray(Xf.copy())))

        rs = RobustScaler(unit_variance=True)
        Xr = rs.fit_transform(np.asfortranarray(Xf.copy()))
        save(group, f"{name}_robust_center", rs.center_)
        save(group, f"{name}_robust_scale", rs.scale_)
        save(group, f"{name}_robust_out", Xr)

        qt = QuantileTransformer(output_distribution="normal", random_state=42)
        Xq = qt.fit_transform(np.asfortranarray(Xf.copy()))
        save(group, f"{name}_qt_quantiles", qt.quantiles_)
        save(group, f"{name}_qt_out", Xq)

        rtdl = Pipeline([
            ("quantile_rtdl", RTDLQuantileTransformer(random_state=42)),
            ("std", StandardScaler()),
        ])
        Xrt = rtdl.fit_transform(np.asfortranarray(Xf.copy()))
        save(group, f"{name}_rtdl_out", Xrt)
        # test-path transform on shifted data (exercises interp on unseen values)
        X_test = X + rng.randn(*X.shape) * 0.1
        save(group, f"{name}_Xtest", X_test)
        save(group, f"{name}_qt_test_out", qt.transform(np.asfortranarray(X_test.copy())))
        save(group, f"{name}_rtdl_test_out", rtdl.transform(np.asfortranarray(X_test.copy())))
        save(group, f"{name}_yj_test_out", pt.transform(np.asfortranarray(X_test.copy())))
    save_manifest(group, {"note": "normalizer fixtures on F-order inputs"})


def gen_model_small() -> None:
    """Stage-level golden tensors from the real TabICL model (CPU fp32)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import torch
    from tabicl._model.tabicl import TabICL
    from tabicl._model.inference_config import InferenceConfig

    group = "model_small"
    torch.manual_seed(0)

    import glob
    for task, pat in [("clf", "tabicl-classifier-v2-*.ckpt"),
                      ("reg", "tabicl-regressor-v2-*.ckpt")]:
        ckpt_path = sorted(glob.glob(
            "/data/huggingface/hub/models--jingang--TabICL/snapshots/*/" + pat))[-1]
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = TabICL(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        B, T, H, train = 2, 40, 6, 28
        rng = np.random.RandomState(3)
        X = rng.randn(B, T, H).astype(np.float32) * 1.7
        if task == "clf":
            y = rng.randint(0, 3, size=(B, train)).astype(np.float32)
            n_classes = 3
        else:
            y = rng.randn(B, train).astype(np.float32)
            n_classes = 0

        Xt = torch.from_numpy(X)
        yt = torch.from_numpy(y)
        cfg = InferenceConfig()
        cfg.update_from_dict({
            "COL_CONFIG": {"device": "cpu", "use_amp": False, "use_fa3": False},
            "ROW_CONFIG": {"device": "cpu", "use_amp": False, "use_fa3": False},
            "ICL_CONFIG": {"device": "cpu", "use_amp": False, "use_fa3": False},
        })

        captures = {}
        def cap(name):
            def hook(mod, args, out):
                captures[name] = out.detach().float().clone()
            return hook
        hooks = [
            model.col_embedder.register_forward_hook(cap("col_out")),
            model.row_interactor.register_forward_hook(cap("row_out")),
        ]
        with torch.no_grad():
            logits = model(Xt, yt, return_logits=True, inference_config=cfg)
            probs = (model(Xt, yt, return_logits=False, inference_config=cfg)
                     if task == "clf" else None)
        for h in hooks:
            h.remove()

        save(group, f"{task}_X", X)
        save(group, f"{task}_y", y)
        # col_out: (B, T, G+C, E) — save only the real feature slots [C:]
        col = captures["col_out"]
        C = model.row_interactor.num_cls
        save(group, f"{task}_col_out", col[:, :, C:, :].numpy())
        save(group, f"{task}_row_out", captures["row_out"].numpy())
        save(group, f"{task}_logits", logits.numpy())
        if probs is not None:
            save(group, f"{task}_probs", probs.numpy())
        save_manifest(group, {"B": B, "T": T, "H": H, "train": train,
                              "n_classes": n_classes})


def gen_e2e() -> None:
    """Full Python fit/predict outputs for end-to-end estimator parity."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from tabicl import TabICLClassifier, TabICLRegressor

    group = "e2e"
    rng = np.random.RandomState(11)
    common = dict(device="cpu", use_amp=False, use_fa3=False, random_state=42)

    cases = []
    # (name, n_train, n_test, d, n_classes, norm_methods, n_estimators)
    cases.append(("clf_small", 60, 15, 5, 3, None, 8))
    cases.append(("clf_allnorm", 90, 20, 6, 4,
                  ["none", "power", "robust", "quantile", "quantile_rtdl"], 10))
    cases.append(("clf_binary_nan", 50, 12, 4, 2, None, 4))
    cases.append(("reg_small", 70, 18, 5, 0, None, 8))
    cases.append(("clf_many", 140, 20, 8, 14, None, 4))
    meta = {}
    for name, ntr, nte, d, ncls, methods, n_est in cases:
        Xtr = rng.randn(ntr, d) * rng.uniform(0.5, 4, size=d) + rng.uniform(-3, 3, size=d)
        Xte = rng.randn(nte, d) * rng.uniform(0.5, 4, size=d) + rng.uniform(-3, 3, size=d)
        if "nan" in name:
            Xtr[rng.rand(*Xtr.shape) < 0.08] = np.nan
            Xte[rng.rand(*Xte.shape) < 0.08] = np.nan
        if ncls > 0:
            # labels with gaps to exercise LabelEncoder (e.g. 3, 7, 11, ...)
            y = (rng.randint(0, ncls, size=ntr) * 4 + 3).astype(np.float64)
        else:
            y = rng.randn(ntr) * 5.0 + 2.0
        save(group, f"{name}_Xtr", Xtr)
        save(group, f"{name}_ytr", y)
        save(group, f"{name}_Xte", Xte)
        if ncls > 0:
            est = TabICLClassifier(n_estimators=n_est, norm_methods=methods, **common)
            est.fit(Xtr, y)
            proba = est.predict_proba(Xte)
            save(group, f"{name}_proba", proba.astype(np.float32))
            save(group, f"{name}_pred", est.predict(Xte).astype(np.float64))
        else:
            est = TabICLRegressor(n_estimators=n_est, norm_methods=methods, **common)
            est.fit(Xtr, y)
            save(group, f"{name}_mean", np.asarray(est.predict(Xte), dtype=np.float64))
            alphas = np.arange(0.1, 0.95, 0.1)
            q = est.predict(Xte, output_type="quantiles", alphas=list(alphas))
            save(group, f"{name}_quantiles", np.asarray(q, dtype=np.float64))
            save(group, f"{name}_alphas", alphas)
        meta[name] = dict(n_train=ntr, n_test=nte, d=d, n_classes=ncls,
                          n_estimators=n_est, norm_methods=methods)
    save_manifest(group, meta)


SECTIONS = {
    "rng_cpython": gen_rng_cpython,
    "rng_numpy": gen_rng_numpy,
    "shuffler": gen_shuffler,
    "ensemble_configs": gen_ensemble_configs,
    "preprocess_basic": gen_preprocess_basic,
    "normalizers": gen_normalizers,
    "model_small": gen_model_small,
    "e2e": gen_e2e,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"sections to generate (default: all): {sorted(SECTIONS)}")
    args = ap.parse_args()
    sections = args.only or sorted(SECTIONS)
    for s in sections:
        print(f"generating {s} ...")
        SECTIONS[s]()
    print("done:", FIXTURES)


if __name__ == "__main__":
    main()
