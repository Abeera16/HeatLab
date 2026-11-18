"""Experiment suite. Every function takes the dataset DataFrame and returns JSON-serialisable results."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import models as M
from .features import ALL_FEATURES, WIND_AWARE
from .splits import (leave_city_out_masks, random_row_masks, random_station_masks, spatial_block_masks,
                     spatial_folds, temporal_masks)

HEADLINE_CITIES = ["NoviSad", "Rennes", "Biel"]  # >=20 stations: eligible for spatial headline metrics
CITY_NAMES = {"NoviSad": "Novi Sad", "Rennes": "Rennes", "Biel": "Biel/Bienne", "Ghent": "Ghent"}

GRID = [  # hyper-parameter candidates, chosen on inner spatial blocks of the *training* stations only
    dict(name="small",   params=dict(num_leaves=31, min_data_in_leaf=300, learning_rate=0.06), n=300),
    dict(name="medium",  params=dict(num_leaves=63, min_data_in_leaf=300, learning_rate=0.06), n=400),
    dict(name="large",   params=dict(num_leaves=127, min_data_in_leaf=500, learning_rate=0.05), n=400),
    dict(name="smooth",  params=dict(num_leaves=63, min_data_in_leaf=1000, learning_rate=0.05, lambda_l2=30.0), n=500),
]


def _log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.0f}s] {msg}", flush=True)


def err_of(d, mask, pred):
    return pred - d.loc[mask, "anom"].to_numpy()


def tune(d: pd.DataFrame, folds_table: pd.DataFrame, grid=GRID):
    """Pick hyper-parameters with an inner spatial split of the training stations of outer fold 0."""
    tr, _te, _ = spatial_block_masks(d, 0, folds_table=folds_table)
    sub = d[tr]
    inner = spatial_folds(sub, k=3, seed=1)
    tr2, te2, _ = spatial_block_masks(sub.reset_index(drop=True), 1, folds_table=inner)
    sub = sub.reset_index(drop=True)
    res = []
    for g in grid:
        t = time.time()
        m = M.fit_lgbm(sub, M.train_rows(sub, tr2), ALL_FEATURES, g["n"], g["params"])
        p = M.predict(m, sub, te2, ALL_FEATURES)
        e = err_of(sub, te2, p)
        res.append({"name": g["name"], "rmse": M.metrics(e)["rmse"], "mae": M.metrics(e)["mae"], "secs": round(time.time() - t, 1)})
        _log(f"tune {g['name']}: rmse={res[-1]['rmse']:.4f}")
    best = min(res, key=lambda r: r["rmse"])["name"]
    chosen = next(g for g in grid if g["name"] == best)
    return {"candidates": res, "chosen": best}, chosen


def spatial_cv(d, folds_table, cfg, k=5):
    """Pooled spatial-block cross-validation: fold f holds out cluster f of every city at once."""
    d = d.copy()
    keys = {"full": ALL_FEATURES, "no_wind_aware": M.FEATURE_SETS["no_wind_aware"], "no_static": M.FEATURE_SETS["no_static"]}
    cols = ["era5_raw", "hour_mean", "idw_live_sensors", "ridge", "lgbm_full", "lgbm_no_wind_aware", "lgbm_no_static",
            "q10", "q90"]
    preds = pd.DataFrame(index=d.index, columns=cols, dtype=np.float64)
    fold_of = pd.Series(-1, index=d.index)
    sizes, fold_models, fold_masks = [], [], []
    for f in range(k):
        tr, te, info = spatial_block_masks(d, f, folds_table=folds_table)
        trT = M.train_rows(d, tr)
        sizes.append({"fold": f, **info, "train_rows": int(trT.sum()), "test_rows": int(te.sum())})
        fold_of[te] = f
        preds.loc[te, "era5_raw"] = 0.0
        preds.loc[te, "hour_mean"] = M.baseline_hour_mean(d, tr, te)
        preds.loc[te, "idw_live_sensors"] = M.baseline_idw(d, tr, te)
        r = M.fit_ridge(d, trT, ALL_FEATURES)
        preds.loc[te, "ridge"] = M.predict(r, d, te, ALL_FEATURES)
        for name, feats in keys.items():
            m = M.fit_lgbm(d, trT, feats, cfg["n"], cfg["params"])
            preds.loc[te, f"lgbm_{name}"] = M.predict(m, d, te, feats)
            if name == "full":
                fold_models.append(m)
                fold_masks.append(te)
                m10 = M.fit_lgbm(d, trT, feats, 200, cfg["params"], alpha=0.1)
                m90 = M.fit_lgbm(d, trT, feats, 200, cfg["params"], alpha=0.9)
                preds.loc[te, "q10"] = M.predict(m10, d, te, feats)
                preds.loc[te, "q90"] = M.predict(m90, d, te, feats)
        _log(f"spatial fold {f} done ({info})")
    return preds, fold_of, sizes, fold_models, fold_masks


def summarise_spatial(d, preds, fold_of):
    te = fold_of >= 0
    dt = d[te]
    y = dt["anom"].to_numpy()
    out = {"overall": {}, "by_city": {}, "by_hour": {}, "by_daynight": {}, "by_fold": {}, "coverage": {}}
    model_cols = ["era5_raw", "hour_mean", "idw_live_sensors", "ridge", "lgbm_full", "lgbm_no_wind_aware", "lgbm_no_static"]
    hour = dt["time"].dt.hour.to_numpy()
    night = (dt["is_night"].to_numpy() > 0.5)
    for c in model_cols:
        p = preds.loc[te, c].to_numpy(np.float64)
        e = p - y
        out["overall"][c] = M.metrics(e)
        out["by_city"][c] = M.by_group(dt, p, dt["city"].to_numpy())
        out["by_hour"][c] = M.by_group(dt, p, hour)
        out["by_daynight"][c] = M.by_group(dt, p, np.where(night, "night", "day"))
        out["by_fold"][c] = M.by_group(dt, p, fold_of[te].to_numpy())
    q10, q90 = preds.loc[te, "q10"].to_numpy(np.float64), preds.loc[te, "q90"].to_numpy(np.float64)
    inside = (y >= q10) & (y <= q90)
    out["coverage"] = {"nominal": 0.8, "empirical": float(inside.mean()), "mean_width": float(np.mean(q90 - q10)),
                       "by_city": {c: float(inside[(dt["city"] == c).to_numpy()].mean()) for c in dt["city"].unique()}}
    return out


def leakage_study(d, folds_table, cfg, spatial_rmse_by_city, k=5, n_folds=3):
    """Same model, same data, three validation protocols: how much does a naive split flatter the model?"""
    res = {"random_rows": {}, "random_stations": {}, "spatial_blocks": spatial_rmse_by_city}
    for name, fn in (("random_rows", random_row_masks), ("random_stations", random_station_masks)):
        per_city = {c: [] for c in HEADLINE_CITIES}
        allerr = []
        for f in range(n_folds):
            tr, te = fn(d, f, k)
            m = M.fit_lgbm(d, M.train_rows(d, tr), ALL_FEATURES, cfg["n"], cfg["params"])
            p = M.predict(m, d, te, ALL_FEATURES)
            e = err_of(d, te, p)
            allerr.append(e)
            cities = d.loc[te, "city"].to_numpy()
            for c in HEADLINE_CITIES:
                per_city[c].append(e[cities == c])
            _log(f"leakage {name} fold {f}: rmse={M.metrics(e)['rmse']:.4f}")
        res[name] = {"overall": M.metrics(np.concatenate(allerr)),
                     "by_city": {c: M.metrics(np.concatenate(v)) for c, v in per_city.items()}}
    return res


def temporal_holdout(d, cfg):
    tr, te, info = temporal_masks(d)
    trT = M.train_rows(d, tr)
    out = {"split": info, "models": {}, "by_city": {}}
    dt = d[te]
    y = dt["anom"].to_numpy()
    preds = {"era5_raw": np.zeros(int(te.sum())), "hour_mean": M.baseline_hour_mean(d, tr, te)}
    for name, feats in (("lgbm_full", ALL_FEATURES), ("lgbm_no_static", M.FEATURE_SETS["no_static"])):
        m = M.fit_lgbm(d, trT, feats, cfg["n"], cfg["params"])
        preds[name] = M.predict(m, d, te, feats)
    for k_, p in preds.items():
        out["models"][k_] = M.metrics(p - y)
        out["by_city"][k_] = M.by_group(dt, p, dt["city"].to_numpy())
    _log("temporal holdout done")
    return out


def leave_city_out(d, cfg):
    out = {}
    for city in d["city"].unique():
        tr, te = leave_city_out_masks(d, city)
        trT = M.train_rows(d, tr)
        dt = d[te]
        y = dt["anom"].to_numpy()
        res = {"era5_raw": M.metrics(0 - y), "hour_mean": M.metrics(M.baseline_hour_mean(d, tr, te) - y)}
        for name, feats in (("lgbm_full", ALL_FEATURES), ("lgbm_no_static", M.FEATURE_SETS["no_static"])):
            m = M.fit_lgbm(d, trT, feats, cfg["n"], cfg["params"])
            res[name] = M.metrics(M.predict(m, d, te, feats) - y)
        # city-mean oracle: what a bias correction using the target city's own sensors would achieve
        res["oracle_city_hour_mean"] = M.metrics(M.baseline_hour_mean(d, te, te) - y)
        out[city] = res
        _log(f"LOCO {city}: lgbm={res['lgbm_full']['rmse']:.3f} raw={res['era5_raw']['rmse']:.3f}")
    return out
