"""Models, baselines and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from .features import ALL_FEATURES, ERA5_DYNAMIC, TIME_FEATS, STATIC_ISO, WIND_AWARE

LGB_DEFAULT = dict(objective="regression", learning_rate=0.06, num_leaves=63, min_data_in_leaf=300,
                   feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
                   max_bin=127, n_jobs=2, verbose=-1, seed=7)

FEATURE_SETS = {
    "full": ALL_FEATURES,
    "no_wind_aware": [f for f in ALL_FEATURES if f not in WIND_AWARE],
    "no_static": ERA5_DYNAMIC + TIME_FEATS,          # weather + time only: no knowledge of the location
}


def train_rows(d: pd.DataFrame, mask: np.ndarray, every: int = 2) -> np.ndarray:
    """Thin training rows to every `every`-th hour to bound compute (test rows are never thinned)."""
    keep = (d["time"].dt.hour % every == 0).to_numpy()
    return mask & keep


def fit_lgbm(d: pd.DataFrame, mask: np.ndarray, features: list[str], n_estimators=400, params=None, alpha=None):
    p = {**LGB_DEFAULT, **(params or {})}
    if alpha is not None:
        p.update(objective="quantile", alpha=alpha)
    ds = lgb.Dataset(d.loc[mask, features].to_numpy(np.float32), d.loc[mask, "anom"].to_numpy(np.float32),
                     feature_name=features, free_raw_data=True)
    return lgb.train(p, ds, num_boost_round=n_estimators)


def fit_ridge(d: pd.DataFrame, mask: np.ndarray, features: list[str]):
    X = d.loc[mask, features].fillna(0).to_numpy(np.float32)
    m = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    m.fit(X, d.loc[mask, "anom"].to_numpy(np.float32))
    return m


def predict(model, d: pd.DataFrame, mask: np.ndarray, features: list[str]) -> np.ndarray:
    X = d.loc[mask, features]
    X = X.to_numpy(np.float32) if isinstance(model, lgb.Booster) else X.fillna(0).to_numpy(np.float32)
    return model.predict(X)


# ----------------------------------------------------------------------------- baselines
def baseline_hour_mean(d: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """ERA5 + mean anomaly of the training rows at the same hour of day (no location information)."""
    hod = d["time"].dt.hour.to_numpy()
    tr = d.loc[train]
    tbl = tr.groupby(tr["time"].dt.hour)["anom"].mean()
    return pd.Series(hod[test]).map(tbl).fillna(tr["anom"].mean()).to_numpy()


def baseline_idw(d: pd.DataFrame, train: np.ndarray, test: np.ndarray, power: float = 2.0) -> np.ndarray:
    """Inverse-distance weighting of the *observed* anomalies of training stations at the same hour.

    This baseline needs live sensors at prediction time, which HeatLab's sensor-free model does not.
    """
    out = np.full(int(test.sum()), np.nan)
    te_idx = np.flatnonzero(test)
    fallback = baseline_hour_mean(d, train, test)
    for city in d.loc[test, "city"].unique():
        trc = d[train & (d["city"] == city).to_numpy()]
        tec_mask = test & (d["city"] == city).to_numpy()
        tec = d[tec_mask]
        if trc.empty or tec.empty:
            continue
        A = trc.pivot_table(index="time", columns="station", values="anom")
        st_tr = trc.groupby("station")[["lat", "lon"]].first().loc[A.columns]
        for sid, g in tec.groupby("station"):
            la, lo = g["lat"].iloc[0], g["lon"].iloc[0]
            kx = 111.32 * np.cos(np.deg2rad(la))
            dist = np.hypot((st_tr["lon"].to_numpy() - lo) * kx, (st_tr["lat"].to_numpy() - la) * 110.54)
            w = 1.0 / np.maximum(dist, 0.05) ** power
            sub = A.reindex(g["time"])
            vals = sub.to_numpy()
            msk = ~np.isnan(vals)
            num = np.where(msk, vals, 0.0) @ w
            den = msk @ w
            pred = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
            pos = np.searchsorted(te_idx, np.flatnonzero(tec_mask)[(tec["station"] == sid).to_numpy()])
            out[pos] = pred
    return np.where(np.isnan(out), fallback, out)


# ----------------------------------------------------------------------------- metrics
def metrics(err: np.ndarray) -> dict:
    err = err[np.isfinite(err)]
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)), "n": int(err.size)}


def by_group(d_test: pd.DataFrame, pred_anom: np.ndarray, key: np.ndarray) -> dict:
    err = pred_anom - d_test["anom"].to_numpy()
    out = {}
    for k in np.unique(key):
        m = key == k
        out[str(k)] = metrics(err[m])
    return out
