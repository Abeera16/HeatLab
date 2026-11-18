"""Model explanation: out-of-sample SHAP, sensitivity curves, and the observed station-level relationships."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import models as M
from .features import ALL_FEATURES

SENS_FEATURES = ["veg_250", "bld_250", "wat_250", "road_250", "elev_rel", "up_veg"]


def shap_summary(fold_models, d, folds_masks, n_per_fold=3000, seed=0):
    """Mean |SHAP| per feature, computed on test rows of the fold each model never saw."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(len(ALL_FEATURES))
    rows = 0
    signed = np.zeros(len(ALL_FEATURES))
    for model, te in zip(fold_models, folds_masks):
        idx = np.flatnonzero(te)
        pick = rng.choice(idx, size=min(n_per_fold, idx.size), replace=False)
        X = d.iloc[pick][ALL_FEATURES].to_numpy(np.float32)
        contrib = model.predict(X, pred_contrib=True)[:, :-1]
        acc += np.abs(contrib).sum(0)
        rows += len(pick)
    imp = acc / rows
    order = np.argsort(-imp)
    return [{"feature": ALL_FEATURES[i], "mean_abs_shap": float(imp[i])} for i in order]


def sensitivity_curves(model, d, n=6000, seed=0, grid_points=9):
    """Average predicted anomaly as one surroundings feature is moved across its observed 5-95% range.

    Other inputs keep their observed values. These are *associations learned from 95 stations*, not
    controlled interventions; the UI labels them accordingly.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for period, mask in (("night", d["is_night"] > 0.5), ("day", d["is_night"] <= 0.5)):
        idx = np.flatnonzero(mask.to_numpy())
        pick = rng.choice(idx, size=min(n, idx.size), replace=False)
        base = d.iloc[pick][ALL_FEATURES].copy()
        for f in SENS_FEATURES:
            lo, hi = np.nanpercentile(d[f], [5, 95])
            xs = np.linspace(lo, hi, grid_points)
            ys = []
            for x in xs:
                b = base.copy()
                b[f] = x
                if f in ("veg_250",):  # keep the dependent product features coherent
                    b["up_veg_x_ws"] = b["up_veg"] * b["e_ws"]
                ys.append(float(model.predict(b.to_numpy(np.float32)).mean()))
            out.setdefault(f, {})[period] = {"x": [float(v) for v in xs], "y": ys}
    return out


def observed_station_relations(d: pd.DataFrame):
    """Spearman correlation between a station's mean anomaly (vs its city-hour mean) and its surroundings."""
    d = d.copy()
    d["rel"] = d["t_obs"] - d.groupby(["city", "time"])["t_obs"].transform("mean")
    out = {}
    for period, m in (("night", d["is_night"] > 0.5), ("day", d["is_night"] <= 0.5)):
        g = d[m].groupby(["city", "station"]).agg(rel=("rel", "mean"), **{f: (f, "first") for f in ["veg_250", "bld_250", "wat_250", "road_250", "elev_rel"]}).reset_index()
        for f in ["veg_250", "bld_250", "wat_250", "road_250", "elev_rel"]:
            rows = []
            for city, gg in g.groupby("city"):
                if len(gg) >= 20 and gg[f].nunique() > 3:
                    r, p = spearmanr(gg[f], gg["rel"])
                    rows.append({"city": city, "rho": float(r), "p": float(p), "n": int(len(gg))})
            out.setdefault(f, {})[period] = rows
    return out
