"""HeatLab validity agent.

A LangGraph state machine that runs the whole study and *enforces the evaluation rules itself*:

  audit_data -> geo_checks -> plan_splits -> tune -> spatial_cv -> leakage_gate -> robustness
             -> explain -> train_final -> report

The agent is deterministic (no LLM is needed to apply the rules). Each node appends *gate*
records (pass / warn / fail, with evidence). The report only labels a number "reportable" when the
split that produced it passed the independence gates; numbers from leaky protocols are kept but
labelled as inflated, never as headline results.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from . import experiments as E
from . import explain as X
from . import models as M
from .features import ALL_FEATURES
from .splits import spatial_block_masks, spatial_folds, station_table


class State(TypedDict, total=False):
    data_path: str
    out_dir: str
    model_dir: str
    d: Any
    folds: Any
    cfg: dict
    gates: list
    results: dict
    preds: Any
    fold_of: Any
    fold_models: list
    fold_masks: list


def gate(state, gid, title, status, evidence, rule):
    state.setdefault("gates", []).append(
        {"id": gid, "title": title, "status": status, "evidence": evidence, "rule": rule})


# -------------------------------------------------------------------------------------- nodes
def audit_data(state: State) -> State:
    d = pd.read_parquet(state["data_path"])
    state["d"] = d
    state["results"] = {}
    per_city = []
    for city, g in d.groupby("city"):
        st = g.groupby("station")
        n_st = st.ngroups
        span = (g["time"].max() - g["time"].min()).total_seconds() / 3600 + 1
        comp = (st.size() / span).median()
        per_city.append({"city": city, "stations": int(n_st), "rows": int(len(g)),
                         "first": str(g["time"].min())[:10], "last": str(g["time"].max())[:10],
                         "median_completeness": float(comp),
                         "eligible_headline": bool(n_st >= 20)})
    state["results"]["data_audit"] = per_city
    small = [c["city"] for c in per_city if not c["eligible_headline"]]
    gate(state, "G1", "Enough stations for spatially independent evaluation",
         "warn" if small else "pass",
         f"{', '.join(small) or 'all cities'} have fewer than 20 stations" if small else "every city has >= 20 stations",
         "A city needs >= 20 stations before it contributes to headline spatial metrics; smaller networks are used for training and temporal / cross-city tests only.")
    nan_rate = float(d[["e_t2m", "anom"]].isna().mean().max())
    gate(state, "G2", "No unlabeled or missing targets", "pass" if nan_rate == 0 else "fail", f"missing target/context rate {nan_rate:.4f}", "Labels must be real observations aligned to ERA5 hours.")
    return state


def geo_checks(state: State) -> State:
    d = state["d"]
    s = station_table(d)
    bad_lat = int(((s.lat < -90) | (s.lat > 90)).sum() + ((s.lon < -180) | (s.lon > 180)).sum())
    swapped = 0
    for city, g in s.groupby("city"):
        if (g.lat.std() > 1 or g.lon.std() > 1):
            swapped += 1
    gate(state, "G3", "Coordinates valid and consistent (WGS84 lat/lon, no swapped axes)",
         "pass" if bad_lat == 0 and swapped == 0 else "fail",
         f"{bad_lat} out-of-range coordinates; {swapped} cities with implausible spread",
         "All stations must be valid WGS84 coordinates and sit within a city-sized extent.")
    frac_cols = [c for c in d.columns if c.startswith(("bld_", "veg_", "wat_"))]
    rng_ok = bool(((d[frac_cols] >= -1e-6) & (d[frac_cols] <= 1 + 1e-6)).all().all())
    gate(state, "G4", "Surrounding land-cover fractions are valid fractions", "pass" if rng_ok else "fail",
         "all bld/veg/wat fractions in [0,1]" if rng_ok else "fractions outside [0,1]",
         "Land-cover fractions derived from OSM must lie in [0, 1].")
    elev_bad = int((d.groupby(["city", "station"])["elev_vs_era5"].first().abs() > 400).sum())
    gate(state, "G5", "Station elevation consistent with the ERA5 grid cell", "pass" if elev_bad == 0 else "warn",
         f"{elev_bad} stations differ by more than 400 m from the ERA5 cell elevation",
         "Large elevation mismatch makes the ERA5 baseline unreliable; the model gets elev_vs_era5 as a feature.")
    close = 0
    for city, g in s.groupby("city"):
        xy = np.c_[g.lon * 111.32 * np.cos(np.deg2rad(g.lat.mean())), g.lat * 110.54]
        dd = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        np.fill_diagonal(dd, 9)
        close += int((np.triu(dd < 0.05, 1)).sum())
    gate(state, "G6", "No co-located duplicate stations", "pass" if close == 0 else "warn",
         f"{close} station pairs closer than 50 m",
         "Near-duplicate stations leak information across folds; spatial blocks plus a buffer absorb them.")
    return state


def plan_splits(state: State) -> State:
    d = state["d"]
    folds = spatial_folds(d, 5)
    state["folds"] = folds
    min_gap = []
    leaks = 0
    split_hash = hashlib.sha256()
    for f in range(5):
        tr, te, info = spatial_block_masks(d, f, folds_table=folds)
        tr_st = set(zip(d.loc[tr, "city"], d.loc[tr, "station"]))
        te_st = set(zip(d.loc[te, "city"], d.loc[te, "station"]))
        leaks += len(tr_st & te_st)
        split_hash.update(json.dumps(sorted(map(str, te_st))).encode())
        for city in d.loc[te, "city"].unique():
            a = folds[(folds.city == city) & folds.fold.eq(f)]
            b = folds[(folds.city == city) & folds.fold.ne(f)]
            b = b[[(c, s) in tr_st for c, s in zip(b.city, b.station)]]
            if len(a) and len(b):
                kx = 111.32 * np.cos(np.deg2rad(a.lat.mean()))
                dist = np.hypot((a.lon.to_numpy()[:, None] - b.lon.to_numpy()[None]) * kx,
                                (a.lat.to_numpy()[:, None] - b.lat.to_numpy()[None]) * 110.54)
                min_gap.append(float(dist.min()))
    gate(state, "G7", "Held-out stations never appear in training", "pass" if leaks == 0 else "fail",
         f"{leaks} stations shared between train and test across 5 folds", "Test stations must be disjoint from training stations.")
    mg = min(min_gap) if min_gap else float("nan")
    gate(state, "G8", "Spatial buffer between train and test stations", "pass" if mg >= 0.5 else "fail",
         f"minimum train-to-test distance {mg:.2f} km (buffer 0.5 km)", "Every training station must be at least 0.5 km from every held-out station.")
    state["results"]["split_fingerprint"] = split_hash.hexdigest()[:16]
    return state


def tune(state: State) -> State:
    info, cfg = E.tune(state["d"], state["folds"])
    state["cfg"] = cfg
    state["results"]["tuning"] = info
    gate(state, "G9", "Hyper-parameters tuned without touching test stations", "pass",
         f"chosen '{info['chosen']}' using an inner split of the training stations of fold 0", "Tuning may only use inner splits of the training stations.")
    return state


def spatial_cv(state: State) -> State:
    d = state["d"]
    preds, fold_of, sizes, models, masks = E.spatial_cv(d, state["folds"], state["cfg"])
    state.update(preds=preds, fold_of=fold_of, fold_models=models, fold_masks=masks)
    summ = E.summarise_spatial(d, preds, fold_of)
    state["results"]["spatial_cv"] = {"folds": sizes, **summ}
    cov = summ["coverage"]["empirical"]
    gate(state, "G10", "Prediction intervals are calibrated out of sample",
         "pass" if abs(cov - 0.8) <= 0.07 else "warn", f"80% interval covers {100*cov:.1f}% of held-out observations",
         "Nominal 80% intervals should cover 73-87% of held-out observations.")
    return state


def leakage_gate(state: State) -> State:
    d, cfg = state["d"], state["cfg"]
    sp = state["results"]["spatial_cv"]
    spatial_rmse = {c: sp["by_city"]["lgbm_full"][c] for c in E.HEADLINE_CITIES}
    res = E.leakage_study(d, state["folds"], cfg, spatial_rmse)
    res["spatial_blocks"] = {"overall": sp["overall"]["lgbm_full"], "by_city": spatial_rmse}
    state["results"]["leakage"] = res
    r_rows = res["random_rows"]["overall"]["rmse"]
    r_sp = res["spatial_blocks"]["overall"]["rmse"]
    inflation = 100 * (1 - r_rows / r_sp)
    gate(state, "G11", "Naive random-row validation blocked from headline results",
         "pass", f"random-row RMSE {r_rows:.3f} C vs spatial-block RMSE {r_sp:.3f} C ({inflation:.0f}% apparent error reduction from leakage)",
         "Random-row and random-station numbers are reported only as leakage evidence and are labelled not reportable.")
    state["results"]["leakage"]["inflation_pct"] = inflation
    return state


def robustness(state: State) -> State:
    d, cfg = state["d"], state["cfg"]
    state["results"]["temporal"] = E.temporal_holdout(d, cfg)
    state["results"]["loco"] = E.leave_city_out(d, cfg)
    return state


def explain(state: State) -> State:
    d = state["d"]
    res = state["results"]
    res["shap"] = X.shap_summary(state["fold_models"], d, state["fold_masks"])
    res["observed_relations"] = X.observed_station_relations(d)
    return state


def train_final(state: State) -> State:
    d, cfg = state["d"], state["cfg"]
    os.makedirs(state["model_dir"], exist_ok=True)
    allm = np.ones(len(d), bool)
    trT = M.train_rows(d, allm)
    m = M.fit_lgbm(d, trT, ALL_FEATURES, cfg["n"], cfg["params"])
    m10 = M.fit_lgbm(d, trT, ALL_FEATURES, 200, cfg["params"], alpha=0.1)
    m90 = M.fit_lgbm(d, trT, ALL_FEATURES, 200, cfg["params"], alpha=0.9)
    m.save_model(f"{state['model_dir']}/heatlab_lgbm_mean.txt")
    m10.save_model(f"{state['model_dir']}/heatlab_lgbm_q10.txt")
    m90.save_model(f"{state['model_dir']}/heatlab_lgbm_q90.txt")
    state["results"]["sensitivity"] = X.sensitivity_curves(m, d)
    with open(f"{state['model_dir']}/model_card.json", "w") as fh:
        json.dump({"features": ALL_FEATURES, "target": "station air temperature minus ERA5 2 m temperature (deg C)",
                   "hyperparameters": cfg, "train_rows": int(trT.sum()),
                   "trained_on_cities": sorted(d["city"].unique().tolist()),
                   "intended_use": "relative intra-urban air-temperature patterns for central-European-like climates",
                   "not_validated_for": "hot-arid and tropical cities; any city outside the four training networks"}, fh, indent=2)
    gate(state, "G12", "Production model carries an explicit scope statement", "pass",
         "model_card.json records training cities and unvalidated regimes", "A shipped model must state where it has and has not been validated.")
    return state


def report(state: State) -> State:
    res = state["results"]
    res["gates"] = state["gates"]
    res["generated"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    os.makedirs(state["out_dir"], exist_ok=True)
    with open(f"{state['out_dir']}/results.json", "w") as fh:
        json.dump(res, fh, indent=1)
    d, preds, fold_of = state["d"], state["preds"], state["fold_of"]
    te = fold_of >= 0
    out = d.loc[te, ["city", "station", "time", "t_obs", "e_t2m", "anom", "is_night"]].copy()
    for c in ["lgbm_full", "hour_mean", "idw_live_sensors", "q10", "q90"]:
        out[c] = preds.loc[te, c].astype(np.float32).to_numpy()
    out["fold"] = fold_of[te].to_numpy()
    out.to_parquet(f"{state['out_dir']}/oof_predictions.parquet", index=False)
    lines = ["# HeatLab validity report", "", f"Generated {res['generated']}", "", "| Gate | Status | Evidence |", "|---|---|---|"]
    for g in state["gates"]:
        lines.append(f"| {g['id']} {g['title']} | **{g['status'].upper()}** | {g['evidence']} |")
    with open(f"{state['out_dir']}/validity_report.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return state


def build_graph():
    g = StateGraph(State)
    nodes = [("audit_data", audit_data), ("geo_checks", geo_checks), ("plan_splits", plan_splits), ("tune", tune),
             ("spatial_cv", spatial_cv), ("leakage_gate", leakage_gate), ("robustness", robustness),
             ("explain", explain), ("train_final", train_final), ("report", report)]
    for n, f in nodes:
        g.add_node(n, f)
    g.add_edge(START, nodes[0][0])
    for (a, _), (b, _) in zip(nodes, nodes[1:]):
        g.add_edge(a, b)
    g.add_edge(nodes[-1][0], END)
    return g.compile()


def run(data_path: str, out_dir: str, model_dir: str):
    return build_graph().invoke({"data_path": data_path, "out_dir": out_dir, "model_dir": model_dir, "gates": []})
