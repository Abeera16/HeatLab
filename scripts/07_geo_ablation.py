"""Does geospatial context from real rasters / GeoPandas vectors improve street-level temperature prediction?

Same protocol as the main study: pooled 5-fold spatial-block CV (KMeans station clusters per city, 0.5 km buffer),
same LightGBM hyper-parameters (the config chosen by inner-spatial tuning), same training-row thinning.
Only the *static feature set* changes, so differences are attributable to the surroundings data.

Feature sets
  no_static        ERA5 weather + time only (no knowledge of the location)
  elev_only        + station elevation relative to the city, vs ERA5 cell, distance to centroid
  osm_js           + OSM fractions from the browser (grid) pipeline  [= the published `no_wind_aware` model]
  osm_gpd          + OSM fractions recomputed with GeoPandas (UTM, exact overlay)           (if available)
  raster           + WorldCover 2021 + Copernicus DSM fractions/terrain (rasterio)
  osm_gpd+raster / osm_js+raster
Outputs results/geo_ablation.json
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, f"{ROOT}/src")
from heatlab import models as M
from heatlab.features import ERA5_DYNAMIC, OSM_STATIC, RASTER_FEATURES, STATIC_ISO, TIME_FEATS
from heatlab.splits import spatial_block_masks, spatial_folds

DATA = os.environ.get("HEATLAB_DATASET", f"{ROOT}/data/processed/dataset_raster.parquet")
GPD_CSV = f"{ROOT}/data/raw/osm_station_features_gpd.csv"
CFG = dict(params=dict(num_leaves=63, min_data_in_leaf=1000, learning_rate=0.05, lambda_l2=30.0), n=500)  # 'smooth'
CTX = ERA5_DYNAMIC + TIME_FEATS
ELEV = ["elev_rel", "elev_vs_era5", "dist_center_km"]


def main():
    d = pd.read_parquet(DATA)
    sets = {"no_static": CTX, "elev_only": CTX + ELEV, "osm_js": CTX + STATIC_ISO,
            "raster": CTX + ELEV + RASTER_FEATURES, "osm_js+raster": CTX + STATIC_ISO + RASTER_FEATURES}
    if os.path.exists(GPD_CSV):
        g = pd.read_csv(GPD_CSV)[["city", "station"] + OSM_STATIC].rename(columns={c: f"g_{c}" for c in OSM_STATIC})
        d = d.merge(g, on=["city", "station"], how="left")
        gcols = [f"g_{c}" for c in OSM_STATIC]
        assert d[gcols].notna().all().all()
        sets["osm_gpd"] = CTX + gcols + ELEV
        sets["osm_gpd+raster"] = CTX + gcols + ELEV + RASTER_FEATURES
    only = os.environ.get("SETS")
    if only:
        sets = {k: v for k, v in sets.items() if k in only.split(",")}
    folds = spatial_folds(d, 5)
    preds = {k: np.full(len(d), np.nan) for k in sets}
    fold_of = np.full(len(d), -1)
    t0 = time.time()
    for f in range(5):
        tr, te, _ = spatial_block_masks(d, f, folds_table=folds)
        trT = M.train_rows(d, tr)
        fold_of[te] = f
        for name, feats in sets.items():
            m = M.fit_lgbm(d, trT, feats, CFG["n"], CFG["params"])
            preds[name][te] = M.predict(m, d, te, feats)
        print(f"fold {f} done {time.time() - t0:.0f}s", flush=True)
    te = fold_of >= 0
    y = d.loc[te, "anom"].to_numpy()
    city = d.loc[te, "city"].to_numpy()
    out = {"protocol": "5-fold spatial-block CV, 0.5 km buffer, LightGBM 'smooth' config", "n_test_rows": int(te.sum()),
           "feature_counts": {k: len(v) for k, v in sets.items()}, "sets": {}}
    base = None
    for name in sets:
        e = preds[name][te] - y
        r = {"overall": M.metrics(e), "by_city": {c: M.metrics(e[city == c]) for c in np.unique(city)},
             "by_fold": [M.metrics(e[fold_of[te] == f]) for f in range(5)]}
        out["sets"][name] = r
    # paired fold-level comparison vs osm_js (positive = improvement in RMSE)
    ref = "osm_js" if "osm_js" in out["sets"] else next(iter(out["sets"]))
    for name in out["sets"]:
        diffs = [out["sets"][ref]["by_fold"][f]["rmse"] - out["sets"][name]["by_fold"][f]["rmse"] for f in range(5)]
        out["sets"][name]["delta_vs_" + ref] = {"mean": float(np.mean(diffs)), "folds_improved": int(sum(x > 0 for x in diffs))}
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    tag = os.environ.get("TAG", "")
    oof = d.loc[te, ["city", "station", "time", "anom"]].copy()
    oof["fold"] = fold_of[te]
    for name in sets:
        oof[f"lgbm_{name}"] = preds[name][te].astype(np.float32)
    oof.to_parquet(f"{ROOT}/results/geo_ablation_oof{tag}.parquet", index=False)
    json.dump(out, open(f"{ROOT}/results/geo_ablation{tag}.json", "w"), indent=1)
    for name, r in out["sets"].items():
        print(f"{name:16s} rmse={r['overall']['rmse']:.4f} mae={r['overall']['mae']:.4f}  " +
              " ".join(f"{c}={v['rmse']:.3f}" for c, v in r["by_city"].items()), r.get("delta_vs_" + ref))


if __name__ == "__main__":
    main()
