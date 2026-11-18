"""Neural network (Colab T4) vs LightGBM on identical spatial-CV rows. Writes results/dl_vs_lgbm.json"""
import json, os, numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
tag = os.environ.get("TAG", "_raster")
L = pd.read_parquet(f"{ROOT}/results/geo_ablation_oof{tag}.parquet")
N = pd.read_parquet(f"{ROOT}/results/mlp_oof.parquet").drop(columns=["anom", "fold"])
m = L.merge(N, on=["city", "station", "time"], how="inner")
assert len(m) == len(L) == len(N), (len(m), len(L), len(N))
y = m["anom"].to_numpy(np.float64)
rm = lambda e: float(np.sqrt(np.mean(e ** 2)))
out = {"n_rows": int(len(m)), "rows": {}}
for fs in ["osm_js", "osm_js+raster"]:
    lg, nn = m[f"lgbm_{fs}"].to_numpy(np.float64), m[f"mlp_{fs}"].to_numpy(np.float64)
    ens = (lg + nn) / 2
    r = {}
    for name, p in [("lightgbm", lg), ("mlp_gpu", nn), ("average", ens)]:
        e = p - y
        r[name] = {"rmse": rm(e), "mae": float(np.abs(e).mean()), "bias": float(e.mean()),
                   "by_city": {c: rm(e[(m.city == c).to_numpy()]) for c in sorted(m.city.unique())},
                   "by_fold": [rm(e[(m.fold == f).to_numpy()]) for f in range(5)]}
    lo, hi = m[f"mlp_{fs}_q10"].to_numpy(), m[f"mlp_{fs}_q90"].to_numpy()
    r["mlp_gpu"]["coverage80"] = float(((y >= lo) & (y <= hi)).mean())
    r["mlp_gpu"]["width80"] = float((hi - lo).mean())
    d = [r["lightgbm"]["by_fold"][f] - r["mlp_gpu"]["by_fold"][f] for f in range(5)]
    r["mlp_better_than_lgbm_folds"] = int(sum(x > 0 for x in d))  # d = lgbm - mlp, so x > 0 means the MLP RMSE is lower
    out["rows"][fs] = r
    print(fs, {k: round(v["rmse"], 4) for k, v in r.items() if isinstance(v, dict) and "rmse" in v},
          "MLP better in folds:", sum(x > 0 for x in d), "| mlp cov80 %.3f" % r["mlp_gpu"]["coverage80"])
# best LightGBM (GeoPandas OSM + rasters) averaged with the GPU network (fixed weights; no tuning on test folds)
G = f"{ROOT}/results/geo_ablation_oof_geo.parquet"
if os.path.exists(G):
    L2 = pd.read_parquet(G).merge(N, on=["city", "station", "time"], how="inner")
    assert len(L2) == len(N)
    yy = L2["anom"].to_numpy(np.float64); lg = L2["lgbm_osm_gpd+raster"].to_numpy(np.float64); nn = L2["mlp_osm_js+raster"].to_numpy(np.float64)
    bb = {"lgbm_best": rm(lg - yy), "mlp": rm(nn - yy)}
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):
        bb[f"blend_w{w}"] = rm(w * lg + (1 - w) * nn - yy)
    avg = (lg + nn) / 2
    bb["blend_5050_by_city"] = {c: rm((avg - yy)[(L2.city == c).to_numpy()]) for c in sorted(L2.city.unique())}
    bb["blend_5050_by_fold"] = [rm((avg - yy)[(L2.fold == f).to_numpy()]) for f in range(5)]
    bb["lgbm_by_fold"] = [rm((lg - yy)[(L2.fold == f).to_numpy()]) for f in range(5)]
    out["best_blend"] = bb
    print("best blend 50/50: %.4f vs best LightGBM %.4f" % (bb["blend_w0.5"], bb["lgbm_best"]))
json.dump(out, open(f"{ROOT}/results/dl_vs_lgbm.json", "w"), indent=1)
