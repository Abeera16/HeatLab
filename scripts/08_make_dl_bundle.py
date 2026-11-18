"""Pack the modelling data for the GPU notebook as three small normalised tables (joined back inside the notebook):
  era5_ctx.parquet  (city, time) -> weather + time features            ~82k rows
  stations.parquet  station -> static surroundings + spatial-CV fold/train flags
  obs.parquet       (city, station, time) -> anomaly target (int16, 0.01 degC)  ~1.3M rows
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, f"{ROOT}/src")
from heatlab.features import ERA5_DYNAMIC, TIME_FEATS, STATIC_ISO, RASTER_FEATURES
from heatlab.splits import spatial_block_masks, spatial_folds

d = pd.read_parquet(f"{ROOT}/data/processed/dataset_raster.parquet")
folds = spatial_folds(d, 5)
out = f"{ROOT}/data/processed/dl_bundle"; os.makedirs(out, exist_ok=True)
ctx = d[["city", "time"] + ERA5_DYNAMIC + TIME_FEATS].drop_duplicates(["city", "time"])
# context columns must be a pure function of (city, time)
assert d.groupby(["city", "time"])[ERA5_DYNAMIC + TIME_FEATS].nunique().max().max() == 1
ctx.to_parquet(f"{out}/era5_ctx.parquet", index=False, compression="zstd", compression_level=9)
st = d.groupby(["city", "station"], as_index=False)[STATIC_ISO + RASTER_FEATURES].first()
assert d.groupby(["city", "station"])[STATIC_ISO + RASTER_FEATURES].nunique().max().max() == 1
st["fold"] = -1
for f in range(5):
    tr, te, info = spatial_block_masks(d, f, folds_table=folds)
    ids_tr = set(zip(d.city[tr], d.station[tr])); ids_te = set(zip(d.city[te], d.station[te]))
    st[f"tr{f}"] = [int((c, s) in ids_tr) for c, s in zip(st.city, st.station)]
    st.loc[[(c, s) in ids_te for c, s in zip(st.city, st.station)], "fold"] = f
st.to_parquet(f"{out}/stations.parquet", index=False, compression="zstd")
q = np.round(d["anom"].to_numpy(np.float64) * 100)
print("max quantisation error degC:", float(np.abs(q / 100 - d["anom"].to_numpy(np.float64)).max()))
ob = pd.DataFrame({"city": d.city.astype(str), "station": d.station.astype(str), "time": d.time, "anom_c": q.astype(np.int16)})
ob.to_parquet(f"{out}/obs.parquet", index=False, compression="zstd", compression_level=9)
for fn in os.listdir(out): print(fn, round(os.path.getsize(f"{out}/{fn}") / 1e6, 2), "MB")
