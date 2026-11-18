"""Independent check of the two OSM feature methods (browser grid vs GeoPandas exact overlay) against ESA WorldCover.
Writes results/vector_referee.json."""
import json, os, numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
o = pd.read_csv(f"{ROOT}/data/raw/osm_station_features.csv"); n = pd.read_csv(f"{ROOT}/data/raw/osm_station_features_gpd.csv")
r = pd.read_csv(f"{ROOT}/data/raw/raster_station_features.csv")
m = o.merge(n, on=["city", "station"], suffixes=("_grid", "_gpd")).merge(r, on=["city", "station"])
def corr(a, b):
    return float(a.corr(b)) if a.std() > 0 and b.std() > 0 else None
out = {"water_250": {}, "vegetation_250": {}, "buildings_250": {}}
green = m.wc_tree_250 + m.wc_lowveg_250 + m.wc_crop_250
for c, g in m.groupby("city"):
    gg = green[g.index]
    out["water_250"][c] = {"grid": corr(g.wat_250_grid, g.wc_water_250), "geopandas": corr(g.wat_250_gpd, g.wc_water_250),
                           "mean_worldcover": float(g.wc_water_250.mean()), "mean_grid": float(g.wat_250_grid.mean()), "mean_geopandas": float(g.wat_250_gpd.mean())}
    out["vegetation_250"][c] = {"grid": corr(g.veg_250_grid, gg), "geopandas": corr(g.veg_250_gpd, gg)}
    out["buildings_250"][c] = {"grid": corr(g.bld_250_grid, g.wc_built_250), "geopandas": corr(g.bld_250_gpd, g.wc_built_250)}
json.dump(out, open(f"{ROOT}/results/vector_referee.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:1500])
