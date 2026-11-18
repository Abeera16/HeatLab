"""Recompute the OSM surroundings features with GeoPandas (UTM projection, exact polygon overlay).

Input : data/raw/osm_geometries/<city>.json   raw OSM ways/relations fetched with tools/overpass_geometries.js
        data/raw/osm_station_features.csv     station list + the earlier browser-grid features (for comparison)
Output: data/raw/osm_station_features_gpd.csv same columns as the original table, new values
        results/vector_validation.json        agreement between the grid and the exact-overlay method
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, f"{ROOT}/src")
from heatlab.geo import city_features
from heatlab.features import OSM_STATIC


def main():
    old = pd.read_csv(f"{ROOT}/data/raw/osm_station_features.csv")
    parts = []
    for city, st in old.groupby("city"):
        raw = json.load(open(f"{ROOT}/data/raw/osm_geometries/{city}.json"))
        f = city_features(raw, st[["station", "lat", "lon"]].reset_index(drop=True))
        f.insert(0, "city", city)
        parts.append(f)
        print(city, len(raw), "OSM objects ->", len(f), "stations, UTM EPSG", f.utm_epsg.iloc[0], flush=True)
    new = pd.concat(parts, ignore_index=True)
    out = old[["city", "station", "lat", "lon", "elev"]].merge(new, on=["city", "station"])
    out.to_csv(f"{ROOT}/data/raw/osm_station_features_gpd.csv", index=False)

    m = old.merge(new, on=["city", "station"], suffixes=("_grid", "_gpd"))
    rep = {}
    for c in OSM_STATIC + [f"s{k}_{i}" for k in ("bld", "veg", "wat") for i in range(8)]:
        a, b = m[f"{c}_grid"], m[f"{c}_gpd"]
        rep[c] = {"pearson": float(a.corr(b)), "spearman": float(a.corr(b, method="spearman")),
                  "mean_grid": float(a.mean()), "mean_gpd": float(b.mean()), "mae": float((a - b).abs().mean())}
    json.dump(rep, open(f"{ROOT}/results/vector_validation.json", "w"), indent=1)
    show = [c for c in OSM_STATIC if c.endswith("_250")]
    print(pd.DataFrame({k: rep[k] for k in show}).T.round(3).to_string())
    print("min Pearson over 12 isotropic features: %.3f" % min(rep[c]["pearson"] for c in OSM_STATIC))


if __name__ == "__main__":
    main()
