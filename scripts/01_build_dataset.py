"""Build the modelling table from real data: FAIRUrbTemp stations + ERA5 + OSM surroundings."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from heatlab.features import build_dataset

root = os.path.join(os.path.dirname(__file__), "..")
d = build_dataset(f"{root}/data/raw/fairurbtemp", f"{root}/data/raw/era5", f"{root}/data/raw/osm_station_features.csv")
os.makedirs(f"{root}/data/processed", exist_ok=True)
d.to_parquet(f"{root}/data/processed/dataset.parquet", index=False)
import os.path as _p
if _p.exists(f"{root}/data/raw/raster_station_features.csv"):   # same table + WorldCover/DSM columns (run scripts/05 first)
    d2 = build_dataset(f"{root}/data/raw/fairurbtemp", f"{root}/data/raw/era5", f"{root}/data/raw/osm_station_features.csv",
                       raster_csv=f"{root}/data/raw/raster_station_features.csv")
    d2.to_parquet(f"{root}/data/processed/dataset_raster.parquet", index=False)
print(d.groupby("city").agg(stations=("station", "nunique"), rows=("t_obs", "size"), t0=("time", "min"), t1=("time", "max")))
print(d.shape, "NaN share per column (top):")
print(d.isna().mean().sort_values(ascending=False).head(6))
