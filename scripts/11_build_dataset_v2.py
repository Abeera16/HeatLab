"""Modelling table v2: GeoPandas OSM surroundings (UTM, exact overlay) + WorldCover/DSM rasters + the same ERA5 context."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from heatlab.features import build_dataset

root = os.path.join(os.path.dirname(__file__), "..")
d = build_dataset(f"{root}/data/raw/fairurbtemp", f"{root}/data/raw/era5", f"{root}/data/raw/osm_station_features_gpd.csv",
                  raster_csv=f"{root}/data/raw/raster_station_features.csv")
d.to_parquet(f"{root}/data/processed/dataset_v2.parquet", index=False)
print(d.shape, int(d.isna().sum().sum()), "NaNs")
