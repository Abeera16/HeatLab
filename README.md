# HeatLab

Downscaling urban 2 m air temperature from ERA5 to street-level stations, with a validity-guarded
agent that refuses to report numbers that fail its checks. Every number comes from real data. No synthetic data is used.

## Data (all real)
| Source | What | Used for |
|---|---|---|
| FAIRUrbTemp (SEF files, QC-flagged values excluded) | 95 stations: Biel 40, Novi Sad 26, Rennes 23, Ghent 6 | Target: station minus ERA5 temperature |
| ERA5 hourly via Open-Meteo archive | 2 m temp, humidity, wind, radiation, cloud, pressure, precipitation | Predictors |
| OpenStreetMap via Overpass | Building, vegetation, water, road fractions at 100/250/500 m, 8 compass sectors | Predictors |
| Open-Meteo elevation API | Station elevation | Predictors |
| ESA WorldCover 2021 v200 (10 m, COG, EPSG:4326) | Built, tree, low-vegetation, crop, water shares in 100/250/500 m buffers | Predictors (rasterio, buffers built in UTM) |
| Copernicus GLO-30 DSM (30 m, EPSG:4326) | Station height, relief, height relative to local mean | Predictors |

## Results (spatial-block CV, 5 folds, 0.5 km buffer, 1.34M station-hours)
| Method | RMSE °C |
|---|---|
| Raw ERA5 | 1.59 |
| ERA5 + hour-of-day offset | 1.42 |
| Ridge | 1.39 |
| LightGBM without surroundings | 1.18 |
| **HeatLab (LightGBM, GeoPandas OSM + WorldCover/DSM rasters)** | **1.09** |
| Neighbour sensors, IDW (needs live sensors) | 0.91 |

- Random-row validation reports 0.92 °C for the same model, a 15% flattering gap caused by leakage.
- Wind-aware upwind features made no measurable difference (1.0864 without vs 1.0876 °C with).
- Leave-one-city-out: beats raw ERA5 in all four cities (barely in Novi Sad) but beats a plain hour offset in Biel, Ghent and Rennes (not Novi Sad).
- The 80% interval covers 71%: slightly too narrow (gate G10 warns).

## Geospatial stack and GPU model (second iteration)
The headline model above was retrained on these features (v2; `HEATLAB_FEATURES=v1` reproduces the first iteration, archived in `archive_v1/`). The first-iteration headline was 1.113 °C.
Same protocol as above (5-fold spatial-block CV, LightGBM, identical rows). Only the inputs or the model change.

| Inputs / model | RMSE °C |
|---|---|
| Weather and time only | 1.188 |
| + OSM fractions, original browser grid method | 1.114 |
| + OSM fractions recomputed with GeoPandas (UTM, exact overlay) | 1.102 |
| WorldCover + Copernicus DSM rasters only, no OSM | 1.122 |
| GeoPandas OSM + rasters | **1.098** |
| Neural network on a Colab T4 GPU (OSM grid + rasters) | 1.129 |
| Average of GeoPandas LightGBM and the neural network (equal weights) | **1.046** |

- **GeoPandas rework.** OSM geometries are fetched raw, multipolygon relations are assembled with `polygonize`, everything is projected to the local UTM zone (EPSG 32630/32631/32632/32634) and areas are overlaid exactly. It agrees with the first method at r = 0.96-1.00 in Rennes and Ghent, but fixes Biel: the old method missed the lake (water share correlation with WorldCover 0.26, now 0.98). `scripts/10_vector_referee.py`.
- **Rasters.** WorldCover and the Copernicus DSM are cropped per city and sampled with rasterio; buffers are built in UTM and transformed to the raster CRS. `scripts/05_raster_features.py`.
- **xarray / Zarr.** The four ERA5 CSVs are converted into one `(city, time)` xarray Dataset stored as Zarr (`data/processed/era5.zarr`, 82k hours, 11 variables). `scripts/04_era5_to_zarr.py`. This is a conversion from CSV, not a native NetCDF/Zarr source.
- **Neural network.** `notebooks/heatlab_mlp_gpu.ipynb` trains a residual MLP (mean + 10/90 % quantile heads, mixed precision) on a Colab Tesla T4, one fold at a time, with scaling fitted on training rows only. It ties LightGBM on its own (better in 2 of 5 folds); its 80 % band covers only about 50 %, so it is not used for uncertainty. A `DataParallel` branch exists for several GPUs but was only run on one. Weights: `models/dl/`.
- The headline numbers in the dashboard Overview come from the original pipeline; the variants above are on the dashboard page "Geospatial & GPU".

Not done: forecasting by lead time (no archived forecasts overlap the 2014-2023 station records with lead times), physical thermal simulation, shade and reflective-surface modelling.

Extra scripts: `04_era5_to_zarr.py`, `05_raster_features.py`, `06_vector_features.py` (needs `data/raw/osm_geometries/<city>.json` from `tools/overpass_geometries.js`), `07_geo_ablation.py`, `08_make_dl_bundle.py`, `09_compare_dl.py`, `10_vector_referee.py`.

## Layout
- `src/heatlab/` pipeline: `io`, `features`, `splits`, `models`, `experiments`, `explain`, `agent` (LangGraph, gates G1-G12)
- `scripts/01_build_dataset.py` builds `data/processed/dataset.parquet`; `02_run_agent.py` trains and evaluates; `03_export_ui.py` writes `app/data.js` (filterable aggregates computed from the out-of-fold predictions)
- `tools/overpass_features.js` how the OSM features were extracted (runs in a browser)
- `data/raw/osm_station_features.csv` the extracted station surroundings (95 rows)
- `models/` trained LightGBM boosters (mean, q10, q90) and `model_card.json`
- `results/` `results.json`, `validity_report.md`, `oof_predictions.parquet`
- `app/index.html` + `app/data.js` the analytics dashboard (views: Overview, Time & conditions, Stations, Cities & transfer, Drivers, Geospatial & GPU, Validity & scope). Global filters for city, day/night and hour range drive every KPI and chart. Open `index.html` directly; no server needed.

## Reproduce
```
pip install -r requirements.txt
python scripts/01_build_dataset.py
python scripts/02_run_agent.py     # about 17 min on CPU
python scripts/03_export_ui.py
```
Raw FAIRUrbTemp zips (BORIS portal) and the ERA5 CSVs go in `data/raw/`; see `src/heatlab/io.py` for the expected layout.

## Limits
Only four European networks; hot-arid and tropical transfer is untested. Ghent has 6 stations and is not a headline spatial result.
OSM is a present-day snapshot with uneven tagging. Time spans differ per city. Curves are associations, not causal effects.
