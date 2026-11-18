"""Dataset construction: station labels + ERA5 context + static surroundings + wind-aware footprint."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .io import CITIES, load_era5, load_static, load_stations

SECTOR_BEARING = np.arange(8) * 45.0  # sector k is centred on bearing 45*k degrees (0 = north)

# --- feature groups (used by ablations and by the validity agent) ---------------------------------
ERA5_DYNAMIC = ["e_t2m", "e_rh", "e_dew", "e_p", "e_cloud", "e_ws", "e_u", "e_v", "e_sw", "e_dir", "e_dif",
                "e_pr", "e_dt3", "e_t24", "e_sw3", "e_dtd"]
TIME_FEATS = ["hod_sin", "hod_cos", "doy_sin", "doy_cos", "is_night"]
STATIC_ISO = [f"{k}_{b}" for b in (100, 250, 500) for k in ("bld", "veg", "wat", "road")] + \
             ["elev_rel", "elev_vs_era5", "dist_center_km"]
WIND_AWARE = ["up_bld", "up_veg", "up_wat", "down_veg", "up_veg_x_ws", "up_wat_x_ws"]
ALL_FEATURES_V1 = ERA5_DYNAMIC + TIME_FEATS + STATIC_ISO + WIND_AWARE
ALL_FEATURES = ALL_FEATURES_V1   # reassigned below once RASTER_FEATURES is defined (HEATLAB_FEATURES=v2 adds them)

# --- raster-derived surroundings (ESA WorldCover 2021 10 m + Copernicus GLO-30 DSM), sampled with rasterio ---
RASTER_FEATURES = [f"wc_{k}_{b}" for b in (100, 250, 500) for k in ("built", "tree", "lowveg", "crop", "water")] + \
                  ["dsm_std_500", "dsm_range_500", "dsm_tpi_250", "dsm_tpi_500"]
if os.environ.get("HEATLAB_FEATURES", "v2") == "v2":   # v2 (default) = v1 + WorldCover/DSM rasters; HEATLAB_FEATURES=v1 reproduces the first iteration
    ALL_FEATURES = ALL_FEATURES_V1 + RASTER_FEATURES
# static columns that come from OSM vectors (used when swapping the JS-derived table for the GeoPandas one)
OSM_STATIC = [f"{k}_{b}" for b in (100, 250, 500) for k in ("bld", "veg", "wat", "road")]


def _era5_derived(e: pd.DataFrame) -> pd.DataFrame:
    e = e.copy()
    wd = np.deg2rad(e["e_wd"])
    # meteorological convention: direction the wind blows FROM; u/v are the blowing-towards components
    e["e_u"] = -e["e_ws"] * np.sin(wd)
    e["e_v"] = -e["e_ws"] * np.cos(wd)
    e["e_dt3"] = e["e_t2m"] - e["e_t2m"].shift(3)
    e["e_t24"] = e["e_t2m"].rolling(24, min_periods=12).mean()
    e["e_sw3"] = e["e_sw"].rolling(3, min_periods=1).sum()
    e["e_dtd"] = e["e_t2m"] - e["e_t24"]
    return e


def _sector_weights(wd_deg: np.ndarray, ws: np.ndarray, power: float = 2.0) -> np.ndarray:
    """Weights over the 8 compass sectors for air arriving from `wd_deg` (the upwind footprint).

    w_k ~ max(0, cos(bearing_k - wd))^power, normalised to sum to one. Calm air (<0.7 m/s) falls back
    to an isotropic footprint because no preferred advection direction exists.
    """
    d = np.deg2rad(wd_deg[:, None] - SECTOR_BEARING[None, :])
    w = np.clip(np.cos(d), 0, None) ** power
    s = w.sum(axis=1, keepdims=True)
    w = np.where(s > 0, w / np.where(s > 0, s, 1), 1 / 8)
    calm = (ws < 0.7)[:, None]
    return np.where(calm, 1 / 8, w)


def build_dataset(data_dir: str, era5_dir: str, static_json: str, cities=None, raster_csv: str | None = None) -> pd.DataFrame:
    import glob

    obs, stations = load_stations(data_dir, cities)
    static = load_static(static_json)
    meta = pd.DataFrame([s.__dict__ for s in stations]).rename(columns={"lat": "lat0", "lon": "lon0"})
    static = static.merge(meta[["city", "sid"]].rename(columns={"sid": "station"}), on=["city", "station"])
    if raster_csv:
        rs = pd.read_csv(raster_csv)[["city", "station"] + RASTER_FEATURES]
        static = static.merge(rs, on=["city", "station"], how="left")
    parts = []
    for city, (_n, pat, name, country) in CITIES.items():
        if cities and city not in cities:
            continue
        o = obs[obs.city == city]
        st = static[static.city == city].copy()
        if o.empty or st.empty:
            continue
        e, cell = load_era5(glob.glob(f"{era5_dir}/{pat}")[0])
        e = _era5_derived(e)
        st["elev_rel"] = st["elev"] - st["elev"].mean()
        st["elev_vs_era5"] = st["elev"] - cell["elev"]
        kx = 111.32 * np.cos(np.deg2rad(st["lat"].mean()))
        st["dist_center_km"] = np.hypot((st["lon"] - st["lon"].mean()) * kx, (st["lat"] - st["lat"].mean()) * 110.54)
        d = o.merge(st, on=["city", "station"], how="inner").merge(e, left_on="time", right_index=True, how="inner")
        d = d.dropna(subset=["e_t2m"]).copy()
        # wind-aware footprint: sector fractions weighted by where the air comes from
        w = _sector_weights(d["e_wd"].to_numpy(), d["e_ws"].to_numpy())
        wd_ = np.roll(w, 4, axis=1)  # opposite side = downwind footprint
        for key, short in (("sveg", "veg"), ("sbld", "bld"), ("swat", "wat")):
            S = d[[f"{key}_{k}" for k in range(8)]].to_numpy()
            d[f"up_{short}"] = (w * S).sum(axis=1)
            if short == "veg":
                d["down_veg"] = (wd_ * S).sum(axis=1)
        d["up_veg_x_ws"] = d["up_veg"] * d["e_ws"]
        d["up_wat_x_ws"] = d["up_wat"] * d["e_ws"]
        d["city_name"], d["country"] = name, country
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    h = d["time"].dt.hour + d["time"].dt.minute / 60
    doy = d["time"].dt.dayofyear
    d["hod_sin"], d["hod_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    d["is_night"] = (d["e_sw"] < 5).astype(np.float32)
    d["anom"] = d["t_obs"] - d["e_t2m"]  # modelling target: station minus ERA5 grid-cell temperature
    keep = ["city", "city_name", "country", "station", "time", "lat", "lon", "t_obs", "anom"] + ALL_FEATURES
    if raster_csv:
        keep += [c for c in RASTER_FEATURES if c not in keep]
    d = d[keep].copy()
    f32 = [c for c in d.columns if d[c].dtype == "float64" and c not in ("lat", "lon")]
    d[f32] = d[f32].astype(np.float32)
    return d.sort_values(["city", "station", "time"]).reset_index(drop=True)
