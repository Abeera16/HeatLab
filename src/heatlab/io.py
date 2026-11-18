"""Readers for the real data sources used by HeatLab.

* FAIRUrbTemp station files (SEF format, quality-control flags kept in the Meta column)
* ERA5 hourly reanalysis downloaded through the Open-Meteo archive API (CSV)
* OSM-derived station surroundings produced by tools/overpass_features.js
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

CITIES = {
    # city: (native sampling of the QC file we use, ERA5 csv glob, human name, country)
    "NoviSad": ("10min", "open-meteo-45.25N19.75E*.csv", "Novi Sad", "Serbia"),
    "Rennes": ("hourly", "open-meteo-48.00N1.75W*.csv", "Rennes", "France"),
    "Biel": ("10min", "open-meteo-47.25N7.25E*.csv", "Biel/Bienne", "Switzerland"),
    "Ghent": ("hourly", "open-meteo-51.00N3.75E*.csv", "Ghent", "Belgium"),
}


@dataclass
class Station:
    city: str
    sid: str
    lat: float
    lon: float
    alt: float | None
    n_obs: int
    n_flagged: int


def read_sef(path: str):
    """Return (header dict, DataFrame[time, value, flagged, flag_kind]) from one SEF file."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    hdr_i = next(i for i, l in enumerate(lines) if l.startswith("Year\t"))
    meta = {}
    for line in lines[:hdr_i]:
        parts = line.split("\t")
        if len(parts) >= 2:
            meta[parts[0]] = parts[1]
    df = pd.read_csv(path, sep="\t", skiprows=hdr_i, dtype=str, keep_default_na=False)
    value = pd.to_numeric(df["Value"].str.strip().replace({"NA": np.nan, "": np.nan}), errors="coerce")
    ts = pd.to_datetime(
        dict(year=df.Year.astype(int), month=df.Month.astype(int), day=df.Day.astype(int),
             hour=df.Hour.astype(int), minute=df.Minute.astype(int)), errors="coerce")
    flagged = df["Meta"].str.contains("qc=", regex=False)
    kind = df["Meta"].str.extract(r"qc=([a-z_]+)")[0]
    out = pd.DataFrame({"time": ts, "value": value, "flagged": flagged, "flag_kind": kind})
    out = out.dropna(subset=["time"]).drop_duplicates("time").set_index("time")
    return meta, out


def hourly_series(df: pd.DataFrame, native: str) -> pd.Series:
    """QC-clean hourly series centred on the full hour (matches ERA5 instantaneous fields).

    Flagged values are excluded. 10-min data are averaged over [HH-30min, HH+30min) and need at
    least 3 valid readings; hourly files are used as delivered.
    """
    good = df.loc[~df["flagged"] & df["value"].notna(), "value"]
    if native == "10min":
        shifted = good.copy()
        shifted.index = shifted.index + pd.Timedelta("30min")
        g = shifted.resample("1h")
        h = g.mean().where(g.count() >= 3)
        h.index = h.index  # label = full hour
        return h.dropna()
    return good.resample("1h").mean().dropna()


def load_stations(data_dir: str, cities=None):
    """Load every station of every city; return (long DataFrame, list[Station])."""
    frames, stations = [], []
    for city, (native, *_rest) in CITIES.items():
        if cities and city not in cities:
            continue
        for path in sorted(glob.glob(f"{data_dir}/{city}/SEF_output/QC/*_AirTemp_{native}_UTC.tsv")):
            sid = os.path.basename(path).split("_AirTemp")[0]
            meta, df = read_sef(path)
            h = hourly_series(df, native)
            if len(h) == 0:
                continue
            alt = meta.get("Alt")
            stations.append(Station(city, sid, float(meta["Lat"]), float(meta["Lon"]),
                                    None if alt in (None, "NA") else float(alt),
                                    int(df["value"].notna().sum()), int(df["flagged"].sum())))
            frames.append(pd.DataFrame({"city": city, "station": sid, "time": h.index, "t_obs": h.values}))
    return pd.concat(frames, ignore_index=True), stations


def load_era5(csv_path: str) -> tuple[pd.DataFrame, dict]:
    """Open-Meteo ERA5 CSV -> DataFrame indexed by UTC hour with short column names."""
    with open(csv_path, encoding="utf-8") as fh:
        head = [next(fh) for _ in range(3)]
    lat, lon, elev = head[1].strip().split(",")[:3]
    df = pd.read_csv(csv_path, skiprows=3)
    df["time"] = pd.to_datetime(df["time"])
    df.columns = [re.sub(r"\s*\(.*\)", "", c) for c in df.columns]
    df = df.rename(columns={
        "temperature_2m": "e_t2m", "relative_humidity_2m": "e_rh", "dew_point_2m": "e_dew",
        "surface_pressure": "e_p", "cloud_cover": "e_cloud", "wind_speed_10m": "e_ws",
        "wind_direction_10m": "e_wd", "shortwave_radiation": "e_sw", "direct_radiation": "e_dir",
        "diffuse_radiation": "e_dif", "precipitation": "e_pr"}).set_index("time")
    return df, {"lat": float(lat), "lon": float(lon), "elev": float(elev)}


def load_static(path: str) -> pd.DataFrame:
    """OSM surroundings + elevation per station.

    Produced by tools/overpass_features.js (runs in a browser against the public Overpass API and the
    Open-Meteo elevation API). Stored as CSV; a legacy JSON layout is also accepted.
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        return df.rename(columns={"elev": "elev", "nPoly": "n_poly", "nRoad": "n_road"})
    with open(path) as fh:
        raw = json.load(fh)
    rows = []
    for r in raw.values() if isinstance(raw, dict) else raw:
        if "f" not in r:
            continue
        row = {"city": r["city"], "station": r["id"], "lat": r["lat"], "lon": r["lon"], "elev": r["elev"],
               "n_poly": r.get("nPoly", np.nan), "n_road": r.get("nRoad", np.nan)}
        row.update(r["f"])
        rows.append(row)
    return pd.DataFrame(rows)
