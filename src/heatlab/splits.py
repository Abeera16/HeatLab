"""Evaluation protocols. Every split returns boolean train/test masks over the dataset rows.

Protocols
---------
random_rows       naive 80/20 row split (leaky: same station and neighbouring hours on both sides)
random_stations   hold out random stations
spatial_blocks    hold out spatially contiguous station clusters, with a buffer around them
temporal          train on the earlier period of every station, test on the later one
leave_city_out    train on three cities, test on the fourth
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def station_table(d: pd.DataFrame) -> pd.DataFrame:
    s = d.groupby(["city", "station"], as_index=False).agg(lat=("lat", "first"), lon=("lon", "first"), n=("t_obs", "size"))
    return s


def _km(lat1, lon1, lat2, lon2):
    kx = 111.32 * np.cos(np.deg2rad((lat1 + lat2) / 2))
    return np.hypot((lon1 - lon2) * kx, (lat1 - lat2) * 110.54)


def spatial_folds(d: pd.DataFrame, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Assign each station a fold id 0..k-1 by clustering coordinates within each city."""
    s = station_table(d)
    s["fold"] = -1
    for city, g in s.groupby("city"):
        kk = min(k, len(g))
        xy = np.c_[g["lon"] * np.cos(np.deg2rad(g["lat"].mean())), g["lat"]]
        lab = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(xy)
        # order clusters west->east so fold ids are reproducible and comparable
        order = np.argsort([xy[lab == c, 0].mean() for c in range(kk)])
        remap = {c: i for i, c in enumerate(order)}
        s.loc[g.index, "fold"] = [remap[c] for c in lab]
    return s


def spatial_block_masks(d: pd.DataFrame, fold: int, k: int = 5, buffer_km: float = 0.5, folds_table=None):
    """Train = stations outside the held-out cluster and farther than `buffer_km` from every test station."""
    s = folds_table if folds_table is not None else spatial_folds(d, k)
    test_ids, drop_ids = set(), set()
    for city, g in s.groupby("city"):
        te = g[g.fold == fold]
        test_ids |= set(zip(te.city, te.station))
        for r in g[g.fold != fold].itertuples():
            if len(te) and _km(r.lat, r.lon, te.lat.to_numpy(), te.lon.to_numpy()).min() < buffer_km:
                drop_ids.add((r.city, r.station))
    key = list(zip(d["city"], d["station"]))
    is_test = np.fromiter((x in test_ids for x in key), bool, len(d))
    is_drop = np.fromiter((x in drop_ids for x in key), bool, len(d))
    return ~is_test & ~is_drop, is_test, {"test_stations": len(test_ids), "buffer_dropped": len(drop_ids)}


def random_station_masks(d: pd.DataFrame, fold: int, k: int = 5, seed: int = 0):
    s = station_table(d)
    rng = np.random.default_rng(seed)
    s["fold"] = -1
    for city, g in s.groupby("city"):
        ids = rng.permutation(len(g))
        s.loc[g.index, "fold"] = ids % min(k, len(g))
    test = set(zip(s[s.fold == fold].city, s[s.fold == fold].station))
    key = list(zip(d["city"], d["station"]))
    is_test = np.fromiter((x in test for x in key), bool, len(d))
    return ~is_test, is_test


def random_row_masks(d: pd.DataFrame, fold: int, k: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    f = rng.integers(0, k, len(d))
    return f != fold, f == fold


def temporal_masks(d: pd.DataFrame, frac_train: float = 0.75, gap_days: int = 2):
    """Per-station chronological split at the `frac_train` quantile of that city's time span."""
    train = np.zeros(len(d), bool)
    test = np.zeros(len(d), bool)
    info = {}
    for city, idx in d.groupby("city").indices.items():
        t = d["time"].iloc[idx]
        cut = t.min() + (t.max() - t.min()) * frac_train
        train[idx] = (t < cut - pd.Timedelta(days=gap_days)).to_numpy()
        test[idx] = (t >= cut).to_numpy()
        info[city] = {"cut": str(cut)[:10]}
    return train, test, info


def leave_city_out_masks(d: pd.DataFrame, city: str):
    te = (d["city"] == city).to_numpy()
    return ~te, te
