"""Vector processing of OpenStreetMap surroundings with GeoPandas / Shapely, with explicit CRS handling.

Pipeline per city
  1. raw OSM ways / relations (WGS84 lon/lat, EPSG:4326) -> GeoDataFrame
  2. multipolygon relations are assembled with polygonize (outer rings minus inner rings)
  3. everything is projected to the city's local UTM zone (metres) before any distance or area is computed
  4. land-cover classes are resolved with the same priority as the original grid method
     (water > building > vegetation) using exact polygon overlay instead of a 15 m grid
  5. per station: area fractions inside 100/250/500 m circles, 8 compass-sector wedges (50-500 m ring),
     and road length density.
"""
from __future__ import annotations

import json
import math
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union
from shapely.validation import make_valid

BANDS = (100, 250, 500)
RING_MIN, RING_MAX = 50.0, 500.0

VEG_NAT = {"wood", "scrub", "grassland", "heath", "wetland"}
VEG_LANDUSE = {"forest", "grass", "meadow", "recreation_ground", "village_green", "orchard", "vineyard",
               "allotments", "cemetery", "farmland"}
VEG_LEISURE = {"park", "garden", "pitch", "golf_course", "nature_reserve", "playground"}


def classify(t: dict) -> str | None:
    if "building" in t:
        return "bld"
    if t.get("natural") == "water" or t.get("waterway") == "riverbank" or t.get("landuse") in {"basin", "reservoir"} or "water" in t:
        return "wat"
    if t.get("natural") in VEG_NAT or t.get("landuse") in VEG_LANDUSE or t.get("leisure") in VEG_LEISURE:
        return "veg"
    return None


def _closed(g):
    return len(g) >= 4 and g[0] == g[-1]


def raw_to_gdf(raw: dict) -> gpd.GeoDataFrame:
    """Overpass `out geom` elements (compact form written by tools/overpass_geometries.js) -> GeoDataFrame in EPSG:4326."""
    rows = []
    for key, e in raw.items():
        t = e["t"]
        if "g" in e:  # way
            g = e["g"]
            if "highway" in t:
                if len(g) >= 2:
                    rows.append(("road", key, LineString(g)))
                continue
            c = classify(t)
            if c and _closed(g):
                rows.append((c, key, Polygon(g)))
        else:  # relation: assemble rings from (possibly split) member ways
            c = classify(t)
            if not c:
                continue
            outer = [LineString(m["g"]) for m in e["m"] if m["r"] == "outer" and len(m["g"]) >= 2]
            inner = [LineString(m["g"]) for m in e["m"] if m["r"] == "inner" and len(m["g"]) >= 2]
            if not outer:
                continue
            shells = unary_union(list(polygonize(unary_union(outer))))
            if shells.is_empty:
                continue
            if inner:
                holes = unary_union(list(polygonize(unary_union(inner))))
                shells = shells.difference(holes)
            if not shells.is_empty:
                rows.append((c, key, shells))
    gdf = gpd.GeoDataFrame(rows, columns=["cls", "osm_id", "geometry"], crs="EPSG:4326")
    gdf["geometry"] = gdf.geometry.apply(lambda g: g if g.is_valid else make_valid(g))
    return gdf


def wedge(cx: float, cy: float, bearing_deg: float, r0: float, r1: float, half=22.5, n=24) -> Polygon:
    """Annular sector centred on a compass bearing (0 = north, clockwise) in projected metres."""
    angs = np.deg2rad(np.linspace(bearing_deg - half, bearing_deg + half, n))
    outer = [(cx + r1 * math.sin(a), cy + r1 * math.cos(a)) for a in angs]
    inner = [(cx + r0 * math.sin(a), cy + r0 * math.cos(a)) for a in angs[::-1]]
    return Polygon(outer + inner)


def _area_frac(geom, zone: Polygon) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    return float(geom.intersection(zone).area / zone.area)


def station_features(city_gdf_utm: gpd.GeoDataFrame, x: float, y: float) -> dict:
    pt = Point(x, y)
    zone = pt.buffer(RING_MAX + 30, 64)
    sidx = city_gdf_utm.sindex
    cand = city_gdf_utm.iloc[sidx.query(zone, predicate="intersects")]
    by = {c: cand[cand.cls == c].geometry for c in ("bld", "veg", "wat", "road")}
    # priority overlay: water > building > vegetation (identical priority to the previous grid method)
    W = unary_union(list(by["wat"])) if len(by["wat"]) else Polygon()
    B = unary_union(list(by["bld"])) if len(by["bld"]) else Polygon()
    V = unary_union(list(by["veg"])) if len(by["veg"]) else Polygon()
    B = B.difference(W) if not W.is_empty else B
    V = V.difference(W) if not W.is_empty else V
    V = V.difference(B) if not B.is_empty else V
    R = unary_union(list(by["road"])) if len(by["road"]) else LineString()
    f = {}
    for b in BANDS:
        z = pt.buffer(b, 64)
        f[f"bld_{b}"], f[f"veg_{b}"], f[f"wat_{b}"] = _area_frac(B, z), _area_frac(V, z), _area_frac(W, z)
        f[f"road_{b}"] = (R.intersection(z).length / z.area * 1000.0) if not R.is_empty else 0.0
    for k in range(8):
        wz = wedge(x, y, 45.0 * k, RING_MIN, RING_MAX)
        f[f"sbld_{k}"], f[f"sveg_{k}"], f[f"swat_{k}"] = _area_frac(B, wz), _area_frac(V, wz), _area_frac(W, wz)
    f["nPoly"] = int((cand.cls != "road").sum())
    f["nRoad"] = int((cand.cls == "road").sum())
    return f


def city_features(raw: dict, stations: pd.DataFrame) -> pd.DataFrame:
    """stations: columns station, lat, lon (WGS84). Returns one feature row per station."""
    gdf = raw_to_gdf(raw)
    pts = gpd.GeoDataFrame(stations.copy(), geometry=gpd.points_from_xy(stations.lon, stations.lat), crs="EPSG:4326")
    utm = pts.estimate_utm_crs()
    gdf_u, pts_u = gdf.to_crs(utm), pts.to_crs(utm)
    out = []
    for (_, s), p in zip(stations.iterrows(), pts_u.geometry):
        out.append({"station": s.station, **station_features(gdf_u, p.x, p.y), "utm_epsg": utm.to_epsg()})
    return pd.DataFrame(out)
