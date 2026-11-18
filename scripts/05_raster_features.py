"""Sample real land-cover (ESA WorldCover 2021, 10 m) and elevation (Copernicus GLO-30 DSM, 30 m) rasters at stations.

CRS handling
  * rasters are EPSG:4326 (degrees); stations are WGS84 lon/lat.
  * buffers are built in the station's local UTM zone (metres) with pyproj/GeoPandas,
    then transformed to the raster CRS before masking, so a "250 m" buffer is really 250 m.
Outputs: data/raw/raster_station_features.csv
"""
import os, sys
import numpy as np, pandas as pd, geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from shapely.geometry import Point

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAS = os.path.join(ROOT, os.environ.get("HEATLAB_RASTERS", "data/raw/rasters"))
BANDS = [100, 250, 500]
WC = {"tree": [10], "lowveg": [20, 30], "crop": [40], "built": [50], "bare": [60], "water": [80, 90]}


def zonal(src, geom_ll, nodata, bounds_pad=1e-4):
    """Pixels of `src` inside a lon/lat polygon -> (values, mask-of-valid)."""
    minx, miny, maxx, maxy = geom_ll.bounds
    win = from_bounds(minx - bounds_pad, miny - bounds_pad, maxx + bounds_pad, maxy + bounds_pad, src.transform)
    win = win.round_offsets().round_lengths()
    arr = src.read(1, window=win, boundless=True, fill_value=nodata).astype("float64")
    tf = src.window_transform(win)
    inside = ~geometry_mask([geom_ll], out_shape=arr.shape, transform=tf, all_touched=False)
    v = arr[inside]
    return v[v != nodata]


def main():
    st = pd.read_csv(os.path.join(ROOT, "data/raw/osm_station_features.csv"))[["city", "station", "lat", "lon"]]
    rows = []
    for city, g in st.groupby("city"):
        pts = gpd.GeoDataFrame(g.copy(), geometry=gpd.points_from_xy(g.lon, g.lat), crs="EPSG:4326")
        utm = pts.estimate_utm_crs()
        pu = pts.to_crs(utm)
        wc = rasterio.open(f"{RAS}/worldcover2021_{city}.tif")
        dm = rasterio.open(f"{RAS}/cop_dsm_glo30_{city}.tif")
        assert wc.crs.to_epsg() == 4326 and dm.crs.to_epsg() == 4326
        for (_, r), pt in zip(g.iterrows(), pu.geometry):
            f = {"city": city, "station": r.station, "utm_epsg": utm.to_epsg()}
            # point samples (point is already lon/lat for the rasters' CRS)
            f["wc_class_pt"] = int(next(wc.sample([(r.lon, r.lat)])).item())
            z0 = next(dm.sample([(r.lon, r.lat)], masked=True))[0]
            f["dsm_pt"] = float(z0) if not np.ma.is_masked(z0) else np.nan
            ring_means = {}
            for b in BANDS:
                circ_ll = gpd.GeoSeries([pt.buffer(b, 64)], crs=utm).to_crs("EPSG:4326").iloc[0]
                v = zonal(wc, circ_ll, wc.nodata or 0)
                n = max(len(v), 1)
                for name, codes in WC.items():
                    f[f"wc_{name}_{b}"] = float(np.isin(v, codes).sum() / n)
                d = zonal(dm, circ_ll, dm.nodata)
                ring_means[b] = float(d.mean()) if len(d) else np.nan
                f[f"dsm_mean_{b}"] = ring_means[b]
                if b == 500:
                    f["dsm_std_500"] = float(d.std()) if len(d) else np.nan
                    f["dsm_range_500"] = float(d.max() - d.min()) if len(d) else np.nan
            f["dsm_tpi_250"] = f["dsm_pt"] - ring_means[250]
            f["dsm_tpi_500"] = f["dsm_pt"] - ring_means[500]
            rows.append(f)
        print(city, len(g), "stations done, UTM", utm.to_epsg(), flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "data/raw/raster_station_features.csv"), index=False)
    print(out.describe().T[["count", "mean", "min", "max"]].round(3).to_string())
    print("NaN cells:", int(out.isna().sum().sum()))


if __name__ == "__main__":
    main()
