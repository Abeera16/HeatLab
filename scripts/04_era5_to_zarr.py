"""Convert the four Open-Meteo ERA5 CSVs into one xarray Dataset stored as Zarr.
Dims: (city, time). Cities have different spans; missing hours are NaN (outer join).
"""
import glob, os, re, sys
import numpy as np, pandas as pd, xarray as xr, zarr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITY = {"48.00N1.75W": "Rennes", "47.25N7.25E": "Biel", "45.25N19.75E": "NoviSad", "51.00N3.75E": "Ghent"}
das = []
for f in sorted(glob.glob(f"{ROOT}/data/raw/era5/*.csv")):
    key = re.search(r"open-meteo-(.+?)\d+m\.csv", os.path.basename(f)).group(1)
    city = CITY[key]
    meta = pd.read_csv(f, nrows=1)
    df = pd.read_csv(f, skiprows=3, parse_dates=["time"]).set_index("time")
    df.columns = [re.sub(r"\s*\(.*\)", "", c) for c in df.columns]
    ds = xr.Dataset.from_dataframe(df).astype("float32")
    ds = ds.expand_dims(city=[city])
    ds = ds.assign_coords(lat=("city", [float(meta.latitude[0])]), lon=("city", [float(meta.longitude[0])]),
                          grid_elev_m=("city", [float(meta.elevation[0])]))
    das.append(ds)
ds = xr.concat(das, dim="city", join="outer")
units = {"temperature_2m": "degC", "relative_humidity_2m": "%", "dew_point_2m": "degC", "surface_pressure": "hPa",
         "cloud_cover": "%", "wind_speed_10m": "m s-1", "wind_direction_10m": "degree", "shortwave_radiation": "W m-2",
         "direct_radiation": "W m-2", "diffuse_radiation": "W m-2", "precipitation": "mm"}
for v, u in units.items(): ds[v].attrs["units"] = u
ds.attrs.update(title="ERA5 hourly reanalysis at the 4 HeatLab city grid cells", source="ERA5 via Open-Meteo archive CSV",
                crs="EPSG:4326 (cell-centre lat/lon)", note="Converted from CSV; not native NetCDF. NaN = hour not in that city's record.")
out = f"{ROOT}/data/processed/era5.zarr"
import shutil; shutil.rmtree(out, ignore_errors=True)
ds.chunk({"city": 1, "time": 8760}).to_zarr(out, mode="w", consolidated=True)
chk = xr.open_zarr(out)
print(chk); 
raw = pd.concat([pd.read_csv(f, skiprows=3).iloc[:, :2] for f in glob.glob(f"{ROOT}/data/raw/era5/*.csv")])
print("raw non-null T rows:", raw.iloc[:,1].notna().sum(), "| zarr non-null:", int(chk.temperature_2m.notnull().sum()))
