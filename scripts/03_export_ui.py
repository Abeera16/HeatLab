"""Pack real results + out-of-fold predictions into app/data.js (embedded so the UI works from file://)."""
import json, os, numpy as np, pandas as pd
root = os.path.join(os.path.dirname(__file__), "..")
r = json.load(open(f"{root}/results/results.json"))
o = pd.read_parquet(f"{root}/results/oof_predictions.parquet")
st = pd.read_csv(f"{root}/data/raw/osm_station_features.csv")
o["hour"] = pd.to_datetime(o["time"]).dt.hour
o["ao"] = o.t_obs - o.e_t2m
o["ap"] = o.lgbm_full
o["ah"] = o.hour_mean
# lgbm_full / hour_mean columns hold predicted temperature -> convert to anomaly
o["ap"] = o.lgbm_full          # model columns are anomalies (station - ERA5)
o["ah"] = o.hour_mean
o["err"] = o.e_t2m + o.lgbm_full - o.t_obs
stations = []
for _, s in st.iterrows():
    g = o[(o.city == s.city) & (o.station == s.station)]
    if g.empty: continue
    h = g.groupby("hour")[["ao", "ap"]].mean().reindex(range(24))
    stations.append(dict(city=s.city, id=str(s.station), lat=s.lat, lon=s.lon, elev=float(s.elev),
        bld=float(s.bld_250), veg=float(s.veg_250), wat=float(s.wat_250), road=float(s.road_250),
        rmse=float(np.sqrt((g.err**2).mean())), n=int(len(g)),
        obs=[None if np.isnan(x) else round(float(x), 2) for x in h.ao],
        pred=[None if np.isnan(x) else round(float(x), 2) for x in h.ap]))
weeks = {}
for (c, s), g in o.groupby(["city", "station"]):
    g = g.sort_values("time"); t = pd.to_datetime(g.time)
    mid = t.iloc[len(t) // 2]; w = g[(t >= mid) & (t < mid + pd.Timedelta(days=7))]
    weeks[f"{c}|{s}"] = dict(start=str(mid)[:16], t_obs=[round(float(x), 2) for x in w.t_obs],
        era5=[round(float(x), 2) for x in w.e_t2m], pred=[round(float(x), 2) for x in (w.e_t2m + w.lgbm_full)],
        q10=[round(float(x), 2) for x in (w.e_t2m + w.q10)], q90=[round(float(x), 2) for x in (w.e_t2m + w.q90)])

# ---------- aggregates for the interactive dashboard (all from out-of-fold predictions) ----------
cities = ["Rennes", "Biel", "NoviSad", "Ghent"]
o["ci"] = o.city.map({c: i for i, c in enumerate(cities)})
o["nf"] = o.is_night.astype(int)
o["ym"] = pd.to_datetime(o.time).dt.strftime("%Y-%m")
E = {"e": -o.anom, "h": o.hour_mean - o.anom, "l": o.lgbm_full - o.anom, "i": o.idw_live_sensors - o.anom}
for k, v in E.items():
    o["se_" + k] = v ** 2; o["ae_" + k] = v.abs(); o["b_" + k] = v; o["w_" + k] = (v.abs() <= 1).astype(float)
o["cov"] = ((o.anom >= o.q10) & (o.anom <= o.q90)).astype(float)
o["n"] = 1.0
cols = ["n", "cov"] + [f"{a}_{k}" for k in "ehli" for a in ("se", "ae", "b", "w")]
g = o.groupby(["ci", "hour", "nf"])[cols].sum().reset_index()
agg = [[int(r.ci), int(r.hour), int(r.nf)] + [round(float(r[c]), 2) for c in cols] for _, r in g.iterrows()]
gm = o.groupby(["ci", "ym", "nf"])[["n", "se_e", "se_h", "se_l"]].sum().reset_index()
yms = sorted(o.ym.unique()); yi = {y: i for i, y in enumerate(yms)}
monthly = [[int(r.ci), yi[r.ym], int(r.nf), int(r.n), round(r.se_e, 1), round(r.se_h, 1), round(r.se_l, 1)] for _, r in gm.iterrows()]
o["tb"] = (np.floor(o.e_t2m / 5) * 5).astype(int)
gt = o.groupby(["ci", "tb"])[["n", "se_e", "se_h", "se_l"]].sum().reset_index()
temp = [[int(r.ci), int(r.tb), int(r.n), round(r.se_e, 1), round(r.se_h, 1), round(r.se_l, 1)] for _, r in gt.iterrows() if r.n >= 300]
samp = []
for c in cities:
    t = o[o.city == c].sample(4000, random_state=3)
    samp += [[cities.index(c), int(h), int(nf), round(float(a), 2), round(float(hm), 2), round(float(l), 2)] for h, nf, a, hm, l in zip(t.hour, t.nf, t.anom, t.hour_mean, t.lgbm_full)]
def rm(g, k): return float(np.sqrt(g["se_" + k].mean())) if len(g) else None
for s_ in stations:
    g = o[(o.city == s_["city"]) & (o.station == s_["id"])]
    parts = [g, g[g.nf == 0], g[g.nf == 1]]
    s_["re"] = [round(rm(x, "e"), 3) if len(x) else None for x in parts]
    s_["rh"] = [round(rm(x, "h"), 3) if len(x) else None for x in parts]
    s_["rl"] = [round(rm(x, "l"), 3) if len(x) else None for x in parts]
    s_["bl"] = [round(float(x.b_l.mean()), 3) if len(x) else None for x in parts]
    s_["ad"] = [round(float(x.anom.mean()), 3) if len(x) else None for x in parts]   # mean measured anomaly: all/day/night
    s_["pd"] = [round(float(x.lgbm_full.mean()), 3) if len(x) else None for x in parts]


# ---------- geospatial / GPU results (optional: only if the experiments were run) ----------
def J(name):
    f = f"{root}/results/{name}"
    return json.load(open(f)) if os.path.exists(f) else None
geo = None
a0, a1, dl = J("geo_ablation_raster.json"), J("geo_ablation_geo.json"), J("dl_vs_lgbm.json")
if a0 and a1 and dl:
    lab = {"no_static": "Weather and time only", "elev_only": "+ elevation and distance to centre",
           "osm_js": "+ OSM fractions, browser grid", "osm_gpd": "+ OSM fractions, GeoPandas (UTM)",
           "raster": "WorldCover + DSM only, no OSM", "osm_js+raster": "OSM grid + rasters", "osm_gpd+raster": "OSM GeoPandas + rasters"}
    order = ["no_static", "elev_only", "osm_js", "osm_gpd", "raster", "osm_js+raster", "osm_gpd+raster"]
    allsets = {**a0["sets"], **a1["sets"]}; nf = {**a0["feature_counts"], **a1["feature_counts"]}
    abl = [dict(k=k, label=lab[k], n=nf[k], rmse=allsets[k]["overall"]["rmse"], mae=allsets[k]["overall"]["mae"],
                by_city={c: v["rmse"] for c, v in allsets[k]["by_city"].items()},
                folds_vs_grid=allsets[k].get("delta_vs_osm_js")) for k in order]
    zinfo = None
    try:
        import xarray as xr
        z = xr.open_zarr(f"{root}/data/processed/era5.zarr")
        zinfo = dict(dims={k: int(v) for k, v in z.sizes.items()}, nvars=len(z.data_vars),
                     t0=str(z.time.values[0])[:10], t1=str(z.time.values[-1])[:10])
    except Exception as e:
        print("zarr info skipped:", e)
    geo = dict(abl=abl, referee=J("vector_referee.json"), dl=dl, mlp=J("mlp_metrics.json"), zarr=zinfo,
               spatial_rmse_published=r["spatial_cv"]["overall"]["lgbm_full"]["rmse"])

keep = {k: r[k] for k in ["data_audit", "tuning", "leakage", "temporal", "loco", "shap", "observed_relations", "sensitivity", "gates"]}
sc = r["spatial_cv"]; keep["spatial"] = {k: sc[k] for k in ["overall", "by_city", "by_hour", "by_daynight", "coverage"]}
card = json.load(open(f"{root}/models/model_card.json"))
out = dict(stations=stations, weeks=weeks, results=keep, card=card, cities=cities, agg=dict(cols=cols, rows=agg), yms=yms, monthly=monthly, temp=temp, samp=samp, geo=geo)
open(f"{root}/app/data.js", "w").write("window.HEATLAB=" + json.dumps(out, separators=(",", ":"), default=float) + ";")
print(len(stations), os.path.getsize(f"{root}/app/data.js") // 1024, "KB")
