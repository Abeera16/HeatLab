# HeatLab validity report

Generated 2026-11-17 21:38 UTC

| Gate | Status | Evidence |
|---|---|---|
| G1 Enough stations for spatially independent evaluation | **WARN** | Ghent have fewer than 20 stations |
| G2 No unlabeled or missing targets | **PASS** | missing target/context rate 0.0000 |
| G3 Coordinates valid and consistent (WGS84 lat/lon, no swapped axes) | **PASS** | 0 out-of-range coordinates; 0 cities with implausible spread |
| G4 Surrounding land-cover fractions are valid fractions | **PASS** | all bld/veg/wat fractions in [0,1] |
| G5 Station elevation consistent with the ERA5 grid cell | **PASS** | 0 stations differ by more than 400 m from the ERA5 cell elevation |
| G6 No co-located duplicate stations | **WARN** | 1 station pairs closer than 50 m |
| G7 Held-out stations never appear in training | **PASS** | 0 stations shared between train and test across 5 folds |
| G8 Spatial buffer between train and test stations | **PASS** | minimum train-to-test distance 0.53 km (buffer 0.5 km) |
| G9 Hyper-parameters tuned without touching test stations | **PASS** | chosen 'medium' using an inner split of the training stations of fold 0 |
| G10 Prediction intervals are calibrated out of sample | **WARN** | 80% interval covers 70.8% of held-out observations |
| G11 Naive random-row validation blocked from headline results | **PASS** | random-row RMSE 0.921 C vs spatial-block RMSE 1.088 C (15% apparent error reduction from leakage) |
| G12 Production model carries an explicit scope statement | **PASS** | model_card.json records training cities and unvalidated regimes |
