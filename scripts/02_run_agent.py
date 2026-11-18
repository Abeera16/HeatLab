"""Run the full validity-guarded study (tuning, spatial CV, leakage study, transfer tests, final models)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from heatlab.agent import run

root = os.path.join(os.path.dirname(__file__), "..")
run(f"{root}/data/processed/dataset.parquet", f"{root}/results", f"{root}/models")
print("done")
