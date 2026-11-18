"""Run the validity-guarded agent on the v2 feature set. Writes results_v2/ and models_v2/ (promote after review)."""
import os, sys
os.environ["HEATLAB_FEATURES"] = "v2"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from heatlab.agent import run

root = os.path.join(os.path.dirname(__file__), "..")
run(f"{root}/data/processed/dataset_v2.parquet", f"{root}/results_v2", f"{root}/models_v2")
print("done")
