"""Reproduce into a fresh directory, preserving the published study."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

SOURCE = Path(__file__).resolve().parents[1]
(SOURCE / "runs").mkdir(exist_ok=True)
run = Path(tempfile.mkdtemp(prefix="experiment-", dir=SOURCE / "runs"))
shutil.copyfile(SOURCE / "protocol.json", run / "protocol.json")
(run / "data").mkdir()
(run / "reports").mkdir()
for relative in ["data/provenance.json", "data/splits.csv", "reports/data_audit.json"]:
    if (SOURCE / relative).exists():
        shutil.copyfile(SOURCE / relative, run / relative)
env = dict(os.environ, WASTESORT_ROOT=str(run))
print(f"Fresh experiment: {run}", flush=True)
for module in ["data", "preparation", "train", "intro_predictions", "evaluate", "exercises", "figures"]:
    subprocess.run([sys.executable, "-u", "-m", f"wastesort.{module}"], cwd=SOURCE, env=env, check=True)
print(f"Completed: {run}", flush=True)
