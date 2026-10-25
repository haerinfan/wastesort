import hashlib
import json
import subprocess
import sys

import pytest

from scripts import launch
from wastesort.common import CLASSES
from wastesort.doctor import check_artifact


def test_install_retries_failure_and_skips_success(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 12, 0))
    environment = tmp_path / ".venv"
    executable = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    executable.parent.mkdir(parents=True)
    executable.touch()
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example==1.0\n")
    marker = environment / "wastesort-requirements.sha256"
    calls = []

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        launch.prepare(tmp_path)
    assert not marker.exists()
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append(a))
    assert launch.prepare(tmp_path) == executable
    launch.prepare(tmp_path)
    assert len(calls) == 1
    requirements.write_text("example==2.0\n")
    launch.prepare(tmp_path)
    assert len(calls) == 2
    launch.prepare(tmp_path, reinstall=True)
    assert len(calls) == 3


def test_doctor_rejects_changed_labels_model_and_external_path(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    artifact = models / "deployed.pt"
    artifact.write_bytes(b"test checkpoint")
    metadata = {"artifact": "models/deployed.pt", "classes": CLASSES,
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
    record = models / "deployment.json"
    record.write_text(json.dumps(metadata))
    assert check_artifact(tmp_path) == metadata
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="incomplete or changed"):
        check_artifact(tmp_path)
    metadata["classes"] = list(reversed(CLASSES))
    record.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="label order"):
        check_artifact(tmp_path)
    metadata["artifact"] = "../outside.pt"
    record.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="inside the project"):
        check_artifact(tmp_path)
