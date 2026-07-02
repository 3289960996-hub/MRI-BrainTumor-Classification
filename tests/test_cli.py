import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_help(script_name):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / script_name), "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_train_help_runs_without_training_dependencies():
    run_help("train.py")


def test_predict_help_runs_without_training_dependencies():
    run_help("predict.py")
