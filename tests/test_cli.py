import subprocess
import sys


def run_help(script_name):
    result = subprocess.run(
        [sys.executable, script_name, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_train_help_runs_without_training_dependencies():
    run_help("train.py")


def test_predict_help_runs_without_training_dependencies():
    run_help("predict.py")
