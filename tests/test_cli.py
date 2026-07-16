"""Tests for the qsv CLI - the CI-gate semantics."""

import json

import numpy as np
import pytest

from qsv.cli import main


@pytest.fixture()
def npy_valid(tmp_path):
    s = np.array([0.6, 0.8, 0.0, 0.0], dtype=complex)
    p = tmp_path / "valid.npy"
    np.save(p, s)
    return p


@pytest.fixture()
def npy_batch_mixed(tmp_path):
    batch = np.array([[0.6, 0.8, 0, 0], [0.9, 0.9, 0, 0]], dtype=complex)
    p = tmp_path / "mixed.npy"
    np.save(p, batch)
    return p


def test_exit_zero_when_all_valid(npy_valid, capsys):
    assert main(["validate", str(npy_valid)]) == 0
    assert "all valid" in capsys.readouterr().out


def test_exit_one_when_any_invalid(npy_batch_mixed, capsys):
    assert main(["validate", str(npy_batch_mixed)]) == 1
    assert "1 invalid" in capsys.readouterr().out


def test_exit_two_on_usage_error(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "missing.npy")]) == 2
    assert "error" in capsys.readouterr().err


def test_json_output(npy_batch_mixed, capsys):
    assert main(["validate", str(npy_batch_mixed), "--json"]) == 1
    body = json.loads(capsys.readouterr().out)
    assert body["n_states"] == 2 and body["n_invalid"] == 1
    assert body["results"][0]["valid"] is True


def test_noisy_mode_flag(npy_valid, capsys):
    assert main(["validate", str(npy_valid), "--n-shots", "500", "--quiet"]) == 0
    assert "noisy N=500" in capsys.readouterr().out


def test_csv_dataset_input(tmp_path):
    from qsv.data_generation import create_dataset

    df = create_dataset(n_valid=20, n_invalid=10, dim=4, seed=1)
    p = tmp_path / "ds.csv"
    df.to_csv(p, index=False)
    # dataset contains invalid states by construction -> exit 1
    assert main(["validate", str(p), "--quiet"]) == 1
