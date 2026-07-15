"""Tests for the prediction API (milestone 5a)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def _payload(state, **extra):
    return {"real": list(state.real), "imag": list(state.imag), **extra}


@pytest.fixture()
def valid_state():
    rng = np.random.default_rng(1)
    s = rng.normal(size=4) + 1j * rng.normal(size=4)
    return s / np.linalg.norm(s)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_validate_exact_accepts_valid(valid_state):
    r = client.post("/validate", json=_payload(valid_state))
    body = r.json()
    assert r.status_code == 200
    assert body["mode"] == "exact" and body["valid"] is True


def test_validate_exact_rejects_scaled(valid_state):
    r = client.post("/validate", json=_payload(1.2 * valid_state))
    assert r.json()["valid"] is False


def test_validate_noisy_mode_and_budget_warning(valid_state):
    # Large budget: noise well below the threshold, verdict trustworthy
    r = client.post("/validate", json=_payload(valid_state, n_shots=10_000))
    body = r.json()
    assert body["mode"] == "noisy"
    assert body["valid"] is True and body["budget_sufficient"] is True
    # Tiny budget: the API must flag a noise-dominated verdict
    r = client.post("/validate", json=_payload(valid_state, n_shots=10))
    assert r.json()["budget_sufficient"] is False


def test_validate_rejects_mismatched_lengths():
    r = client.post("/validate", json={"real": [1.0, 0.0], "imag": [0.0]})
    assert r.status_code == 422


def test_preparation_qa_channels(valid_state):
    target = _payload(valid_state)
    base = {"target_real": target["real"], "target_imag": target["imag"]}

    # ok: state == target
    r = client.post("/preparation-qa", json=_payload(valid_state, **base))
    assert r.json()["error_type"] == "ok" and r.json()["prep_ok"] is True

    # gain error: scaled copy - fidelity blind, norm channel catches it
    r = client.post("/preparation-qa", json=_payload(1.3 * valid_state, **base))
    body = r.json()
    assert body["error_type"] == "gain_error"
    assert body["pointing_channel_ok"] is True

    # pointing error: normalised state orthogonal-ish to target
    rng = np.random.default_rng(2)
    chi = rng.normal(size=4) + 1j * rng.normal(size=4)
    chi -= np.vdot(valid_state, chi) * valid_state
    chi /= np.linalg.norm(chi)
    mixed = np.sqrt(0.7) * valid_state + np.sqrt(0.3) * chi
    r = client.post("/preparation-qa", json=_payload(mixed, **base))
    body = r.json()
    assert body["error_type"] == "pointing_error"
    assert body["gain_channel_ok"] is True


def test_preparation_qa_validation(valid_state):
    # non-normalised target refused
    r = client.post(
        "/preparation-qa",
        json=_payload(
            valid_state,
            target_real=[1.0, 0, 0, 0.5],
            target_imag=[0.0, 0, 0, 0],
        ),
    )
    assert r.status_code == 422
    # dimension mismatch refused
    r = client.post(
        "/preparation-qa",
        json=_payload(valid_state, target_real=[1.0, 0], target_imag=[0.0, 0]),
    )
    assert r.status_code == 422
