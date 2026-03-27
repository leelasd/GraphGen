# tests/chemistry/test_generate_dpo_pairs.py
import pytest
import json


def test_dpo_pair_format():
    from chemistry.dpo.generate_dpo_pairs import DPOPair
    pair = DPOPair(
        prompt="Predict LogD for CCO",
        chosen="Step 1: I identify a hydroxyl group... Prediction: LogD ≈ -0.3",
        rejected="The molecule has no notable features. LogD ≈ 2.5",
    )
    assert pair.prompt
    assert pair.chosen
    assert pair.rejected
    d = pair.to_dict()
    assert set(d.keys()) == {"prompt", "chosen", "rejected"}


def test_extract_logd_from_response():
    from chemistry.dpo.generate_dpo_pairs import extract_logd_prediction
    resp = "After analysis, the LogD ≈ 2.3 (range 1.8–2.8, confidence: medium)"
    val = extract_logd_prediction(resp)
    assert val is not None
    assert abs(val - 2.3) < 0.01

    resp2 = "Prediction: LogD = -1.5"
    assert abs(extract_logd_prediction(resp2) - (-1.5)) < 0.01

    assert extract_logd_prediction("No prediction here") is None


def test_build_perturbation_rejected():
    from chemistry.dpo.generate_dpo_pairs import perturb_reasoning
    correct = (
        "Step 1: Hydroxyl group identified.\n"
        "Step 2: HBD = 1, decreases LogD.\n"
        "Step 3: At pH 7.4, neutral.\n"
        "Prediction: LogD ≈ -0.3"
    )
    rejected = perturb_reasoning(correct, perturbation="ignore_ionization")
    assert rejected != correct
    assert "LogD" in rejected


def test_save_dpo_pairs():
    from chemistry.dpo.generate_dpo_pairs import DPOPair, save_dpo_pairs
    import tempfile, json
    from pathlib import Path
    pairs = [
        DPOPair("Q1", "good answer 1", "bad answer 1"),
        DPOPair("Q2", "good answer 2", "bad answer 2"),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "dpo.jsonl"
        save_dpo_pairs(pairs, path)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        obj = json.loads(lines[0])
        assert {"prompt", "chosen", "rejected"} == set(obj.keys())
