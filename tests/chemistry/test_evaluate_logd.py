# tests/chemistry/test_evaluate_logd.py
import pytest
import pandas as pd


def test_extract_logd_from_responses():
    from chemistry.evaluate.evaluate_logd import extract_predictions
    responses = [
        "Step 1: ... Prediction: LogD ≈ 2.3 (range 1.8–2.8, confidence: medium)",
        "Prediction: LogD = -1.5",
        "The LogD should be approximately 0.7",
        "No numeric prediction here",
    ]
    preds = extract_predictions(responses)
    assert preds[0] == pytest.approx(2.3, abs=0.01)
    assert preds[1] == pytest.approx(-1.5, abs=0.01)
    assert preds[2] == pytest.approx(0.7, abs=0.1)
    assert preds[3] is None


def test_compute_metrics():
    from chemistry.evaluate.evaluate_logd import compute_metrics
    experimental = [1.0, 2.0, 3.0, 0.5, -1.0]
    predicted    = [1.1, 2.2, 2.8, 0.6, -0.8]
    metrics = compute_metrics(experimental, predicted)
    assert "spearman_r" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "n_evaluated" in metrics
    assert metrics["spearman_r"] > 0.95   # these are close predictions
    assert metrics["rmse"] < 0.3


def test_compute_fg_accuracy():
    from chemistry.evaluate.evaluate_logd import compute_fg_accuracy
    import yaml
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    # Response that correctly names a FG
    responses = ["I identify a carboxylic acid group and aromatic ring. LogD ≈ 1.5"]
    smiles = ["c1ccccc1C(=O)O"]  # benzoic acid: has carboxylic_acid + aromatic_ring
    acc = compute_fg_accuracy(responses, smiles, fg_defs)
    assert acc["fg_recall"] > 0.5


def test_evaluation_report():
    from chemistry.evaluate.evaluate_logd import generate_report
    metrics = {"spearman_r": 0.78, "rmse": 1.9, "mae": 1.4, "n_evaluated": 50}
    report = generate_report(metrics, gnn_baseline={"spearman_r": 0.76, "rmse": 2.1})
    assert "spearman_r" in report
    assert "0.78" in report
    assert "PASS" in report or "FAIL" in report
