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

    resp3 = "LogD ~ 3.1"
    assert abs(extract_logd_prediction(resp3) - 3.1) < 0.01

    assert extract_logd_prediction("No prediction here") is None


def test_build_perturbation_rejected():
    from chemistry.dpo.generate_dpo_pairs import perturb_reasoning
    correct = (
        "Step 1: Hydroxyl group identified.\n"
        "Step 2: HBD = 1, decreases LogD.\n"
        "Step 3: At pH 7.4, neutral.\n"
        "Prediction: LogD ≈ -0.3"
    )
    expected_corrupted = {
        "ignore_ionization": pytest.approx(0.3, abs=0.01),   # sign-inverted: -(-0.3) = 0.3
        "wrong_ph": pytest.approx(1.7, abs=0.01),            # +2.0: -0.3 + 2.0 = 1.7
        "miss_fg": pytest.approx(1.2, abs=0.01),             # +1.5: -0.3 + 1.5 = 1.2
    }
    for strategy, expected_val in expected_corrupted.items():
        rejected = perturb_reasoning(correct, strategy)
        assert rejected != correct, f"Perturbation '{strategy}' did not modify the chain"
        assert "LogD" in rejected, f"Perturbation '{strategy}' removed LogD from output"
        import re
        match = re.search(r"Prediction: LogD ≈ (-?\d+\.?\d*)", rejected)
        assert match is not None, f"Perturbation '{strategy}' missing prediction line"
        assert float(match.group(1)) == expected_val, \
            f"Perturbation '{strategy}': expected {expected_val}, got {float(match.group(1))}"


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


def test_generate_dpo_pairs_logic():
    """Test pair generation logic using a mocked _call_llm."""
    from unittest.mock import patch
    from chemistry.dpo.generate_dpo_pairs import generate_dpo_pairs, DPOPair
    import pandas as pd

    accurate_response = (
        "Step 1: Hydroxyl group.\n"
        "Step 2: HBD = 1.\n"
        "Step 3: Neutral at pH 7.4.\n"
        "Prediction: LogD ≈ -0.3"
    )
    inaccurate_response = "No functional groups. Prediction: LogD ≈ 5.0"

    df = pd.DataFrame({
        "smiles": ["CCO", "c1ccccc1"],
        "logd_exp": [-0.31, 1.56],
    })
    gnn_preds = pd.Series([-2.5, 1.5], index=df.index)  # mol_0 GNN diverges by 2.2 > 1.0

    # mol_0: LLM accurate (|−0.3 − (−0.31)| = 0.01 < 0.5), GNN diverges (2.2 > 1.0) → Case 1 + Case 2 = 4 pairs
    # mol_1: LLM inaccurate (|5.0 − 1.56| = 3.44 > 0.5) → 0 pairs
    call_count = 0
    def mock_llm(prompt, model="llama-3-2-3b"):
        nonlocal call_count
        call_count += 1
        # First call: mol_0 zero-shot → accurate
        # Second call: mol_0 GNN-oracle rejected → return a response
        if call_count == 1:
            return accurate_response
        elif call_count == 2:
            return "Step 1: Some reasoning. Prediction: LogD ≈ -2.5"
        else:
            return inaccurate_response

    with patch("chemistry.dpo.generate_dpo_pairs._call_llm", side_effect=mock_llm):
        pairs = generate_dpo_pairs(df, gnn_preds)

    # mol_0 should produce: 1 GNN-oracle pair + 3 perturbation pairs = 4 total
    assert len(pairs) == 4, f"Expected 4 pairs, got {len(pairs)}"
    for pair in pairs:
        assert isinstance(pair, DPOPair)
        assert pair.prompt
        assert pair.chosen == accurate_response
        assert pair.rejected != accurate_response
