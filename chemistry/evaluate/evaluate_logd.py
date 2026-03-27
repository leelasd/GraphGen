# chemistry/evaluate/evaluate_logd.py
"""Evaluate fine-tuned model LogD predictions: Spearman R, RMSE, FG accuracy."""
from __future__ import annotations
import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy.stats import spearmanr

from chemistry.kg1_build.compute_edges import detect_functional_groups

logger = logging.getLogger(__name__)

GNN_BASELINE = {"spearman_r": 0.76, "rmse": 2.1}


def extract_predictions(responses: list[str]) -> list[Optional[float]]:
    """Extract LogD numeric predictions from model response strings."""
    patterns = [
        r"Prediction:\s*LogD\s*[≈=~]\s*(-?\d+\.?\d*)",
        r"LogD\s*[≈=~]\s*(-?\d+\.?\d*)",
        r"(?:predicted|estimate[d]?)\s+LogD.*?(-?\d+\.?\d+)",
        r"approximately\s+(-?\d+\.?\d+)",
    ]
    results = []
    for response in responses:
        found = None
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    found = float(match.group(1))
                    break
                except ValueError:
                    continue
        results.append(found)
    return results


def compute_metrics(
    experimental: list[float],
    predicted: list[float],
) -> dict:
    """Compute Spearman R, RMSE, MAE between experimental and predicted LogD."""
    pairs = [(e, p) for e, p in zip(experimental, predicted) if p is not None]
    if len(pairs) < 2:
        return {"spearman_r": None, "rmse": None, "mae": None, "n_evaluated": len(pairs)}

    exp_vals = [p[0] for p in pairs]
    pred_vals = [p[1] for p in pairs]

    r, _ = spearmanr(exp_vals, pred_vals)
    rmse = math.sqrt(sum((e - p) ** 2 for e, p in zip(exp_vals, pred_vals)) / len(pairs))
    mae = sum(abs(e - p) for e, p in zip(exp_vals, pred_vals)) / len(pairs)

    return {
        "spearman_r": round(r, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "n_evaluated": len(pairs),
        "n_total": len(experimental),
        "coverage": round(len(pairs) / len(experimental), 3),
    }


def compute_fg_accuracy(
    responses: list[str],
    smiles_list: list[str],
    fg_defs: list[dict],
) -> dict:
    """Measure how often the model correctly identifies FGs present in the molecule."""
    total_fgs = 0
    correctly_identified = 0

    for response, smiles in zip(responses, smiles_list):
        true_fgs = set(detect_functional_groups(smiles, fg_defs))
        if not true_fgs:
            continue
        for fg_name in true_fgs:
            total_fgs += 1
            # Check if FG name or its common synonym appears in the response
            if fg_name.replace("_", " ") in response.lower() or fg_name in response.lower():
                correctly_identified += 1

    recall = correctly_identified / total_fgs if total_fgs > 0 else 0.0
    return {
        "fg_recall": round(recall, 3),
        "correctly_identified": correctly_identified,
        "total_fgs": total_fgs,
    }


def generate_report(
    metrics: dict,
    gnn_baseline: dict | None = None,
) -> str:
    """Generate a human-readable evaluation report."""
    gnn = gnn_baseline or GNN_BASELINE
    spearman_pass = (metrics.get("spearman_r") or 0) >= 0.75
    rmse_pass = (metrics.get("rmse") or 999) <= 2.0
    overall = "PASS" if spearman_pass and rmse_pass else "FAIL"

    lines = [
        "=" * 55,
        "  LogD Prediction Evaluation Report",
        "=" * 55,
        f"  Molecules evaluated : {metrics.get('n_evaluated', 'N/A')} / {metrics.get('n_total', 'N/A')}",
        f"  Coverage            : {metrics.get('coverage', 'N/A')}",
        "",
        f"  spearman_r          : {metrics.get('spearman_r', 'N/A'):<8}  (GNN baseline: {gnn['spearman_r']})  {'✓' if spearman_pass else '✗'}",
        f"  RMSE (log units)    : {metrics.get('rmse', 'N/A'):<8}  (GNN baseline: {gnn['rmse']})  {'✓' if rmse_pass else '✗'}",
        f"  MAE  (log units)    : {metrics.get('mae', 'N/A')}",
        "",
        f"  POC Target          : [{overall}]",
        "=" * 55,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import os, yaml
    logging.basicConfig(level=logging.INFO)

    # Load test set
    df = pd.read_csv("test_logd.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    logd_col = next((c for c in df.columns if "logd" in c.lower()), None)
    smiles_col = next((c for c in df.columns if "smile" in c.lower()), "smiles")
    df = df.rename(columns={logd_col: "logd_exp", smiles_col: "smiles"}).dropna()

    # Load model predictions (expected: JSONL with 'smiles' and 'response' fields)
    predictions_file = Path("chemistry/evaluate/model_predictions.jsonl")
    if not predictions_file.exists():
        print(f"No predictions file found at {predictions_file}")
        print("Run model inference first and save responses to chemistry/evaluate/model_predictions.jsonl")
        print("Format: one JSON per line with keys: smiles, response")
        exit(1)

    records = [json.loads(l) for l in predictions_file.read_text().strip().split("\n")]
    response_map = {r["smiles"]: r["response"] for r in records}
    responses = [response_map.get(smi, "") for smi in df["smiles"]]

    preds = extract_predictions(responses)
    metrics = compute_metrics(df["logd_exp"].tolist(), preds)

    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    valid_responses = [r for r in responses if r]
    valid_smiles = [df["smiles"].iloc[i] for i, r in enumerate(responses) if r]
    fg_metrics = compute_fg_accuracy(valid_responses, valid_smiles, fg_defs)

    print(generate_report(metrics))
    print(f"\nFG Recall: {fg_metrics['fg_recall']} ({fg_metrics['correctly_identified']}/{fg_metrics['total_fgs']} FGs identified)")
