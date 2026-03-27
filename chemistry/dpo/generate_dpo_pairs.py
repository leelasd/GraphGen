# chemistry/dpo/generate_dpo_pairs.py
"""Generate DPO preference pairs using GNN oracle + LiteLLM zero-shot."""
from __future__ import annotations
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

PREDICTION_PROMPT = (
    "Predict the LogD at pH 7.4 for the following molecule: {smiles}\n\n"
    "Provide a step-by-step analysis:\n"
    "Step 1 — Identify all functional groups in the SMILES.\n"
    "Step 2 — Estimate the lipophilic contribution of each group.\n"
    "Step 3 — Account for ionizable groups at pH 7.4.\n"
    "Step 4 — Sum contributions and state the final prediction.\n\n"
    "End your response with: 'Prediction: LogD ≈ X.X (range: Y.Y – Z.Z, confidence: low/medium/high)'"
)

WRONG_TARGET_PROMPT = (
    "Predict the LogD at pH 7.4 for the following molecule: {smiles}\n\n"
    "Note: A computational model predicts LogD = {gnn_pred:.2f} for this molecule. "
    "Provide a step-by-step reasoning that supports this prediction value.\n\n"
    "Step 1 — Identify all functional groups in the SMILES.\n"
    "Step 2 — Estimate contributions to justify LogD ≈ {gnn_pred:.2f}.\n"
    "Step 3 — Account for ionization.\n"
    "Step 4 — Conclude with LogD ≈ {gnn_pred:.2f}.\n\n"
    "End with: 'Prediction: LogD ≈ {gnn_pred:.2f}'"
)

PERTURBATION_TEMPLATES = {
    "ignore_ionization": (
        "Note: Ignore pH effects and ionization in your analysis. "
        "Treat all groups as neutral regardless of pH."
    ),
    "wrong_ph": (
        "Note: Predict LogD at pH 2.0 instead of pH 7.4."
    ),
    "miss_fg": (
        "Note: Focus only on the carbon skeleton. Ignore any heteroatom functional groups."
    ),
}


@dataclass
class DPOPair:
    prompt: str
    chosen: str
    rejected: str

    def to_dict(self) -> dict:
        return asdict(self)


def extract_logd_prediction(response: str) -> Optional[float]:
    """Extract numeric LogD value from model response."""
    patterns = [
        r"LogD\s*[≈=~]\s*(-?\d+\.?\d*)",
        r"Prediction:\s*-?\d+\.?\d*\s*[≈=~]\s*(-?\d+\.?\d*)",
        r"predicted\s+LogD.*?(-?\d+\.?\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return None


def perturb_reasoning(correct_chain: str, perturbation: str) -> str:
    """Generate a flawed reasoning chain by applying a perturbation."""
    note = PERTURBATION_TEMPLATES.get(perturbation, "")
    lines = correct_chain.split("\n")
    # Insert perturbation note after Step 1
    output = []
    for line in lines:
        output.append(line)
        if line.startswith("Step 1"):
            output.append(f"[PERTURBATION: {note}]")
    # Corrupt the final prediction by inverting sign or adding offset
    result = "\n".join(output)
    result = re.sub(
        r"Prediction: LogD ≈ (-?\d+\.?\d*)",
        lambda m: f"Prediction: LogD ≈ {-float(m.group(1)):.1f}",
        result,
    )
    return result


def _call_llm(prompt: str, model: str = "llama-3-2-3b") -> str:
    """Call LiteLLM proxy at localhost:4000 (routes to AWS Bedrock — no OpenAI models)."""
    import httpx
    base_url = os.getenv("SYNTHESIZER_BASE_URL", "http://localhost:4000").rstrip("/")
    api_key = os.getenv("SYNTHESIZER_API_KEY", "your-master-key-here")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = httpx.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""


def generate_dpo_pairs(
    df: pd.DataFrame,
    gnn_predictions: pd.Series,
    chosen_threshold: float = 0.5,
    gnn_diverge_threshold: float = 1.0,
    model: str = "llama-3-2-3b",
) -> list[DPOPair]:
    """
    Generate DPO pairs for each molecule.

    Args:
        df: DataFrame with 'smiles' and 'logd_exp' columns
        gnn_predictions: Series of GNN-predicted LogD values (same index as df)
        chosen_threshold: |LLM - exp| < this → chosen
        gnn_diverge_threshold: |GNN - exp| > this → use GNN as rejected target
    """
    pairs = []
    prompt_template = PREDICTION_PROMPT

    for idx, row in df.iterrows():
        smiles = row["smiles"]
        logd_exp = float(row["logd_exp"])
        gnn_pred = float(gnn_predictions[idx])

        # Generate LLM response (zero-shot)
        prompt = prompt_template.format(smiles=smiles)
        try:
            llm_response = _call_llm(prompt, model=model)
        except Exception as exc:
            logger.warning("LLM call failed for %s: %s", smiles, exc)
            continue

        llm_pred = extract_logd_prediction(llm_response)
        if llm_pred is None:
            logger.warning("Could not extract prediction from LLM response for %s", smiles)
            continue

        llm_error = abs(llm_pred - logd_exp)
        gnn_error = abs(gnn_pred - logd_exp)

        # Case 1: LLM is accurate → chosen; GNN diverges → generate rejected
        if llm_error < chosen_threshold and gnn_error > gnn_diverge_threshold:
            rejected_prompt = WRONG_TARGET_PROMPT.format(smiles=smiles, gnn_pred=gnn_pred)
            try:
                rejected_response = _call_llm(rejected_prompt, model=model)
            except Exception as exc:
                logger.warning("Rejected generation failed for %s: %s", smiles, exc)
                continue
            pairs.append(DPOPair(
                prompt=prompt,
                chosen=llm_response,
                rejected=rejected_response,
            ))

        # Case 2: LLM is accurate → chosen; also generate perturbation rejected
        if llm_error < chosen_threshold:
            for perturbation in ["ignore_ionization", "wrong_ph", "miss_fg"]:
                rejected = perturb_reasoning(llm_response, perturbation)
                pairs.append(DPOPair(
                    prompt=prompt,
                    chosen=llm_response,
                    rejected=rejected,
                ))
            break  # one perturbation set per molecule is enough for POC

    logger.info("Generated %d DPO pairs", len(pairs))
    return pairs


def save_dpo_pairs(pairs: list[DPOPair], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict()) + "\n")
    logger.info("Saved %d DPO pairs to %s", len(pairs), output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv("test_logd.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    logd_col = next((c for c in df.columns if "logd" in c.lower()), None)
    smiles_col = next((c for c in df.columns if "smile" in c.lower()), "smiles")
    df = df.rename(columns={logd_col: "logd_exp", smiles_col: "smiles"})
    df = df.dropna(subset=["smiles", "logd_exp"])

    # Load GNN predictions from CSV.
    # You must supply this file from your existing GNN pipeline.
    # Format: two columns — 'smiles' (matching the test set) and 'gnn_pred' (numeric LogD predictions).
    # Example: run your ADMET GNN on test_logd.csv and save predictions as chemistry/dpo/gnn_predictions.csv
    gnn_csv = "chemistry/dpo/gnn_predictions.csv"
    if Path(gnn_csv).exists():
        gnn_df = pd.read_csv(gnn_csv).set_index("smiles")
        gnn_preds = df["smiles"].map(gnn_df["gnn_pred"])
    else:
        logger.warning(
            "GNN predictions not found at %s. "
            "Using random offsets as stand-in — replace with real GNN output before running DPO.",
            gnn_csv
        )
        import numpy as np
        rng = np.random.default_rng(42)
        gnn_preds = pd.Series(
            df["logd_exp"].values + rng.normal(0, 1.5, len(df)),
            index=df.index
        )

    pairs = generate_dpo_pairs(df, gnn_preds)
    save_dpo_pairs(pairs, Path("chemistry/dpo/dpo_pairs.jsonl"))
    print(f"Generated {len(pairs)} DPO pairs")
