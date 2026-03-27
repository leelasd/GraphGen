"""Compute KG1 edges: MMP δLogD edges and FG co-occurrence edges."""
from __future__ import annotations
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)


def detect_functional_groups(smiles: str, fg_defs: list[dict]) -> list[str]:
    """Return list of FG names present in a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    present = []
    for fg in fg_defs:
        pattern = Chem.MolFromSmarts(fg["smarts"])
        if pattern and mol.HasSubstructMatch(pattern):
            present.append(fg["name"])
    return present


def _murcko_scaffold(smiles: str) -> str:
    """Return Murcko scaffold SMILES, or original SMILES if no scaffold."""
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(scaffold)
        return smi if smi else smiles
    except Exception:
        return smiles


def find_mmps(df: pd.DataFrame, fg_defs: list[dict]) -> list[dict[str, Any]]:
    """Find matched molecular pairs differing by exactly one FG."""
    smiles_col = "smiles"
    logd_col = "logd_exp"

    # Annotate each molecule with scaffold + present FGs
    records = []
    for _, row in df.iterrows():
        smi = row[smiles_col]
        fgs = frozenset(detect_functional_groups(smi, fg_defs))
        scaffold = _murcko_scaffold(smi)
        records.append({"smiles": smi, "logd": row[logd_col], "fgs": fgs, "scaffold": scaffold})

    mmps = []
    for r1, r2 in combinations(records, 2):
        diff = r1["fgs"].symmetric_difference(r2["fgs"])
        if len(diff) != 1:
            continue
        # Only pair molecules sharing the same scaffold (standard MMP constraint)
        if r1["scaffold"] != r2["scaffold"]:
            continue
        fg_name = next(iter(diff))
        delta = round(r1["logd"] - r2["logd"], 3)
        # Positive delta: r1 has the FG and is more lipophilic
        if fg_name in r1["fgs"]:
            mmps.append({"fg_name": fg_name, "smiles_with": r1["smiles"],
                          "smiles_without": r2["smiles"], "delta_logd": delta})
        else:
            mmps.append({"fg_name": fg_name, "smiles_with": r2["smiles"],
                          "smiles_without": r1["smiles"], "delta_logd": -delta})
    return mmps


def build_mmp_edges(df: pd.DataFrame, fg_defs: list[dict]) -> list[dict[str, Any]]:
    """Build MMP δLogD edges: FG_node → 'logd_effect' node."""
    mmps = find_mmps(df, fg_defs)
    # Aggregate: average δLogD per FG
    fg_deltas: dict[str, list[float]] = defaultdict(list)
    for mmp in mmps:
        fg_deltas[mmp["fg_name"]].append(mmp["delta_logd"])

    edges = []
    for fg_name, deltas in fg_deltas.items():
        avg_delta = round(sum(deltas) / len(deltas), 3)
        effect_node_id = f"logd_effect_{fg_name}"
        content = (
            f"Adding [{fg_name}] to a molecule changes LogD by {avg_delta:+.2f} units on average "
            f"(based on {len(deltas)} matched molecular pairs). "
            f"{'Increases' if avg_delta > 0 else 'Decreases'} lipophilicity."
        )
        edges.append({
            "source": fg_name,
            "target": effect_node_id,
            "edge_type": "mmp_delta",
            "delta_logd": avg_delta,
            "n_pairs": len(deltas),
            "content": content,
        })
    return edges


def build_cooccurrence_edges(df: pd.DataFrame, fg_defs: list[dict]) -> list[dict[str, Any]]:
    """Build co-occurrence edges between FGs that appear together in molecules."""
    cooccur: dict[tuple[str, str], int] = defaultdict(int)
    for _, row in df.iterrows():
        fgs = detect_functional_groups(row["smiles"], fg_defs)
        for fg_a, fg_b in combinations(sorted(fgs), 2):
            cooccur[(fg_a, fg_b)] += 1

    edges = []
    for (fg_a, fg_b), count in cooccur.items():
        # Include all co-occurrence pairs regardless of frequency
        content = (
            f"[{fg_a}] and [{fg_b}] co-occur in {count} molecule(s) in the dataset. "
            f"Their combined effect on LogD should be considered together."
        )
        edges.append({
            "source": fg_a,
            "target": fg_b,
            "edge_type": "co_occurrence",
            "count": count,
            "content": content,
        })
    return edges


if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO)
    fg_defs = yaml.safe_load(Path("chemistry/kg1_build/fg_smarts.yaml").read_text())["functional_groups"]
    df = pd.read_csv("test_logd.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    # Expect columns: smiles, logd_exp (or similar)
    mmp_edges = build_mmp_edges(df, fg_defs)
    cooc_edges = build_cooccurrence_edges(df, fg_defs)
    print(f"MMP edges: {len(mmp_edges)}, Co-occurrence edges: {len(cooc_edges)}")
