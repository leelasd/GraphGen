"""Build KG2: molecule graph from ChEMBL experimental data."""
from __future__ import annotations
import logging
from pathlib import Path

import networkx as nx
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from chemistry.kg1_build.compute_edges import detect_functional_groups

logger = logging.getLogger(__name__)


def _logd_bin(logd: float) -> str:
    if logd < 1.0:
        return "low"
    if logd <= 3.0:
        return "mid"
    return "high"


def _compute_mol_descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Invalid SMILES in descriptor computation: %s", smiles)
        return {"logp": 0.0, "hbd": 0, "hba": 0, "tpsa": 0.0, "mw": 0.0, "rotbonds": 0}
    return {
        "logp": round(Crippen.MolLogP(mol), 3),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "mw": round(Descriptors.ExactMolWt(mol), 2),
        "rotbonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }


def _murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""


def build_molecule_nodes(df: pd.DataFrame, fg_defs: list[dict]) -> list[dict]:
    """Build one KG2 node per molecule."""
    nodes = []
    for idx, row in df.iterrows():
        smiles = row["smiles"]
        logd = float(row["logd_exp"])
        fgs = detect_functional_groups(smiles, fg_defs)
        scaffold = _murcko_scaffold(smiles)
        desc = _compute_mol_descriptors(smiles)
        logd_bin = _logd_bin(logd)
        fg_str = ", ".join(fgs) if fgs else "none identified"

        content = (
            f"Molecule SMILES: {smiles}. "
            f"Experimental LogD at pH 7.4: {logd:.2f}. "
            f"LogD category: {logd_bin} ({'< 1.0' if logd_bin == 'low' else '1.0-3.0' if logd_bin == 'mid' else '> 3.0'}). "
            f"Murcko scaffold: {scaffold or 'none'}. "
            f"Functional groups present: {fg_str}. "
            f"Descriptors: MW = {desc['mw']}, LogP = {desc['logp']}, "
            f"HBD = {desc['hbd']}, HBA = {desc['hba']}, "
            f"TPSA = {desc['tpsa']} Angstrom^2, RotBonds = {desc['rotbonds']}."
        )

        nodes.append({
            "id": f"mol_{idx}",
            "smiles": smiles,
            "logd_exp": str(logd),
            "logd_bin": logd_bin,
            "scaffold": scaffold,
            "functional_groups": fg_str,
            "content": content,
            **{k: str(v) for k, v in desc.items()},
        })
    return nodes


def build_similarity_edges(nodes: list[dict], threshold: float = 0.6) -> list[dict]:
    """Build Tanimoto similarity edges between molecules."""
    fps = []
    valid_nodes = []
    for node in nodes:
        mol = Chem.MolFromSmiles(node["smiles"])
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
            valid_nodes.append(node)

    edges = []
    for i in range(len(valid_nodes)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:])
        for j, sim in enumerate(sims, start=i+1):
            if sim >= threshold:
                ni, nj = valid_nodes[i], valid_nodes[j]
                ni_logd = ni.get("logd_exp", "N/A")
                nj_logd = nj.get("logd_exp", "N/A")
                content = (
                    f"Molecules [{ni['smiles']}] and [{nj['smiles']}] are structurally similar "
                    f"(Tanimoto = {sim:.2f}). "
                    f"LogD values: {ni_logd} vs {nj_logd}."
                )
                edges.append({
                    "source": ni["id"], "target": nj["id"],
                    "edge_type": "tanimoto_similarity",
                    "tanimoto": str(round(sim, 3)),
                    "content": content,
                })
    return edges


def build_scaffold_edges(nodes: list[dict]) -> list[dict]:
    """Build edges between molecules sharing a Murcko scaffold."""
    from collections import defaultdict
    scaffold_groups: dict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        if node.get("scaffold"):
            scaffold_groups[node["scaffold"]].append(node)

    edges = []
    for scaffold, group in scaffold_groups.items():
        if len(group) < 2:
            continue
        # Window cap: connect each molecule to its next 5 neighbors only.
        # Prevents O(N^2) explosion for common scaffolds (e.g., bare benzene ring)
        # on real ChEMBL data where hundreds of molecules share the same Murcko scaffold.
        for i in range(len(group)):
            for j in range(i + 1, min(i + 6, len(group))):
                ni, nj = group[i], group[j]
                content = (
                    f"Molecules [{ni['smiles']}] and [{nj['smiles']}] share scaffold: {scaffold}. "
                    f"LogD values: {ni['logd_exp']} vs {nj['logd_exp']}. "
                    f"Structural differences can be attributed to substituent changes."
                )
                edges.append({
                    "source": ni["id"], "target": nj["id"],
                    "edge_type": "shared_scaffold",
                    "scaffold": scaffold,
                    "content": content,
                })
    return edges


def build_fg_class_edges(nodes: list[dict]) -> list[dict]:
    """Build edges between molecules sharing a functional group class."""
    from collections import defaultdict
    fg_groups: dict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        fgs = [f.strip() for f in node.get("functional_groups", "").split(",")
               if f.strip() and f.strip() != "none identified"]
        for fg in fgs:
            fg_groups[fg].append(node)

    edges = []
    for fg, group in fg_groups.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, min(i + 6, len(group))):
                ni, nj = group[i], group[j]
                if ni["id"] == nj["id"]:
                    continue
                content = (
                    f"Both [{ni['smiles']}] and [{nj['smiles']}] contain [{fg}]. "
                    f"LogD: {ni['logd_exp']} vs {nj['logd_exp']}. "
                    f"Compare how other structural differences modulate LogD beyond the shared {fg} group."
                )
                edges.append({
                    "source": ni["id"], "target": nj["id"],
                    "edge_type": "shared_fg_class",
                    "shared_fg": fg,
                    "content": content,
                })
    return edges


def build_kg2(df: pd.DataFrame, fg_defs: list[dict], similarity_threshold: float = 0.6) -> nx.Graph:
    """Build full KG2 molecule graph."""
    G = nx.Graph()

    nodes = build_molecule_nodes(df, fg_defs)
    nodes_with_id = []
    for node in nodes:
        node_id = node["id"]
        node_attrs = {k: v for k, v in node.items() if k != "id"}
        G.add_node(node_id, **node_attrs)
        nodes_with_id.append(node)

    for edge in build_similarity_edges(nodes_with_id, threshold=similarity_threshold):
        src, tgt = edge.pop("source"), edge.pop("target")
        G.add_edge(src, tgt, **edge)

    for edge in build_scaffold_edges(nodes_with_id):
        src, tgt = edge.pop("source"), edge.pop("target")
        if not G.has_edge(src, tgt):
            # First-writer-wins: similarity edges take priority; scaffold edges fill gaps.
            G.add_edge(src, tgt, **edge)

    for edge in build_fg_class_edges(nodes_with_id):
        src, tgt = edge.pop("source"), edge.pop("target")
        if not G.has_edge(src, tgt):
            # First-writer-wins: only add FG class edge if no edge exists between this pair yet.
            G.add_edge(src, tgt, **edge)

    logger.info("KG2 built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def export_graphml(G: nx.Graph, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(output_path))
    logger.info("KG2 exported to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fg_defs = yaml.safe_load(Path("chemistry/kg1_build/fg_smarts.yaml").read_text())["functional_groups"]
    df = pd.read_csv("test_logd.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    logd_col = next((c for c in df.columns if "logd" in c.lower()), None)
    if logd_col and logd_col != "logd_exp":
        df = df.rename(columns={logd_col: "logd_exp"})
    smiles_col = next((c for c in df.columns if "smile" in c.lower()), "smiles")
    if smiles_col != "smiles":
        df = df.rename(columns={smiles_col: "smiles"})
    df = df.dropna(subset=["smiles", "logd_exp"])
    df = df.head(1000)  # POC cap: ~1K molecules per spec
    G = build_kg2(fg_defs=fg_defs, df=df)
    export_graphml(G, Path("chemistry/kg2_build/molecule_graph.graphml"))
    print(f"KG2: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
