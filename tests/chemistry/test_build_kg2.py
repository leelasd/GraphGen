import pytest
import pandas as pd
import networkx as nx
import tempfile
from pathlib import Path

SAMPLE_DF = pd.DataFrame({
    "smiles": [
        "CC(=O)O",             # acetic acid
        "c1ccccc1C(=O)O",      # benzoic acid
        "c1ccccc1",            # benzene
        "CCN",                 # ethylamine
        "c1ccccc1N",           # aniline
        "CC(C)N",              # isopropylamine
        "CCOCC",               # diethyl ether
        "c1ccc(N)cc1C(=O)O",   # 4-aminobenzoic acid
    ],
    "logd_exp": [-0.17, 1.87, 1.56, -0.13, 0.90, 0.26, 0.89, 0.83],
})


def test_build_molecule_nodes():
    from chemistry.kg2_build.build_kg2 import build_molecule_nodes
    import yaml
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    nodes = build_molecule_nodes(SAMPLE_DF, fg_defs)
    assert len(nodes) == len(SAMPLE_DF)
    for node in nodes:
        assert "content" in node
        assert "smiles" in node
        assert "logd_exp" in node
        assert "logd_bin" in node
        assert node["logd_bin"] in ("low", "mid", "high")


def test_build_similarity_edges():
    from chemistry.kg2_build.build_kg2 import build_similarity_edges
    nodes = [
        {"id": "mol_0", "smiles": "c1ccccc1N"},        # aniline
        {"id": "mol_1", "smiles": "c1ccccc1C(=O)O"},   # benzoic acid — Morgan sim=0.316
        {"id": "mol_2", "smiles": "CCN"},               # ethylamine — not similar
    ]
    edges = build_similarity_edges(nodes, threshold=0.3)
    # aniline and benzoic acid should be similar (Morgan r=2 Tanimoto=0.316)
    similar_pairs = [(e["source"], e["target"]) for e in edges]
    assert ("mol_0", "mol_1") in similar_pairs or ("mol_1", "mol_0") in similar_pairs


def test_build_kg2_graph():
    from chemistry.kg2_build.build_kg2 import build_kg2
    import yaml
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    G = build_kg2(SAMPLE_DF, fg_defs)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == len(SAMPLE_DF)
    assert G.number_of_edges() > 0
    for node_id, attrs in G.nodes(data=True):
        assert "content" in attrs


def test_kg2_exports_valid_graphml():
    from chemistry.kg2_build.build_kg2 import build_kg2, export_graphml
    import yaml
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    G = build_kg2(SAMPLE_DF, fg_defs)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_kg2.graphml"
        export_graphml(G, path)
        assert path.exists()
        G2 = nx.read_graphml(str(path))
        assert G2.number_of_nodes() == G.number_of_nodes()
