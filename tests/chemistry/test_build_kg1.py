import pytest
import networkx as nx
from pathlib import Path
import tempfile
import yaml
import pandas as pd


SAMPLE_DF = pd.DataFrame({
    "smiles": ["CC(=O)O", "c1ccccc1", "c1ccccc1C(=O)O", "CCN", "c1ccccc1N", "CCCC"],
    "logd_exp": [-0.17, 1.56, 1.87, -0.13, 0.90, 2.89],
})


def test_build_kg1_returns_graph():
    from chemistry.kg1_build.build_kg1 import build_kg1
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    G = build_kg1(fg_defs, SAMPLE_DF)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() > 0


def test_kg1_nodes_have_content():
    from chemistry.kg1_build.build_kg1 import build_kg1
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    G = build_kg1(fg_defs, SAMPLE_DF)
    for node_id, attrs in G.nodes(data=True):
        assert "content" in attrs, f"Node {node_id} missing content"
        assert len(attrs["content"]) > 20


def test_kg1_exports_valid_graphml():
    from chemistry.kg1_build.build_kg1 import build_kg1, export_graphml
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    G = build_kg1(fg_defs, SAMPLE_DF)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_kg1.graphml"
        export_graphml(G, path)
        assert path.exists()
        G2 = nx.read_graphml(str(path))
        assert G2.number_of_nodes() == G.number_of_nodes()


def test_kg1_has_effect_nodes():
    from chemistry.kg1_build.build_kg1 import build_kg1
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    G = build_kg1(fg_defs, SAMPLE_DF)
    effect_nodes = [n for n in G.nodes if str(n).startswith("logd_effect_")]
    assert len(effect_nodes) > 0
