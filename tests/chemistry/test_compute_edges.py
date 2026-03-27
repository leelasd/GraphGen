import pytest
import pandas as pd

SAMPLE_MOLECULES = pd.DataFrame({
    "smiles": [
        "CC(=O)O",       # acetic acid: carboxylic_acid
        "CCCC",          # butane: no FG
        "c1ccccc1",      # benzene: aromatic_ring
        "c1ccccc1C(=O)O", # benzoic acid: aromatic_ring + carboxylic_acid
        "CCN",           # ethylamine: primary_amine
        "c1ccccc1N",     # aniline: aromatic_ring + primary_amine
    ],
    "logd_exp": [-0.17, 2.89, 1.56, 1.87, -0.13, 0.90],
})


def test_detect_fg_in_molecule():
    from chemistry.kg1_build.compute_edges import detect_functional_groups
    import yaml
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    fgs = detect_functional_groups("CC(=O)O", fg_defs)
    assert "carboxylic_acid" in fgs

    fgs_benzene = detect_functional_groups("c1ccccc1", fg_defs)
    assert "aromatic_ring" in fgs_benzene


def test_find_mmps():
    from chemistry.kg1_build.compute_edges import find_mmps
    import yaml
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    mmps = find_mmps(SAMPLE_MOLECULES, fg_defs)
    # benzoic acid vs benzene: differ by carboxylic_acid, δLogD = 1.87 - 1.56 = 0.31
    assert len(mmps) > 0
    fg_names = [m["fg_name"] for m in mmps]
    assert "carboxylic_acid" in fg_names


def test_build_cooccurrence_edges():
    from chemistry.kg1_build.compute_edges import build_cooccurrence_edges
    import yaml
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    edges = build_cooccurrence_edges(SAMPLE_MOLECULES, fg_defs)
    # benzoic acid has both aromatic_ring and carboxylic_acid → co-occurrence edge
    assert any(
        (e["source"] == "aromatic_ring" and e["target"] == "carboxylic_acid") or
        (e["source"] == "carboxylic_acid" and e["target"] == "aromatic_ring")
        for e in edges
    )


def test_build_mmp_edges():
    from chemistry.kg1_build.compute_edges import build_mmp_edges
    import yaml
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    edges = build_mmp_edges(SAMPLE_MOLECULES, fg_defs)
    assert len(edges) > 0
    for edge in edges:
        assert "source" in edge
        assert "target" in edge
        assert "delta_logd" in edge
        assert "edge_type" in edge
        assert edge["edge_type"] == "mmp_delta"
