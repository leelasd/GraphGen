import pytest
import yaml
from pathlib import Path

def test_fg_smarts_file_exists():
    path = Path("chemistry/kg1_build/fg_smarts.yaml")
    assert path.exists(), "fg_smarts.yaml not found"

def test_fg_smarts_valid_rdkit():
    from rdkit import Chem
    path = Path("chemistry/kg1_build/fg_smarts.yaml")
    data = yaml.safe_load(path.read_text())
    for fg in data["functional_groups"]:
        mol = Chem.MolFromSmarts(fg["smarts"])
        assert mol is not None, f"Invalid SMARTS for {fg['name']}: {fg['smarts']}"

def test_fg_smarts_required_keys():
    path = Path("chemistry/kg1_build/fg_smarts.yaml")
    data = yaml.safe_load(path.read_text())
    for fg in data["functional_groups"]:
        assert "name" in fg
        assert "smarts" in fg
        assert "category" in fg
    assert len(data["functional_groups"]) >= 30


def test_compute_fg_rdkit_attributes():
    from chemistry.kg1_build.compute_nodes import compute_rdkit_attributes
    # Carboxylic acid probe: acetic acid
    attrs = compute_rdkit_attributes("carboxylic_acid", "[CX3](=O)[OX2H1]", "CC(=O)O")
    assert "logp_contribution" in attrs
    assert "hbd" in attrs
    assert "hba" in attrs
    assert "tpsa" in attrs
    assert "mw" in attrs
    assert isinstance(attrs["logp_contribution"], float)
    assert attrs["hbd"] >= 1   # COOH has 1 HBD
    assert attrs["hba"] >= 1   # COOH has HBA


def _openbabel_available():
    try:
        from openbabel import pybel  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _openbabel_available(), reason="OpenBabel not installed")
def test_compute_fg_openbabel_ionization():
    from chemistry.kg1_build.compute_nodes import compute_openbabel_ionization
    # Carboxylic acid: ionized at pH 7.4
    result = compute_openbabel_ionization("[CX3](=O)[OX2H1]", "CC(=O)O")
    assert "ionized_at_ph74" in result
    assert "formal_charge_ph74" in result
    assert result["ionized_at_ph74"] is True
    assert result["formal_charge_ph74"] < 0


def test_build_fg_node():
    from chemistry.kg1_build.compute_nodes import build_fg_node
    node = build_fg_node(
        {"name": "carboxylic_acid", "smarts": "[CX3](=O)[OX2H1]", "category": "acidic"},
        probe_smiles="CC(=O)O"
    )
    assert node["id"] == "carboxylic_acid"
    assert "content" in node
    assert "logp_contribution" in node
    assert "ionized_at_ph74" in node
    assert "carboxylic_acid" in node["content"].lower()


def test_build_all_fg_nodes():
    from chemistry.kg1_build.compute_nodes import build_all_fg_nodes
    import yaml
    data = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))
    nodes = build_all_fg_nodes(data["functional_groups"])
    assert len(nodes) >= 30
    for node in nodes:
        assert "id" in node
        assert "content" in node
        attrs_nonzero = (
            node.get("logp_contribution", 0.0) != 0.0
            or node.get("hbd", 0) > 0
            or node.get("hba", 0) > 0
            or node.get("tpsa", 0.0) > 0.0
        )
        assert attrs_nonzero, f"All-zero RDKit attributes for node: {node['id']}"
