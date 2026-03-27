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
