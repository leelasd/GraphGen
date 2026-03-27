# Chemistry Reasoning Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two RDKit/OpenBabel-derived knowledge graphs (KG1: FG rules, KG2: molecules), run them through GraphGen to produce ~9K chemistry training examples (SFT + DPO), and evaluate fine-tuned Llama 3.2 3B against GNN baselines on LogD prediction.

**Architecture:** KG1 is a functional-group rule graph with computationally-derived node attributes (Crippen LogP, HBD/HBA, TPSA, OpenBabel ionization) and MMP-derived δLogD edges. KG2 is a molecule graph from ChEMBL with Tanimoto/scaffold/FG-class edges. Both KGs are exported as GraphML and fed into the new Ray-based GraphGen pipeline via `GraphmlReader`.

**Tech Stack:** RDKit, OpenBabel (pybel), NetworkX, pandas, scipy, LiteLLM proxy (localhost:4000), GraphGen (Ray-based engine), pytest

**Scope:** This plan covers **data generation only** (KG build → GraphGen → DPO pairs → evaluation script). Model fine-tuning (QLoRA SFT + DPO training) is out of scope and handled separately once the dataset is validated.

**Environment variables required before running GraphGen** (synthesizer/trainee are configured via env, not YAML):

GraphGen has no native Bedrock backend. The routing is:
`GraphGen → http_api backend → LiteLLM proxy (localhost:4000) → AWS Bedrock`

Model names below are LiteLLM aliases defined in `litellm_config.yaml` that map to Bedrock model ARNs.
No OpenAI models are used.

```bash
# Synthesizer: Claude Sonnet 4 on Bedrock (via LiteLLM proxy)
export SYNTHESIZER_BACKEND=http_api
export SYNTHESIZER_BASE_URL=http://localhost:4000
export SYNTHESIZER_MODEL=claude-sonnet-4
export SYNTHESIZER_API_KEY=your-master-key-here

# Trainee: Llama 3.2 3B on Bedrock (via LiteLLM proxy)
export TRAINEE_BACKEND=http_api
export TRAINEE_BASE_URL=http://localhost:4000
export TRAINEE_MODEL=llama-3-2-3b
export TRAINEE_API_KEY=your-master-key-here
```

---

## File Map

| File | Purpose |
|------|---------|
| `chemistry/kg1_build/fg_smarts.yaml` | SMARTS patterns for ~40 medicinal chemistry FGs |
| `chemistry/kg1_build/compute_nodes.py` | RDKit: compute LogP/HBD/HBA/TPSA/MW per FG; OpenBabel: ionization at pH 7.4 |
| `chemistry/kg1_build/compute_edges.py` | MMP δLogD edges + FG co-occurrence edges from ChEMBL |
| `chemistry/kg1_build/build_kg1.py` | Assemble nodes + edges → NetworkX → GraphML |
| `chemistry/kg2_build/build_kg2.py` | ChEMBL CSV → molecule nodes + Tanimoto/scaffold/FG edges → GraphML |
| `chemistry/configs/chemistry_atomic_config.yaml` | GraphGen: KG1 → atomic Alpaca SFT-1 |
| `chemistry/configs/chemistry_cot_config.yaml` | GraphGen: KG1 → CoT ChatML SFT-2 |
| `chemistry/configs/chemistry_multihop_config.yaml` | GraphGen: KG2 → multi-hop ChatML SFT-3 |
| `chemistry/configs/chemistry_aggregated_config.yaml` | GraphGen: KG2 → aggregated ChatML SFT-2 anchor |
| `chemistry/dpo/generate_dpo_pairs.py` | GNN oracle + LiteLLM → chosen/rejected pairs |
| `chemistry/evaluate/evaluate_logd.py` | Spearman R, RMSE, FG accuracy from model output |
| `tests/chemistry/test_compute_nodes.py` | Unit tests for node attribute computation |
| `tests/chemistry/test_compute_edges.py` | Unit tests for MMP detection and edge building |
| `tests/chemistry/test_build_kg1.py` | Integration test: KG1 GraphML structure |
| `tests/chemistry/test_build_kg2.py` | Integration test: KG2 GraphML structure |
| `tests/chemistry/test_generate_dpo_pairs.py` | Unit tests for DPO pair format |
| `tests/chemistry/test_evaluate_logd.py` | Unit tests for metric computation |

---

## Task 1: Project Setup + FG SMARTS Definitions

**Files:**
- Create: `chemistry/kg1_build/fg_smarts.yaml`
- Create: `chemistry/__init__.py`, `chemistry/kg1_build/__init__.py`, `chemistry/kg2_build/__init__.py`, `chemistry/configs/__init__.py`, `chemistry/dpo/__init__.py`, `chemistry/evaluate/__init__.py`
- Create: `tests/chemistry/__init__.py`
- Modify: `requirements-chemistry.txt`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p chemistry/kg1_build chemistry/kg2_build chemistry/configs chemistry/dpo chemistry/evaluate
mkdir -p tests/chemistry
touch chemistry/__init__.py chemistry/kg1_build/__init__.py chemistry/kg2_build/__init__.py
touch chemistry/configs/__init__.py chemistry/dpo/__init__.py chemistry/evaluate/__init__.py
touch tests/chemistry/__init__.py
```

- [ ] **Step 2: Update requirements-chemistry.txt**

```
rdkit>=2023.9.1
openbabel-wheel>=3.1.1
networkx>=3.0
pandas>=2.0
scipy>=1.11
numpy>=1.24
httpx>=0.27
```

- [ ] **Step 3: Write the failing test**

```python
# tests/chemistry/test_compute_nodes.py
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
```

- [ ] **Step 4: Run test to verify it fails**

```bash
cd /Users/ldodda/Documents/Codes/GraphGen
python -m pytest tests/chemistry/test_compute_nodes.py::test_fg_smarts_file_exists -v
```
Expected: FAIL — `AssertionError: fg_smarts.yaml not found`

- [ ] **Step 5: Create fg_smarts.yaml**

```yaml
# chemistry/kg1_build/fg_smarts.yaml
functional_groups:
  # --- Acidic groups ---
  - name: carboxylic_acid
    smarts: "[CX3](=O)[OX2H1]"
    category: acidic
  - name: sulfonamide
    smarts: "[#16X4](=[OX1])(=[OX1])[NX3]"
    category: acidic
  - name: tetrazole
    smarts: "c1nn[nH]n1"
    category: acidic
  - name: phosphoric_acid
    smarts: "P(=O)(O)O"
    category: acidic

  # --- Basic groups ---
  - name: primary_amine
    smarts: "[NX3;H2;!$(NC=O)]"
    category: basic
  - name: secondary_amine
    smarts: "[NX3;H1;!$(NC=O);!$(N~[!#6])]"
    category: basic
  - name: tertiary_amine
    smarts: "[NX3;H0;!$(NC=O);!$(N~[!#6])]"
    category: basic
  - name: piperazine
    smarts: "N1CCNCC1"
    category: basic
  - name: morpholine
    smarts: "N1CCOCC1"
    category: basic
  - name: piperidine
    smarts: "N1CCCCC1"
    category: basic
  - name: pyrrolidine
    smarts: "N1CCCC1"
    category: basic
  - name: pyridine
    smarts: "n1ccccc1"
    category: basic
  - name: imidazole
    smarts: "c1cnc[nH]1"
    category: basic
  - name: guanidine
    smarts: "[NX3][CX3](=[NX2])[NX3]"
    category: basic

  # --- Neutral polar groups ---
  - name: amide
    smarts: "[NX3][CX3](=[OX1])"
    category: neutral_polar
  - name: hydroxyl
    smarts: "[OX2H;!$(OC=O)]"
    category: neutral_polar
  - name: ether
    smarts: "[OD2]([#6])[#6]"
    category: neutral_polar
  - name: ester
    smarts: "[#6][CX3](=O)[OX2H0][#6]"
    category: neutral_polar
  - name: ketone
    smarts: "[#6][CX3](=O)[#6]"
    category: neutral_polar
  - name: aldehyde
    smarts: "[CX3H1](=O)"
    category: neutral_polar
  - name: nitrile
    smarts: "[NX1]#[CX2]"
    category: neutral_polar
  - name: urea
    smarts: "[NX3][CX3](=[OX1])[NX3]"
    category: neutral_polar

  # --- Lipophilic groups ---
  - name: aromatic_ring
    smarts: "c1ccccc1"
    category: lipophilic
  - name: trifluoromethyl
    smarts: "[CX4](F)(F)F"
    category: lipophilic
  - name: cyclopropyl
    smarts: "[C@@H]1CC1"
    category: lipophilic
  - name: indole
    smarts: "c1ccc2[nH]ccc2c1"
    category: lipophilic
  - name: thiophene
    smarts: "c1ccsc1"
    category: lipophilic

  # --- Halogen groups ---
  - name: fluorine
    smarts: "[F;$(F-[#6])]"
    category: halogen
  - name: chlorine
    smarts: "[Cl;$(Cl-[#6])]"
    category: halogen
  - name: bromine
    smarts: "[Br;$(Br-[#6])]"
    category: halogen

  # --- Electron-withdrawing groups ---
  - name: nitro
    smarts: "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]"
    category: ewg
  - name: trifluoromethyl_ewg
    smarts: "cC(F)(F)F"
    category: ewg

  # --- Sulfur groups ---
  - name: thiol
    smarts: "[SX2H]"
    category: sulfur
  - name: thioether
    smarts: "[SX2]([#6])[#6]"
    category: sulfur
  - name: sulfoxide
    smarts: "[#16X3](=[OX1])"
    category: sulfur
  - name: sulfone
    smarts: "[#16X4](=[OX1])(=[OX1])"
    category: sulfur
```

- [ ] **Step 6: Run test to verify it passes**

```bash
python -m pytest tests/chemistry/test_compute_nodes.py -v
```
Expected: PASS — 3 tests green

- [ ] **Step 7: Commit**

```bash
git add chemistry/ tests/chemistry/ requirements-chemistry.txt
git commit -m "feat: add chemistry directory structure and FG SMARTS definitions"
```

---

## Task 2: KG1 Node Computation (RDKit Descriptors + OpenBabel Ionization)

**Files:**
- Create: `chemistry/kg1_build/compute_nodes.py`
- Modify: `tests/chemistry/test_compute_nodes.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/chemistry/test_compute_nodes.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/chemistry/test_compute_nodes.py::test_compute_fg_rdkit_attributes -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'chemistry.kg1_build.compute_nodes'`

- [ ] **Step 3: Implement compute_nodes.py**

```python
# chemistry/kg1_build/compute_nodes.py
"""Compute RDKit and OpenBabel physicochemical attributes for each functional group node."""
from __future__ import annotations
import logging
from typing import Any

import yaml
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

logger = logging.getLogger(__name__)

# Probe molecules for each FG category — used when ChEMBL data unavailable
_CATEGORY_PROBES = {
    "acidic": "CC(=O)O",        # acetic acid
    "basic": "CCN",              # ethylamine
    "neutral_polar": "CCO",      # ethanol
    "lipophilic": "c1ccccc1",    # benzene
    "halogen": "CCF",            # fluoroethane
    "ewg": "CC(=O)C",           # acetone
    "sulfur": "CCS",             # ethanethiol
}


def compute_rdkit_attributes(fg_name: str, smarts: str, probe_smiles: str) -> dict[str, Any]:
    """Compute LogP contribution, HBD, HBA, TPSA, MW for a FG using a probe molecule."""
    probe = Chem.MolFromSmiles(probe_smiles)
    if probe is None:
        logger.warning("Invalid probe SMILES %s for %s", probe_smiles, fg_name)
        return {"logp_contribution": 0.0, "hbd": 0, "hba": 0, "tpsa": 0.0, "mw": 0.0}

    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None or not probe.HasSubstructMatch(pattern):
        logger.warning("SMARTS %s not found in probe %s", smarts, probe_smiles)
        return {"logp_contribution": 0.0, "hbd": 0, "hba": 0, "tpsa": 0.0, "mw": 0.0}

    # Crippen LogP contribution: delta between probe and methane (simplest reference)
    ref = Chem.MolFromSmiles("C")
    probe_logp = Crippen.MolLogP(probe)
    ref_logp = Crippen.MolLogP(ref)
    logp_contrib = round(probe_logp - ref_logp, 3)

    return {
        "logp_contribution": logp_contrib,
        "hbd": Lipinski.NumHDonors(probe),
        "hba": Lipinski.NumHAcceptors(probe),
        "tpsa": round(rdMolDescriptors.CalcTPSA(probe), 2),
        "mw": round(Descriptors.ExactMolWt(probe), 2),
    }


def compute_openbabel_ionization(smarts: str, probe_smiles: str) -> dict[str, Any]:
    """Use OpenBabel to determine ionization state at pH 7.4."""
    try:
        from openbabel import pybel
    except ImportError:
        logger.warning("OpenBabel not available; skipping ionization for %s", smarts)
        return {"ionized_at_ph74": False, "formal_charge_ph74": 0}

    try:
        mol_neutral = pybel.readstring("smi", probe_smiles)
        mol_ph74 = pybel.readstring("smi", probe_smiles)
        mol_ph74.OBMol.AddHydrogens(True, True, 7.4)

        neutral_charge = sum(atom.formalcharge for atom in mol_neutral.atoms)
        ph74_charge = sum(atom.formalcharge for atom in mol_ph74.atoms)

        return {
            "ionized_at_ph74": ph74_charge != neutral_charge,
            "formal_charge_ph74": ph74_charge,
        }
    except Exception as exc:
        logger.warning("OpenBabel ionization failed for %s: %s", probe_smiles, exc)
        return {"ionized_at_ph74": False, "formal_charge_ph74": 0}


def build_fg_node(fg_def: dict[str, str], probe_smiles: str | None = None) -> dict[str, Any]:
    """Build a single KG1 node dict for a functional group."""
    name = fg_def["name"]
    smarts = fg_def["smarts"]
    category = fg_def["category"]

    if probe_smiles is None:
        probe_smiles = _CATEGORY_PROBES.get(category, "CC")

    rdkit_attrs = compute_rdkit_attributes(name, smarts, probe_smiles)
    ob_attrs = compute_openbabel_ionization(smarts, probe_smiles)

    attrs = {**rdkit_attrs, **ob_attrs}
    ionization_note = (
        f"ionized at pH 7.4 (charge {attrs['formal_charge_ph74']})"
        if attrs["ionized_at_ph74"]
        else "neutral at pH 7.4"
    )

    content = (
        f"[{name}] is a {category} functional group (SMARTS: {smarts}). "
        f"Physicochemical properties: LogP contribution = {attrs['logp_contribution']}, "
        f"HBD = {attrs['hbd']}, HBA = {attrs['hba']}, "
        f"TPSA = {attrs['tpsa']} Angstrom^2, MW contribution = {attrs['mw']}. "
        f"Ionization: {ionization_note}. "
        f"Category: {category}."
    )

    return {
        "id": name,
        "name": name,
        "smarts": smarts,
        "category": category,
        "content": content,
        **attrs,
    }


def build_all_fg_nodes(
    fg_defs: list[dict],
    probe_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build all KG1 nodes from the SMARTS definition list."""
    probe_map = probe_map or {}
    nodes = []
    for fg_def in fg_defs:
        probe = probe_map.get(fg_def["name"])
        node = build_fg_node(fg_def, probe_smiles=probe)
        nodes.append(node)
        logger.info("Built node: %s", fg_def["name"])
    return nodes


if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO)
    data = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))
    nodes = build_all_fg_nodes(data["functional_groups"])
    print(f"Built {len(nodes)} FG nodes")
    print(nodes[0]["content"])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/chemistry/test_compute_nodes.py -v
```
Expected: PASS — all 4 tests green

- [ ] **Step 5: Commit**

```bash
git add chemistry/kg1_build/compute_nodes.py tests/chemistry/test_compute_nodes.py
git commit -m "feat: compute RDKit/OpenBabel attributes for KG1 FG nodes"
```

---

## Task 3: KG1 Edge Computation (MMP δLogD + Co-occurrence)

**Files:**
- Create: `chemistry/kg1_build/compute_edges.py`
- Create: `tests/chemistry/test_compute_edges.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/chemistry/test_compute_edges.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/chemistry/test_compute_edges.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'chemistry.kg1_build.compute_edges'`

- [ ] **Step 3: Implement compute_edges.py**

```python
# chemistry/kg1_build/compute_edges.py
"""Compute KG1 edges: MMP δLogD edges and FG co-occurrence edges."""
from __future__ import annotations
import logging
from collections import defaultdict
from itertools import combinations
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

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
        if count < 1:
            continue
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
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    import pandas as pd
    df = pd.read_csv("test_logd.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    # Expect columns: smiles, logd_exp (or similar)
    mmp_edges = build_mmp_edges(df, fg_defs)
    cooc_edges = build_cooccurrence_edges(df, fg_defs)
    print(f"MMP edges: {len(mmp_edges)}, Co-occurrence edges: {len(cooc_edges)}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/chemistry/test_compute_edges.py -v
```
Expected: PASS — all 4 tests green

- [ ] **Step 5: Commit**

```bash
git add chemistry/kg1_build/compute_edges.py tests/chemistry/test_compute_edges.py
git commit -m "feat: compute MMP delta-LogD and co-occurrence edges for KG1"
```

---

## Task 4: Assemble KG1 GraphML

**Files:**
- Create: `chemistry/kg1_build/build_kg1.py`
- Create: `tests/chemistry/test_build_kg1.py`

- [ ] **Step 1: Write failing test**

```python
# tests/chemistry/test_build_kg1.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/chemistry/test_build_kg1.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement build_kg1.py**

```python
# chemistry/kg1_build/build_kg1.py
"""Assemble KG1 from computed nodes and edges, export as GraphML."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import yaml

from chemistry.kg1_build.compute_nodes import build_all_fg_nodes
from chemistry.kg1_build.compute_edges import build_mmp_edges, build_cooccurrence_edges

logger = logging.getLogger(__name__)


def build_kg1(fg_defs: list[dict], molecules_df: pd.DataFrame) -> nx.Graph:
    """Build the full KG1 graph from FG definitions and molecule data."""
    G = nx.Graph()

    # --- Add FG nodes ---
    nodes = build_all_fg_nodes(fg_defs)
    for node in nodes:
        node_id = node.pop("id")
        G.add_node(node_id, **_stringify_attrs(node))

    # --- Add LogD effect nodes + MMP edges ---
    mmp_edges = build_mmp_edges(molecules_df, fg_defs)
    for edge in mmp_edges:
        target = edge["target"]
        if not G.has_node(target):
            effect_content = (
                f"LogD effect node for {edge['source']}. "
                f"Average δLogD = {edge['delta_logd']:+.2f} units "
                f"from {edge['n_pairs']} matched molecular pairs."
            )
            G.add_node(target, content=effect_content, node_type="logd_effect",
                       delta_logd=str(edge["delta_logd"]))
        G.add_edge(
            edge["source"], edge["target"],
            edge_type=edge["edge_type"],
            delta_logd=str(edge["delta_logd"]),
            n_pairs=str(edge["n_pairs"]),
            content=edge["content"],
        )

    # --- Add FG co-occurrence edges ---
    cooc_edges = build_cooccurrence_edges(molecules_df, fg_defs)
    for edge in cooc_edges:
        if G.has_node(edge["source"]) and G.has_node(edge["target"]):
            G.add_edge(
                edge["source"], edge["target"],
                edge_type=edge["edge_type"],
                count=str(edge["count"]),
                content=edge["content"],
            )

    logger.info(
        "KG1 built: %d nodes, %d edges",
        G.number_of_nodes(), G.number_of_edges()
    )
    return G


def _stringify_attrs(attrs: dict[str, Any]) -> dict[str, str]:
    """GraphML requires all attributes to be string/int/float — convert booleans."""
    result = {}
    for k, v in attrs.items():
        if isinstance(v, bool):
            result[k] = str(v)
        else:
            result[k] = v
    return result


def export_graphml(G: nx.Graph, output_path: Path) -> None:
    """Export NetworkX graph to GraphML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(G, str(output_path))
    logger.info("KG1 exported to %s (%d nodes, %d edges)",
                output_path, G.number_of_nodes(), G.number_of_edges())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    df = pd.read_csv("test_logd.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    # Normalize column names — expect 'smiles' and a logd column
    logd_col = next((c for c in df.columns if "logd" in c.lower()), None)
    if logd_col and logd_col != "logd_exp":
        df = df.rename(columns={logd_col: "logd_exp"})
    G = build_kg1(fg_defs, df)
    export_graphml(G, Path("chemistry/kg1_build/chemistry_rule_graph.graphml"))
    print(f"KG1: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/chemistry/test_build_kg1.py -v
```
Expected: PASS — all 4 tests green

- [ ] **Step 5: Run build_kg1 against real data**

```bash
python chemistry/kg1_build/build_kg1.py
```
Expected output (approximate with 37 FG definitions + effect nodes from test_logd.csv):
```
INFO:chemistry.kg1_build.build_kg1:KG1 built: 50-80 nodes, 30-70 edges
KG1: 50-80 nodes, 30-70 edges
```
Note: the spec targets 150-200 nodes at full scale — expand `fg_smarts.yaml` to ~150 FGs for production.
File `chemistry/kg1_build/chemistry_rule_graph.graphml` should be created.

- [ ] **Step 6: Commit**

```bash
git add chemistry/kg1_build/build_kg1.py tests/chemistry/test_build_kg1.py
git add chemistry/kg1_build/chemistry_rule_graph.graphml
git commit -m "feat: assemble and export KG1 functional group rule graph"
```

---

## Task 5: Build KG2 Molecule Graph

**Files:**
- Create: `chemistry/kg2_build/build_kg2.py`
- Create: `tests/chemistry/test_build_kg2.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/chemistry/test_build_kg2.py
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
        {"id": "mol_0", "smiles": "c1ccccc1"},
        {"id": "mol_1", "smiles": "c1ccccc1C(=O)O"},  # similar to benzene
        {"id": "mol_2", "smiles": "CCN"},              # not similar to benzene
    ]
    edges = build_similarity_edges(nodes, threshold=0.3)
    # benzene and benzoic acid should be similar
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/chemistry/test_build_kg2.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement build_kg2.py**

```python
# chemistry/kg2_build/build_kg2.py
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


def build_molecule_nodes(
    df: pd.DataFrame, fg_defs: list[dict]
) -> list[dict]:
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
                content = (
                    f"Molecules [{ni['smiles']}] and [{nj['smiles']}] are structurally similar "
                    f"(Tanimoto = {sim:.2f}). "
                    f"LogD values: {ni['logd_exp']} vs {nj['logd_exp']}."
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
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
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
        fgs = [f.strip() for f in node.get("functional_groups", "").split(",") if f.strip() and f.strip() != "none identified"]
        for fg in fgs:
            fg_groups[fg].append(node)

    edges = []
    for fg, group in fg_groups.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, min(i + 6, len(group))):  # cap at 5 edges per FG per node
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


def build_kg2(
    df: pd.DataFrame,
    fg_defs: list[dict],
    similarity_threshold: float = 0.6,
) -> nx.Graph:
    """Build full KG2 molecule graph."""
    G = nx.Graph()

    nodes = build_molecule_nodes(df, fg_defs)
    for node in nodes:
        node_id = node.pop("id")
        G.add_node(node_id, **node)

    node_list = [{"id": f"mol_{i}", **dict(zip(df.columns, row))} for i, row in enumerate(nodes)]
    # Rebuild with ids for edge building
    nodes_with_id = []
    for i, (_, row) in enumerate(df.iterrows()):
        fgs = detect_functional_groups(row["smiles"], fg_defs)
        nodes_with_id.append({
            "id": f"mol_{i}",
            "smiles": row["smiles"],
            "logd_exp": str(float(row["logd_exp"])),
            "scaffold": _murcko_scaffold(row["smiles"]),
            "functional_groups": ", ".join(fgs) if fgs else "none identified",
        })

    for edge in build_similarity_edges(nodes_with_id, threshold=similarity_threshold):
        G.add_edge(edge.pop("source"), edge.pop("target"), **edge)

    for edge in build_scaffold_edges(nodes_with_id):
        G.add_edge(edge.pop("source"), edge.pop("target"), **edge)

    for edge in build_fg_class_edges(nodes_with_id):
        if not G.has_edge(edge["source"], edge["target"]):
            src, tgt = edge.pop("source"), edge.pop("target")
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
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    df = pd.read_csv("test_logd.csv")
    df.columns = [c.lower().strip() for c in df.columns]
    logd_col = next((c for c in df.columns if "logd" in c.lower()), None)
    if logd_col and logd_col != "logd_exp":
        df = df.rename(columns={logd_col: "logd_exp"})
    smiles_col = next((c for c in df.columns if "smile" in c.lower()), "smiles")
    if smiles_col != "smiles":
        df = df.rename(columns={smiles_col: "smiles"})
    df = df.dropna(subset=["smiles", "logd_exp"])
    G = build_kg2(fg_defs=fg_defs, df=df)
    export_graphml(G, Path("chemistry/kg2_build/molecule_graph.graphml"))
    print(f"KG2: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/chemistry/test_build_kg2.py -v
```
Expected: PASS — all 4 tests green

- [ ] **Step 5: Run build_kg2 against real data**

```bash
python chemistry/kg2_build/build_kg2.py
```
Expected output (approximate with 50-molecule test_logd.csv):
```
INFO:chemistry.kg2_build.build_kg2:KG2 built: 50 nodes, 30-150 edges
KG2: 50 nodes, 30-150 edges
```

- [ ] **Step 6: Commit**

```bash
git add chemistry/kg2_build/build_kg2.py tests/chemistry/test_build_kg2.py
git add chemistry/kg2_build/molecule_graph.graphml
git commit -m "feat: build KG2 molecule graph with Tanimoto/scaffold/FG edges"
```

---

## Task 6: GraphGen Configuration Files

**Prerequisite:** Tasks 4 and 5 must be complete — the configs reference `chemistry_rule_graph.graphml` and `molecule_graph.graphml` which are outputs of those tasks.

**Files:**
- Create: `chemistry/configs/chemistry_atomic_config.yaml`
- Create: `chemistry/configs/chemistry_cot_config.yaml`
- Create: `chemistry/configs/chemistry_multihop_config.yaml`
- Create: `chemistry/configs/chemistry_aggregated_config.yaml`

- [ ] **Step 1: Write failing test**

```python
# tests/chemistry/test_configs.py
import yaml
from pathlib import Path

CONFIGS = [
    "chemistry/configs/chemistry_atomic_config.yaml",
    "chemistry/configs/chemistry_cot_config.yaml",
    "chemistry/configs/chemistry_multihop_config.yaml",
    "chemistry/configs/chemistry_aggregated_config.yaml",
]

def test_configs_exist():
    for path in CONFIGS:
        assert Path(path).exists(), f"Missing config: {path}"

def test_configs_valid_yaml():
    for path in CONFIGS:
        data = yaml.safe_load(Path(path).read_text())
        assert "global_params" in data
        assert "nodes" in data

def test_configs_have_required_nodes():
    required_ops = {"read", "chunk", "build_kg", "partition", "generate"}
    for path in CONFIGS:
        data = yaml.safe_load(Path(path).read_text())
        op_names = {n["op_name"] for n in data["nodes"]}
        assert required_ops == op_names, f"{path} missing ops: {required_ops - op_names}"

def test_configs_point_to_graphml():
    for path in CONFIGS:
        data = yaml.safe_load(Path(path).read_text())
        read_node = next(n for n in data["nodes"] if n["op_name"] == "read")
        input_paths = read_node["params"]["input_path"]
        assert any(p.endswith(".graphml") for p in input_paths), \
            f"{path}: read node should point to a .graphml file"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/chemistry/test_configs.py::test_configs_exist -v
```
Expected: FAIL

- [ ] **Step 3: Create chemistry_atomic_config.yaml**

```yaml
# chemistry/configs/chemistry_atomic_config.yaml
# KG1 → SFT-1 Atomic facts (Alpaca format)
global_params:
  working_dir: chemistry/output/kg1_atomic_cache
  graph_backend: networkx
  kv_backend: json_kv

nodes:
  - id: read
    op_name: read
    type: source
    dependencies: []
    params:
      input_path:
        - chemistry/kg1_build/chemistry_rule_graph.graphml

  - id: chunk
    op_name: chunk
    type: map_batch
    dependencies:
      - read
    params:
      chunk_size: 4096
      chunk_overlap: 0

  - id: build_kg
    op_name: build_kg
    type: map_batch
    execution_params:
      replicas: 1
      batch_size: 64
    dependencies:
      - chunk

  - id: partition
    op_name: partition
    type: aggregate
    dependencies:
      - build_kg
    params:
      method: dfs
      method_params:
        max_units_per_community: 1

  - id: generate
    op_name: generate
    type: map_batch
    dependencies:
      - partition
    execution_params:
      replicas: 1
      batch_size: 64
    save_output: true
    params:
      method: atomic
      data_format: Alpaca
```

- [ ] **Step 4: Create chemistry_cot_config.yaml**

```yaml
# chemistry/configs/chemistry_cot_config.yaml
# KG1 → SFT-2 Chain-of-thought (ChatML format)
global_params:
  working_dir: chemistry/output/kg1_cot_cache
  graph_backend: networkx
  kv_backend: json_kv

nodes:
  - id: read
    op_name: read
    type: source
    dependencies: []
    params:
      input_path:
        - chemistry/kg1_build/chemistry_rule_graph.graphml

  - id: chunk
    op_name: chunk
    type: map_batch
    dependencies:
      - read
    params:
      chunk_size: 4096
      chunk_overlap: 0

  - id: build_kg
    op_name: build_kg
    type: map_batch
    execution_params:
      replicas: 1
      batch_size: 64
    dependencies:
      - chunk

  - id: partition
    op_name: partition
    type: aggregate
    dependencies:
      - build_kg
    params:
      method: leiden
      method_params:
        max_size: 15
        use_lcc: false
        random_seed: 42

  - id: generate
    op_name: generate
    type: map_batch
    dependencies:
      - partition
    execution_params:
      replicas: 1
      batch_size: 64
    save_output: true
    params:
      method: cot
      data_format: Sharegpt
```

- [ ] **Step 5: Create chemistry_multihop_config.yaml**

```yaml
# chemistry/configs/chemistry_multihop_config.yaml
# KG2 → SFT-3 Comparative SAR (ChatML format)
global_params:
  working_dir: chemistry/output/kg2_multihop_cache
  graph_backend: networkx
  kv_backend: json_kv

nodes:
  - id: read
    op_name: read
    type: source
    dependencies: []
    params:
      input_path:
        - chemistry/kg2_build/molecule_graph.graphml

  - id: chunk
    op_name: chunk
    type: map_batch
    dependencies:
      - read
    params:
      chunk_size: 4096
      chunk_overlap: 0

  - id: build_kg
    op_name: build_kg
    type: map_batch
    execution_params:
      replicas: 1
      batch_size: 64
    dependencies:
      - chunk

  - id: partition
    op_name: partition
    type: aggregate
    dependencies:
      - build_kg
    params:
      method: ece
      method_params:
        max_units_per_community: 3
        min_units_per_community: 2
        max_tokens_per_community: 10240
        unit_sampling: random

  - id: generate
    op_name: generate
    type: map_batch
    dependencies:
      - partition
    execution_params:
      replicas: 1
      batch_size: 64
    save_output: true
    params:
      method: multi_hop
      data_format: ChatML
```

- [ ] **Step 6: Create chemistry_aggregated_config.yaml**

```yaml
# chemistry/configs/chemistry_aggregated_config.yaml
# KG2 → SFT-2 anchor: scaffold-class aggregated patterns (ChatML format)
global_params:
  working_dir: chemistry/output/kg2_aggregated_cache
  graph_backend: networkx
  kv_backend: json_kv

nodes:
  - id: read
    op_name: read
    type: source
    dependencies: []
    params:
      input_path:
        - chemistry/kg2_build/molecule_graph.graphml

  - id: chunk
    op_name: chunk
    type: map_batch
    dependencies:
      - read
    params:
      chunk_size: 4096
      chunk_overlap: 0

  - id: build_kg
    op_name: build_kg
    type: map_batch
    execution_params:
      replicas: 1
      batch_size: 64
    dependencies:
      - chunk

  - id: partition
    op_name: partition
    type: aggregate
    dependencies:
      - build_kg
    params:
      method: ece
      method_params:
        max_units_per_community: 5
        min_units_per_community: 2
        max_tokens_per_community: 10240
        unit_sampling: random

  - id: generate
    op_name: generate
    type: map_batch
    dependencies:
      - partition
    execution_params:
      replicas: 1
      batch_size: 64
    save_output: true
    params:
      method: aggregated
      data_format: ChatML
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
python -m pytest tests/chemistry/test_configs.py -v
```
Expected: PASS — all 4 tests green

- [ ] **Step 8: Commit**

```bash
git add chemistry/configs/ tests/chemistry/test_configs.py
git commit -m "feat: add GraphGen configs for all four chemistry dataset generation modes"
```

---

## Task 7: DPO Pair Generation

**Files:**
- Create: `chemistry/dpo/generate_dpo_pairs.py`
- Create: `tests/chemistry/test_generate_dpo_pairs.py`

- [ ] **Step 1: Write failing tests**

```python
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

    assert extract_logd_prediction("No prediction here") is None


def test_build_perturbation_rejected():
    from chemistry.dpo.generate_dpo_pairs import perturb_reasoning
    correct = (
        "Step 1: Hydroxyl group identified.\n"
        "Step 2: HBD = 1, decreases LogD.\n"
        "Step 3: At pH 7.4, neutral.\n"
        "Prediction: LogD ≈ -0.3"
    )
    rejected = perturb_reasoning(correct, perturbation="ignore_ionization")
    assert rejected != correct
    assert "LogD" in rejected


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/chemistry/test_generate_dpo_pairs.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement generate_dpo_pairs.py**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/chemistry/test_generate_dpo_pairs.py -v
```
Expected: PASS — all 4 tests green (note: `_call_llm` is not called in unit tests)

- [ ] **Step 5: Commit**

```bash
git add chemistry/dpo/generate_dpo_pairs.py tests/chemistry/test_generate_dpo_pairs.py
git commit -m "feat: DPO pair generation pipeline with GNN oracle and perturbation strategies"
```

---

## Task 8: Evaluation Script

**Files:**
- Create: `chemistry/evaluate/evaluate_logd.py`
- Create: `tests/chemistry/test_evaluate_logd.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/chemistry/test_evaluate_logd.py
import pytest
import pandas as pd


def test_extract_logd_from_responses():
    from chemistry.evaluate.evaluate_logd import extract_predictions
    responses = [
        "Step 1: ... Prediction: LogD ≈ 2.3 (range 1.8–2.8, confidence: medium)",
        "Prediction: LogD = -1.5",
        "The LogD should be approximately 0.7",
        "No numeric prediction here",
    ]
    preds = extract_predictions(responses)
    assert preds[0] == pytest.approx(2.3, abs=0.01)
    assert preds[1] == pytest.approx(-1.5, abs=0.01)
    assert preds[2] == pytest.approx(0.7, abs=0.1)
    assert preds[3] is None


def test_compute_metrics():
    from chemistry.evaluate.evaluate_logd import compute_metrics
    experimental = [1.0, 2.0, 3.0, 0.5, -1.0]
    predicted    = [1.1, 2.2, 2.8, 0.6, -0.8]
    metrics = compute_metrics(experimental, predicted)
    assert "spearman_r" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "n_evaluated" in metrics
    assert metrics["spearman_r"] > 0.95   # these are close predictions
    assert metrics["rmse"] < 0.3


def test_compute_fg_accuracy():
    from chemistry.evaluate.evaluate_logd import compute_fg_accuracy
    import yaml
    fg_defs = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))["functional_groups"]
    # Response that correctly names a FG
    responses = ["I identify a carboxylic acid group and aromatic ring. LogD ≈ 1.5"]
    smiles = ["c1ccccc1C(=O)O"]  # benzoic acid: has carboxylic_acid + aromatic_ring
    acc = compute_fg_accuracy(responses, smiles, fg_defs)
    assert acc["fg_recall"] > 0.5


def test_evaluation_report():
    from chemistry.evaluate.evaluate_logd import generate_report
    metrics = {"spearman_r": 0.78, "rmse": 1.9, "mae": 1.4, "n_evaluated": 50}
    report = generate_report(metrics, gnn_baseline={"spearman_r": 0.76, "rmse": 2.1})
    assert "spearman_r" in report
    assert "0.78" in report
    assert "PASS" in report or "FAIL" in report
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/chemistry/test_evaluate_logd.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement evaluate_logd.py**

```python
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
        f"  Spearman R          : {metrics.get('spearman_r', 'N/A'):<8}  (GNN baseline: {gnn['spearman_r']})  {'✓' if spearman_pass else '✗'}",
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/chemistry/test_evaluate_logd.py -v
```
Expected: PASS — all 4 tests green

- [ ] **Step 5: Run full test suite to confirm nothing is broken**

```bash
python -m pytest tests/chemistry/ -v
```
Expected: All tests green across all 6 test files

- [ ] **Step 6: Commit**

```bash
git add chemistry/evaluate/evaluate_logd.py tests/chemistry/test_evaluate_logd.py
git commit -m "feat: evaluation script with Spearman R, RMSE, FG accuracy, and GNN comparison"
```

---

## Running the Full Pipeline

Once all tasks are complete, run in order:

```bash
# 1. Build both KGs
python chemistry/kg1_build/build_kg1.py
python chemistry/kg2_build/build_kg2.py

# 2. Start LiteLLM proxy (required for GraphGen)
litellm --config litellm_config.yaml &

# 3. Set synthesizer/trainee env vars
# GraphGen → http_api backend → LiteLLM proxy → AWS Bedrock (no OpenAI models)
export SYNTHESIZER_BACKEND=http_api
export SYNTHESIZER_BASE_URL=http://localhost:4000
export SYNTHESIZER_MODEL=claude-sonnet-4
export SYNTHESIZER_API_KEY=your-master-key-here
export TRAINEE_BACKEND=http_api
export TRAINEE_BASE_URL=http://localhost:4000
export TRAINEE_MODEL=llama-3-2-3b
export TRAINEE_API_KEY=your-master-key-here

# 4. Run GraphGen for all four dataset types
python -m graphgen.run --config chemistry/configs/chemistry_atomic_config.yaml
python -m graphgen.run --config chemistry/configs/chemistry_cot_config.yaml
python -m graphgen.run --config chemistry/configs/chemistry_multihop_config.yaml
python -m graphgen.run --config chemistry/configs/chemistry_aggregated_config.yaml

# 5. Generate DPO pairs (requires GNN predictions CSV)
# Place your GNN predictions at chemistry/dpo/gnn_predictions.csv (columns: smiles, gnn_pred)
python chemistry/dpo/generate_dpo_pairs.py

# 6. Evaluate (after fine-tuning — place model responses at chemistry/evaluate/model_predictions.jsonl)
python chemistry/evaluate/evaluate_logd.py
```
