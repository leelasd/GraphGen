"""Compute RDKit and OpenBabel physicochemical attributes for each functional group node."""
from __future__ import annotations
import logging
from typing import Any

import yaml
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

logger = logging.getLogger(__name__)

# Probe molecules for each FG category — used when no explicit probe provided
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
