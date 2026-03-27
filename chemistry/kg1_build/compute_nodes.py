"""Compute RDKit and OpenBabel physicochemical attributes for each functional group node."""
from __future__ import annotations
import logging
from typing import Any

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

logger = logging.getLogger(__name__)

# Probe molecules: one per FG that actually contains the SMARTS pattern.
# Verified: each SMARTS matches its probe via RDKit HasSubstructMatch.
_FG_PROBES: dict[str, str] = {
    # acidic
    "carboxylic_acid": "CC(=O)O",
    "sulfonamide": "CS(=O)(=O)N",
    "tetrazole": "c1nn[nH]n1",
    "phosphoric_acid": "OP(=O)(O)O",
    # basic
    "primary_amine": "CCN",
    "secondary_amine": "CCNCC",
    "tertiary_amine": "CCN(CC)CC",
    "piperazine": "C1CNCCN1",
    "morpholine": "C1CNCCO1",
    "piperidine": "C1CCNCC1",
    "pyrrolidine": "C1CCNC1",
    "pyridine": "c1ccncc1",
    "imidazole": "c1cnc[nH]1",
    "guanidine": "NC(=N)N",
    # neutral polar
    "amide": "CC(=O)N",
    "hydroxyl": "CCO",
    "ether": "CCOCC",
    "ester": "CC(=O)OCC",
    "ketone": "CC(=O)C",
    "aldehyde": "CC=O",
    "nitrile": "CC#N",
    "urea": "NC(=O)N",
    # lipophilic
    "aromatic_ring": "c1ccccc1",
    "trifluoromethyl": "CC(F)(F)F",
    "cyclopropyl": "C1CC1C",
    "indole": "c1ccc2[nH]ccc2c1",
    "thiophene": "c1ccsc1",
    # halogen
    "fluorine": "CCF",
    "chlorine": "CCCl",
    "bromine": "CCBr",
    # ewg
    "nitro": "CC[N+](=O)[O-]",
    "trifluoromethyl_ewg": "FC(F)(F)c1ccccc1",
    # sulfur
    "thiol": "CCS",
    "thioether": "CCSCC",
    "sulfoxide": "CCS(=O)C",
    "sulfone": "CCS(=O)(=O)C",
}

# Category fallbacks for any FG not in _FG_PROBES
_CATEGORY_PROBES = {
    "acidic": "CC(=O)O",
    "basic": "CCN",
    "neutral_polar": "CCO",
    "lipophilic": "c1ccccc1",
    "halogen": "CCF",
    "ewg": "CC(=O)C",
    "sulfur": "CCS",
}

_REF_MOL = Chem.MolFromSmiles("C")  # methane, LogP = 0.666; used as LogP reference


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
    probe_logp = Crippen.MolLogP(probe)
    logp_contrib = round(probe_logp - Crippen.MolLogP(_REF_MOL), 3)

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
        probe_smiles = _FG_PROBES.get(name) or _CATEGORY_PROBES.get(category, "CC")

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
    import yaml  # noqa: PLC0415
    logging.basicConfig(level=logging.INFO)
    data = yaml.safe_load(open("chemistry/kg1_build/fg_smarts.yaml"))
    nodes = build_all_fg_nodes(data["functional_groups"])
    print(f"Built {len(nodes)} FG nodes")
    print(nodes[0]["content"])
