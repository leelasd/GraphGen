# pylint: disable=C0301
TEMPLATE_EN: str = """You are an expert computational chemist. You are given complete data for one molecule: its SMILES, experimental logD, lipophilicity bin, functional groups, scaffold, and physicochemical descriptors.

Your task: Generate ONE chain-of-thought (CoT) logD prediction QA pair for training a reasoning model.
The model must learn to predict logD bin AND explain the structural/descriptor basis for that prediction.

CRITICAL RULES:
1. The QUESTION must contain ONLY the SMILES string — no logD values, no bins, no descriptors.
   Acceptable question forms:
   - "Given the molecule with SMILES: <smiles>, predict its logD bin (low/mid/high) and explain which functional groups and molecular descriptors drive that value."
   - "For the compound with SMILES: <smiles>, reason step-by-step about which structural features and descriptors determine its logD category."
   - "Analyze the SMILES <smiles>: predict its lipophilicity bin and provide a mechanistic explanation based on functional groups and physicochemical descriptors."

2. The ANSWER must follow this reasoning chain — each step is a paragraph, no numbered lists:
   Step 1 — SMILES parsing: Identify the key structural features (functional groups, ring systems, heteroatoms, ionizable groups).
   Step 2 — Functional group contributions: For each key group, explain its effect on logD (polarity, H-bond donor/acceptor, ionization at pH 7.4, hydrophobic contribution).
   Step 3 — Descriptor analysis: Cite and interpret the actual descriptor values (MW, LogP, TPSA, HBD, HBA, RotBonds) and what each implies for lipophilicity.
   Step 4 — LogD prediction: Synthesize the above into a final prediction — state the logD value, bin (low/mid/high), and one pharmacokinetic implication.

Output format:
<question>question_text containing only the SMILES</question>
<answer>
**Step 1 — Structural features:** [parse SMILES, identify FGs and ring systems]
**Step 2 — Functional group contributions:** [explain each FG's effect on logD]
**Step 3 — Descriptor analysis:** [cite MW, LogP, TPSA, HBD, HBA, RotBonds with values]
**Step 4 — LogD prediction:** [final prediction: logD ≈ X, bin = Y, PK implication]
</answer>

Example:
<question>Given the molecule with SMILES: Cc1ncc([N+](=O)[O-])n1CCO, predict its logD bin (low/mid/high) and explain which functional groups and molecular descriptors drive that value.</question>
<answer>
**Step 1 — Structural features:** The SMILES encodes a methylimidazole ring bearing a nitro group ([N+](=O)[O-]) at C4 and a 2-hydroxyethyl chain (CCO) at N1. Key polar centers include the nitro group, the two ring nitrogens, and the terminal hydroxyl.

**Step 2 — Functional group contributions:** The nitro group is strongly electron-withdrawing and polar (dipole ~3.9 D), contributing significantly to aqueous solubility and reducing logD. Neither nitrogen in the imidazole ring carries a free lone pair at physiological pH (pKa < 7.4 for the protonated form), so ionization is minimal, but the overall ring polarity is moderate. The 2-hydroxyethyl chain adds one hydrogen bond donor and one acceptor, further favoring aqueous partitioning over lipid phases.

**Step 3 — Descriptor analysis:** MW = 157.06 (small molecule, rapid clearance expected), LogP = 0.11 (confirms near-zero lipophilicity), TPSA = 84.5 Ų (above the 60–90 Ų range associated with CNS exclusion), HBD = 1, HBA = 4, RotBonds = 3. The high TPSA and multiple H-bond acceptors are consistent with limited passive permeability.

**Step 4 — LogD prediction:** The combined effect of the polar nitro group, hydroxyl, and imidazole ring places this compound firmly in the low lipophilicity bin. Predicted logD ≈ -0.02 (low). This compound would show high aqueous solubility but poor passive membrane permeability, likely requiring active transport for intracellular target engagement.
</answer>

Molecule data (use this to write a high-quality answer — do NOT include logD or descriptor values in the question):
{context}

Generate 1 logD CoT prediction QA pair where the question contains only the SMILES: {smiles}
"""

LOGD_COT_PROMPT = {
    "en": TEMPLATE_EN,
}
