# pylint: disable=C0301
TEMPLATE_EN: str = """You are an expert computational chemist specializing in molecular property prediction.
You are given complete data for one molecule: its SMILES, experimental logD, lipophilicity bin, functional groups, scaffold, and physicochemical descriptors.

Your task: Generate ONE logD prediction QA pair for training a language model to predict logD from SMILES alone.

CRITICAL RULES:
1. The QUESTION must contain ONLY the SMILES string — no logD values, no bin labels, no descriptors, no functional group names.
   Acceptable question forms:
   - "What is the logD at pH 7.4 for the molecule with SMILES: <smiles>?"
   - "Predict the logD value and lipophilicity category for the compound with SMILES: <smiles>."
   - "What experimental logD would you expect for <smiles>, and what lipophilicity bin does it fall into?"

2. The ANSWER must:
   - State the logD value (matching experimental) and lipophilicity bin (low / mid / high)
   - Cite 2-3 specific structural features visible in the SMILES that drive this value
   - Reference at least one descriptor (MW, TPSA, LogP, HBD, HBA, or RotBonds) to support the prediction
   - Be concise: 3-5 sentences

Output format:
<question>question_text containing only the SMILES</question>
<answer>predicted logD = X (bin lipophilicity). Structural reasoning...</answer>

Example:
<question>What is the logD at pH 7.4 for the molecule with SMILES: Cc1ncc([N+](=O)[O-])n1CCO?</question>
<answer>The predicted logD is -0.02 (low lipophilicity). The nitro group ([N+](=O)[O-]) is strongly electron-withdrawing and polar, reducing lipophilicity substantially. The hydroxylethyl tail (CCO) adds an additional hydrogen bond donor, further increasing aqueous affinity. With TPSA = 84.5 Ų and HBD = 1, the molecule is hydrophilic and expected to have limited passive membrane permeability.</answer>

Molecule data (use this to write a high-quality answer — do NOT include logD or descriptors in the question):
{context}

Generate 1 logD prediction QA pair where the question contains only the SMILES: {smiles}
"""

LOGD_PREDICTION_PROMPT = {
    "en": TEMPLATE_EN,
}
