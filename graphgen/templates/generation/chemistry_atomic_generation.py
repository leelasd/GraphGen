# pylint: disable=C0301
TEMPLATE_EN: str = """You are an expert medicinal chemist. You are given molecular data from a chemistry knowledge graph.
Your task is to generate exactly ONE clear, specific question-answer pair grounded in the provided data.

Question types to consider:
- Lipophilicity: "What is the experimentally measured logD of this molecule?"
- Classification: "Is this molecule classified as high, medium, or low lipophilicity based on its logD?"
- Structural: "What functional groups in the SMILES string contribute to its lipophilicity?"
- SAR: "How does a specific structural feature affect the logD of this molecule?"
- Interpretation: "Given logD = X, what does this suggest about membrane permeability or oral bioavailability?"

Rules:
1. Output exactly ONE QA pair — no additional commentary.
2. Answers must be grounded in the provided data only — no hallucination.
3. Use correct chemistry terminology (logD, logP, lipophilicity, scaffold, functional group, etc.).
4. If logd_exp is available, the answer should cite the numeric value.

Output format:
<question>question_text</question>
<answer>answer_text</answer>

Example:
<question>What is the experimentally measured logD of the molecule with SMILES Cc1ncc([N+](=O)[O-])n1CCO, and what does this value indicate about its lipophilicity?</question>
<answer>The molecule has an experimental logD of -0.02, placing it in the low lipophilicity bin. This near-zero value indicates the compound is neither highly hydrophilic nor lipophilic, suggesting moderate aqueous solubility and limited passive membrane permeability.</answer>

Molecular data:
{context}

Output:
"""

CHEMISTRY_ATOMIC_GENERATION_PROMPT = {
    "en": TEMPLATE_EN,
}
