# pylint: disable=C0301
TEMPLATE_EN: str = """You are an expert medicinal chemist specializing in structure-activity relationships (SAR) and matched molecular pair (MMP) analysis.
You are given data for two structurally related molecules — either a matched molecular pair or two molecules with shared scaffold but different substituents.

Generate ONE SAR question asking how the structural difference between the two molecules affects a molecular property (typically logD), and provide a mechanistic answer.

Question requirements:
- Identify the structural change between the two SMILES strings (different substituent, functional group transformation, ring modification)
- Ask how this change affects logD (or lipophilicity) and why
- Reference both molecule IDs or SMILES

Answer requirements:
- Identify the specific structural difference (e.g., "replacing -OH with -CH3", "adding a fluorine atom", "removing a nitro group")
- State the direction of the logD change and the magnitude if data is available (ΔlogD = logD_2 - logD_1)
- Explain the physicochemical mechanism:
  * Hydrophobic/polar contributions of the added/removed group
  * Electronic effects (electron withdrawal reduces lipophilicity; electron donation can increase it)
  * Steric effects or conformational changes
  * Ionization state effects if relevant
- Conclude with the pharmacokinetic implication of the change

Output format:
<question>question_text</question>
<answer>mechanistic_sar_explanation</answer>

Molecular data:
{context}

Generate 1 matched molecular pair SAR QA:
"""

MATCHED_MOLECULAR_PAIR_GENERATION_PROMPT = {
    "en": TEMPLATE_EN,
}
