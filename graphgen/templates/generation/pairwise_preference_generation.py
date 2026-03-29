# pylint: disable=C0301
TEMPLATE_EN: str = """You are an expert medicinal chemist specializing in drug-likeness and lipophilicity assessment.
You are given data for two molecules from a chemistry knowledge graph.

Generate ONE comparison question asking which molecule has a property advantage, and provide a detailed answer that:
1. Clearly states which molecule is preferred (or if they are equivalent) for the asked property
2. Cites the specific logD values, lipophilicity bins, or structural features from the data
3. Explains the mechanistic or structural reason for the difference
4. Notes any pharmacokinetic implications (e.g., oral absorption, CNS penetration, metabolic stability)

Important: Whenever you reference a molecule in the question or answer, always follow its label with the SMILES in parentheses, e.g.: "Molecule A (SMILES: CC(=O)O)".

Good question types:
- "Which of these two molecules is more lipophilic, and what structural features drive the difference?"
- "Which molecule would you expect to have better passive membrane permeability based on logD?"
- "Which compound is more likely to have higher aqueous solubility, and why?"
- "Comparing these two molecules, which is more drug-like according to Lipinski criteria?"

Output format:
<question>question_text</question>
<answer>detailed_comparative_answer_with_reasoning</answer>

Molecular data:
{context}

Generate 1 pairwise preference QA pair:
"""

PAIRWISE_PREFERENCE_GENERATION_PROMPT = {
    "en": TEMPLATE_EN,
}
