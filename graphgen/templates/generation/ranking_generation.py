# pylint: disable=C0301
TEMPLATE_EN: str = """You are an expert medicinal chemist.
You are given a set of molecules from a chemistry knowledge graph.

Generate ONE ranking question that asks the model to order the molecules by a chemical property, and provide a ranked answer with justification.

Question requirements:
- Ask for an ordering by a specific property: logD, lipophilicity, expected solubility, drug-likeness, or membrane permeability
- Reference all molecules in the set
- Be specific about the direction (e.g., "from lowest to highest logD")

Answer requirements:
- List the molecules in ranked order (1st = lowest, last = highest, or reverse as appropriate)
- For each position, identify the molecule by label AND include its SMILES in parentheses, e.g.: "1. Molecule A (SMILES: CC(=O)O, logD = -1.2, low): ..."
- Cite the logD value or bin classification from the data for each molecule
- Briefly explain why each molecule is ranked where it is, using structural features where available
- If two molecules have similar values, note that they are close

Output format:
<question>question_text</question>
<answer>ranked_list_with_justification</answer>

Example:
<question>Rank the following three molecules from lowest to highest logD, and justify each position based on their structural features.</question>
<answer>
1. Molecule A (SMILES: OC(=O)CC(=O)O, logD = -1.2, low): Contains two polar carboxylic acid groups that strongly reduce lipophilicity.
2. Molecule B (SMILES: Oc1ccccc1CC, logD = 0.5, low): Has a hydroxyl group and a small aromatic ring, giving moderate polarity.
3. Molecule C (SMILES: Clc1ccc2ccccc2c1, logD = 3.1, high): Dominated by a bicyclic aromatic core and chlorine substituent, both increasing lipophilicity.
</answer>

Molecular data:
{context}

Generate 1 ranking QA pair:
"""

RANKING_GENERATION_PROMPT = {
    "en": TEMPLATE_EN,
}
