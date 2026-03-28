# pylint: disable=C0301
TEMPLATE_EN: str = """You are a medicinal chemistry instructor creating true/false assessment items.
Generate {num_of_questions} true/false statements about the molecular data below.

Mix approximately half true and half false statements.
False statements should contain chemically plausible but incorrect claims, such as:
- A logD value that is close but wrong (e.g., +0.5 instead of -0.02)
- An incorrect lipophilicity bin (e.g., "medium" when the data says "low")
- A reversed SAR claim (e.g., "increases" when it should "decreases")
- An incorrect functional group attribution

Good statement types:
- "The experimental logD of molecule X is Y" (cite actual or false value)
- "Molecule X is classified as [low/medium/high] lipophilicity"
- "The [functional group] in molecule X contributes to [higher/lower] logD"
- "Molecule X and Y have similar lipophilicity profiles" (true or false based on their bins)
- "A molecule with logD < 0 is typically more water-soluble than one with logD > 3"

Output Format:
<qa_pairs>
<qa_pair>
<question>Statement text</question>
<answer>True or False</answer>
</qa_pair>
</qa_pairs>

Example:
<qa_pairs>
<qa_pair>
<question>The molecule Cc1ncc([N+](=O)[O-])n1CCO has an experimental logD of -0.02, placing it in the low lipophilicity bin.</question>
<answer>True</answer>
</qa_pair>
<qa_pair>
<question>Molecules with logD values below zero are generally more lipophilic than water-soluble.</question>
<answer>False</answer>
</qa_pair>
</qa_pairs>

Molecular data:
{context}

Generate {num_of_questions} chemistry true/false statements:
"""

CHEMISTRY_TF_GENERATION_PROMPT = {
    "en": TEMPLATE_EN,
}
