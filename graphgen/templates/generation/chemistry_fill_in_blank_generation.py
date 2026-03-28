# pylint: disable=C0301
TEMPLATE_EN: str = """You are creating chemistry study materials for medicinal chemistry students.
Generate {num_of_questions} fill-in-the-blank questions from the molecular data below.

Good targets for blanks:
- The numeric logD value of a molecule (e.g., "The logD of mol_X is ________")
- The lipophilicity bin classification (low / medium / high)
- A key functional group name found in the SMILES
- A property direction (increases / decreases / no change)
- A structural feature that explains a property (scaffold name, substituent type)
- A comparative qualifier (more lipophilic / less soluble than)

Use ________ (four underscores) as the blank placeholder.
Each question must be self-contained and answerable directly from the provided data.

Output Format:
<qa_pairs>
<qa_pair>
<question>Statement with ________ placeholder(s)</question>
<answer>The blank value(s), comma-separated if multiple blanks</answer>
</qa_pair>
</qa_pairs>

Example:
<qa_pairs>
<qa_pair>
<question>The molecule Cc1ncc([N+](=O)[O-])n1CCO has an experimental logD of ________, classifying it as ________ lipophilicity.</question>
<answer>-0.02, low</answer>
</qa_pair>
<qa_pair>
<question>The nitro group ([N+](=O)[O-]) in the SMILES tends to ________ molecular lipophilicity due to its strong electron-withdrawing and polar character.</question>
<answer>decrease</answer>
</qa_pair>
</qa_pairs>

Molecular data:
{context}

Generate {num_of_questions} chemistry fill-in-the-blank questions:
"""

CHEMISTRY_FILL_IN_BLANK_GENERATION_PROMPT = {
    "en": TEMPLATE_EN,
}
