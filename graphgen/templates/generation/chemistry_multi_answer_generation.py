# pylint: disable=C0301
TEMPLATE_EN: str = """You are a medicinal chemistry assessment designer.
Generate {num_of_questions} multiple-select questions about the molecular data below.
Each question should have 4 options with 1 to 3 correct answers.

Focus on topics where multiple answers naturally apply:
- Which structural features in a SMILES string contribute to increased lipophilicity? (aromatic rings, halogens, alkyl chains all may apply)
- Which molecules in the set share a scaffold class or similar Tanimoto similarity?
- Which properties are consistent with Lipinski's Rule of Five?
- Which functional groups are present in a given SMILES? (multiple groups may coexist)
- Which molecules have logD values in the same lipophilicity bin?

Separate multiple correct answer letters with commas (e.g., "A, C" or "A, B, D").
When referencing a specific molecule in a question or answer option, always identify it by its SMILES string, e.g.: "the molecule with SMILES CC(=O)O".

Output Format:
<qa_pairs>
<qa_pair>
<question>Question text</question>
<options>A. Option A
B. Option B
C. Option C
D. Option D</options>
<answer>Correct option letter(s) separated by commas</answer>
</qa_pair>
</qa_pairs>

Example:
<qa_pairs>
<qa_pair>
<question>Which of the following structural features typically increase molecular lipophilicity (logD)?</question>
<options>A. Aromatic ring systems
B. Carboxylic acid groups
C. Halogen substituents (F, Cl, Br)
D. Hydroxyl groups</options>
<answer>A, C</answer>
</qa_pair>
</qa_pairs>

Molecular data:
{context}

Generate {num_of_questions} chemistry multiple-select questions:
"""

CHEMISTRY_MAQ_GENERATION_PROMPT = {
    "en": TEMPLATE_EN,
}
