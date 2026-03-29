# pylint: disable=C0301
TEMPLATE_EN: str = """You are a medicinal chemistry exam question writer.
Generate {num_of_questions} multiple-choice questions about the molecular data below.

Focus on chemistry-relevant aspects:
- Lipophilicity: logD values, bin classifications (low/medium/high), and their pharmacokinetic implications
- Structural features: functional groups present in SMILES, how they affect properties
- Structure-property relationships: why a structural feature raises or lowers logD
- Comparative properties: which molecule is more/less lipophilic among the set
- Drug-likeness: Lipinski Rule of Five, oral bioavailability considerations

Each question must have 4 options (A/B/C/D) with exactly ONE correct answer.
Distractors must be chemically plausible (e.g., nearby numeric logD values, related but incorrect functional group names).
When referencing a specific molecule in a question or answer option, always identify it by its SMILES string, e.g.: "the molecule with SMILES CC(=O)O".

Output Format:
<qa_pairs>
<qa_pair>
<question>Question text</question>
<options>A. Option A
B. Option B
C. Option C
D. Option D</options>
<answer>Correct option letter</answer>
</qa_pair>
</qa_pairs>

Example:
<qa_pairs>
<qa_pair>
<question>The molecule Cc1ncc([N+](=O)[O-])n1CCO has an experimental logD of -0.02. Which lipophilicity bin does this place it in?</question>
<options>A. Very high (logD > 3)
B. Medium (logD 1–3)
C. Low (logD < 1)
D. Cannot be determined from logD alone</options>
<answer>C</answer>
</qa_pair>
</qa_pairs>

Molecular data:
{context}

Generate {num_of_questions} chemistry multiple-choice questions:
"""

CHEMISTRY_MCQ_GENERATION_PROMPT = {
    "en": TEMPLATE_EN,
}
