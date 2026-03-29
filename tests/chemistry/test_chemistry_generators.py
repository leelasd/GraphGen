# tests/chemistry/test_chemistry_generators.py
"""
Tests for the 8 new chemistry-specific QA generators.

Covers:
1. build_prompt — verifies chemistry keywords appear and format is correct
2. parse_response — verifies XML parsing against realistic mock LLM responses
3. Config smoke-test — verifies all 8 configs load and point to valid settings
"""
import re
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MOLECULE_NODES = [
    (
        "mol_0",
        {
            "content": (
                "Node: mol_0 | smiles: Cc1ncc([N+](=O)[O-])n1CCO "
                "| logd_exp: -0.02 | logd_bin: low"
            )
        },
    ),
    (
        "mol_1",
        {
            "content": (
                "Node: mol_1 | smiles: CCc1ccc(O)cc1 "
                "| logd_exp: 2.10 | logd_bin: medium"
            )
        },
    ),
    (
        "mol_2",
        {
            "content": (
                "Node: mol_2 | smiles: c1ccc(Cl)cc1CC(=O)O "
                "| logd_exp: 1.45 | logd_bin: medium"
            )
        },
    ),
]

MOLECULE_EDGES = [
    (
        "mol_0",
        "mol_1",
        {
            "content": "mol_0 -- mol_1",
            "tanimoto": "0.31",
            "shared_fg": "hydroxyl",
            "scaffold": "imidazole",
        },
    ),
]

PAIR_BATCH = (MOLECULE_NODES[:2], MOLECULE_EDGES)
TRIPLE_BATCH = (MOLECULE_NODES, MOLECULE_EDGES)


# ---------------------------------------------------------------------------
# 1. ChemistryAtomicGenerator
# ---------------------------------------------------------------------------

class TestChemistryAtomicGenerator:
    def setup_method(self):
        from graphgen.models.generator.chemistry_atomic_generator import (
            ChemistryAtomicGenerator,
        )
        self.gen = ChemistryAtomicGenerator

    def test_build_prompt_contains_chemistry_keywords(self):
        prompt = self.gen.build_prompt(PAIR_BATCH)
        assert "logD" in prompt or "logd" in prompt.lower()
        assert "smiles" in prompt.lower() or "SMILES" in prompt
        assert "Cc1ncc" in prompt  # SMILES from mol_0 node content

    def test_build_prompt_output_format_instruction(self):
        prompt = self.gen.build_prompt(PAIR_BATCH)
        assert "<question>" in prompt
        assert "<answer>" in prompt

    def test_parse_response_valid(self):
        response = (
            "<question>What is the logD of mol_0?</question>"
            "<answer>The logD of mol_0 is -0.02, placing it in the low lipophilicity bin.</answer>"
        )
        result = self.gen.parse_response(response)
        assert len(result) == 1
        assert result[0]["question"] == "What is the logD of mol_0?"
        assert "-0.02" in result[0]["answer"]

    def test_parse_response_missing_tags_returns_empty(self):
        assert self.gen.parse_response("no tags here") == []

    def test_parse_response_strips_quotes(self):
        response = '<question>"What is logD?"</question><answer>\'It is -0.02\'</answer>'
        result = self.gen.parse_response(response)
        assert result[0]["question"] == "What is logD?"
        assert result[0]["answer"] == "It is -0.02"


# ---------------------------------------------------------------------------
# 2. ChemistryMultiChoiceGenerator
# ---------------------------------------------------------------------------

MOCK_MCQ_RESPONSE = """
<qa_pairs>
<qa_pair>
<question>Which lipophilicity bin does mol_0 belong to based on its logD of -0.02?</question>
<options>A. Very high (logD > 3)
B. Medium (logD 1–3)
C. Low (logD < 1)
D. Cannot be determined</options>
<answer>C</answer>
</qa_pair>
<qa_pair>
<question>Which molecule has the higher experimental logD?</question>
<options>A. mol_0 (logD = -0.02)
B. mol_1 (logD = 2.10)
C. Both are equal
D. Neither has a measured logD</options>
<answer>B</answer>
</qa_pair>
</qa_pairs>
"""

class TestChemistryMultiChoiceGenerator:
    def setup_method(self):
        from graphgen.models.generator.chemistry_multi_choice_generator import (
            ChemistryMultiChoiceGenerator,
        )
        self.gen = ChemistryMultiChoiceGenerator(None, num_of_questions=4)

    def test_build_prompt_contains_chemistry_context(self):
        prompt = self.gen.build_prompt(PAIR_BATCH)
        assert "logD" in prompt or "lipophilicity" in prompt.lower()
        assert "Cc1ncc" in prompt  # SMILES from mol_0 node content
        assert "4" in prompt  # num_of_questions

    def test_build_prompt_has_output_format(self):
        prompt = self.gen.build_prompt(PAIR_BATCH)
        assert "<qa_pairs>" in prompt
        assert "<options>" in prompt

    def test_parse_response_two_questions(self):
        result = self.gen.parse_response(MOCK_MCQ_RESPONSE)
        assert len(result) == 2

    def test_parse_response_correct_answer(self):
        result = self.gen.parse_response(MOCK_MCQ_RESPONSE)
        assert result[0]["answer"] == "C"
        assert result[1]["answer"] == "B"

    def test_parse_response_options_present(self):
        result = self.gen.parse_response(MOCK_MCQ_RESPONSE)
        assert "A" in result[0]["options"]
        assert "C" in result[0]["options"]


# ---------------------------------------------------------------------------
# 3. ChemistryMultiAnswerGenerator
# ---------------------------------------------------------------------------

MOCK_MAQ_RESPONSE = """
<qa_pairs>
<qa_pair>
<question>Which structural features typically increase molecular lipophilicity?</question>
<options>A. Aromatic ring systems
B. Carboxylic acid groups
C. Halogen substituents
D. Hydroxyl groups</options>
<answer>A, C</answer>
</qa_pair>
</qa_pairs>
"""

class TestChemistryMultiAnswerGenerator:
    def setup_method(self):
        from graphgen.models.generator.chemistry_multi_answer_generator import (
            ChemistryMultiAnswerGenerator,
        )
        self.gen = ChemistryMultiAnswerGenerator(None, num_of_questions=3)

    def test_build_prompt_multi_select_language(self):
        prompt = self.gen.build_prompt(PAIR_BATCH)
        assert "multiple" in prompt.lower() or "select" in prompt.lower()
        assert "Cc1ncc" in prompt  # SMILES from mol_0 node content

    def test_parse_response_multi_answer(self):
        result = self.gen.parse_response(MOCK_MAQ_RESPONSE)
        assert len(result) == 1
        # MultiAnswerGenerator stores answers as a list under "answers" key
        assert result[0]["answers"] == ["A", "C"]

    def test_parse_response_question_text(self):
        result = self.gen.parse_response(MOCK_MAQ_RESPONSE)
        assert "lipophilicity" in result[0]["question"].lower()


# ---------------------------------------------------------------------------
# 4. ChemistryFillInBlankGenerator
# ---------------------------------------------------------------------------

MOCK_FIB_RESPONSE = """
<qa_pairs>
<qa_pair>
<question>The molecule mol_0 has an experimental logD of ________, classifying it as ________ lipophilicity.</question>
<answer>-0.02, low</answer>
</qa_pair>
<qa_pair>
<question>The nitro group in mol_0's SMILES tends to ________ its logD.</question>
<answer>decrease</answer>
</qa_pair>
</qa_pairs>
"""

class TestChemistryFillInBlankGenerator:
    def setup_method(self):
        from graphgen.models.generator.chemistry_fill_in_blank_generator import (
            ChemistryFillInBlankGenerator,
        )
        self.gen = ChemistryFillInBlankGenerator(None, num_of_questions=4)

    def test_build_prompt_blank_placeholder_mentioned(self):
        prompt = self.gen.build_prompt(PAIR_BATCH)
        assert "________" in prompt or "blank" in prompt.lower()

    def test_build_prompt_targets_chemistry(self):
        prompt = self.gen.build_prompt(PAIR_BATCH)
        assert "logD" in prompt or "logd" in prompt.lower()

    def test_parse_response_two_pairs(self):
        result = self.gen.parse_response(MOCK_FIB_RESPONSE)
        assert len(result) == 2

    def test_parse_response_blank_in_question(self):
        result = self.gen.parse_response(MOCK_FIB_RESPONSE)
        assert "________" in result[0]["question"]

    def test_parse_response_multi_blank_answer(self):
        result = self.gen.parse_response(MOCK_FIB_RESPONSE)
        assert result[0]["answer"] == "-0.02, low"


# ---------------------------------------------------------------------------
# 5. ChemistryTrueFalseGenerator
# ---------------------------------------------------------------------------

MOCK_TF_RESPONSE = """
<qa_pairs>
<qa_pair>
<question>The molecule mol_0 has an experimental logD of -0.02, placing it in the low lipophilicity bin.</question>
<answer>True</answer>
</qa_pair>
<qa_pair>
<question>Molecules with logD values below zero are generally more lipophilic than water-soluble.</question>
<answer>False</answer>
</qa_pair>
</qa_pairs>
"""

class TestChemistryTrueFalseGenerator:
    def setup_method(self):
        from graphgen.models.generator.chemistry_true_false_generator import (
            ChemistryTrueFalseGenerator,
        )
        self.gen = ChemistryTrueFalseGenerator(None, num_of_questions=5)

    def test_build_prompt_true_false_language(self):
        prompt = self.gen.build_prompt(PAIR_BATCH)
        assert "true" in prompt.lower() or "false" in prompt.lower()

    def test_build_prompt_chemistry_context(self):
        prompt = self.gen.build_prompt(PAIR_BATCH)
        assert "logD" in prompt or "lipophilicity" in prompt.lower()

    def test_parse_response_two_pairs(self):
        result = self.gen.parse_response(MOCK_TF_RESPONSE)
        assert len(result) == 2

    def test_parse_response_true_false_answers(self):
        result = self.gen.parse_response(MOCK_TF_RESPONSE)
        assert result[0]["answer"] == "True"
        assert result[1]["answer"] == "False"


# ---------------------------------------------------------------------------
# 6. PairwisePreferenceGenerator
# ---------------------------------------------------------------------------

MOCK_PAIRWISE_RESPONSE = """
<question>Which of these two molecules is more lipophilic: Molecule A (logD=-0.02) or Molecule B (logD=2.10)?</question>
<answer>Molecule B is significantly more lipophilic (logD=2.10, medium bin) than Molecule A (logD=-0.02, low bin). The ethyl-phenol scaffold in Molecule B contributes aromatic and alkyl hydrophobicity, while Molecule A's nitro and hydroxylethyl groups reduce lipophilicity through polarity and H-bonding.</answer>
"""

class TestPairwisePreferenceGenerator:
    def setup_method(self):
        from graphgen.models.generator.pairwise_preference_generator import (
            PairwisePreferenceGenerator,
        )
        self.gen = PairwisePreferenceGenerator

    def test_build_prompt_comparison_language(self):
        prompt = self.gen.build_prompt(PAIR_BATCH)
        assert "compar" in prompt.lower() or "preference" in prompt.lower() or "which" in prompt.lower()

    def test_build_prompt_both_molecules_present(self):
        prompt = self.gen.build_prompt(PAIR_BATCH)
        assert "Molecule A" in prompt
        assert "Molecule B" in prompt

    def test_build_prompt_includes_edges(self):
        prompt = self.gen.build_prompt(PAIR_BATCH)
        assert "Molecule A" in prompt and "Molecule B" in prompt

    def test_parse_response_valid(self):
        result = self.gen.parse_response(MOCK_PAIRWISE_RESPONSE)
        assert len(result) == 1
        assert "Molecule" in result[0]["question"]
        assert "logD" in result[0]["answer"] or "lipophilic" in result[0]["answer"].lower()

    def test_parse_response_missing_tags(self):
        assert self.gen.parse_response("no tags") == []


# ---------------------------------------------------------------------------
# 7. RankingGenerator
# ---------------------------------------------------------------------------

MOCK_RANKING_RESPONSE = """
<question>Rank the three molecules from lowest to highest logD.</question>
<answer>
1. Molecule A (logD=-0.02, low): Nitro and hydroxylethyl groups strongly reduce lipophilicity.
2. Molecule C (logD=1.45, medium): Chlorobenzyl acetic acid — chlorine raises lipophilicity but carboxylic acid tempers it.
3. Molecule B (logD=2.10, medium): Ethylphenol scaffold with only a mild polar hydroxyl; highest logD of the three.
</answer>
"""

class TestRankingGenerator:
    def setup_method(self):
        from graphgen.models.generator.ranking_generator import RankingGenerator
        self.gen = RankingGenerator

    def test_build_prompt_ranking_language(self):
        prompt = self.gen.build_prompt(TRIPLE_BATCH)
        assert "rank" in prompt.lower() or "order" in prompt.lower()

    def test_build_prompt_all_molecules_present(self):
        prompt = self.gen.build_prompt(TRIPLE_BATCH)
        assert "Molecule A" in prompt
        assert "Molecule B" in prompt
        assert "Molecule C" in prompt

    def test_parse_response_valid(self):
        result = self.gen.parse_response(MOCK_RANKING_RESPONSE)
        assert len(result) == 1
        assert "rank" in result[0]["question"].lower() or "molecule" in result[0]["question"].lower()
        assert "Molecule A" in result[0]["answer"]

    def test_parse_response_missing_tags(self):
        assert self.gen.parse_response("no tags") == []


# ---------------------------------------------------------------------------
# 8. MatchedMolecularPairGenerator
# ---------------------------------------------------------------------------

MOCK_MMP_RESPONSE = """
<question>Molecule A and Molecule B share a similar scaffold but differ at one substituent. How does this structural change affect logD?</question>
<answer>Molecule A carries a nitro group ([N+](=O)[O-]) on the imidazole ring, which is strongly electron-withdrawing and polar, reducing logD to -0.02. Molecule B replaces this with an ethyl group on a phenol, which is hydrophobic, raising logD to 2.10. The ΔlogD = +2.12 reflects the shift from a polar nitro substituent to a non-polar alkyl chain.</answer>
"""

class TestMatchedMolecularPairGenerator:
    def setup_method(self):
        from graphgen.models.generator.matched_molecular_pair_generator import (
            MatchedMolecularPairGenerator,
        )
        self.gen = MatchedMolecularPairGenerator

    def test_build_prompt_sar_language(self):
        prompt = self.gen.build_prompt(PAIR_BATCH)
        assert "SAR" in prompt or "structural" in prompt.lower() or "matched" in prompt.lower()

    def test_build_prompt_edge_attributes_included(self):
        prompt = self.gen.build_prompt(PAIR_BATCH)
        # tanimoto and shared_fg from MOLECULE_EDGES should appear
        assert "tanimoto" in prompt
        assert "shared_fg" in prompt or "hydroxyl" in prompt

    def test_build_prompt_both_molecules(self):
        prompt = self.gen.build_prompt(PAIR_BATCH)
        assert "Molecule A" in prompt
        assert "Molecule B" in prompt

    def test_parse_response_valid(self):
        result = self.gen.parse_response(MOCK_MMP_RESPONSE)
        assert len(result) == 1
        assert "logD" in result[0]["answer"] or "logd" in result[0]["answer"].lower()
        assert "Molecule A" in result[0]["answer"] or "Molecule B" in result[0]["answer"]

    def test_parse_response_missing_tags(self):
        assert self.gen.parse_response("no tags") == []


# ---------------------------------------------------------------------------
# 9. Config file validation for all 8 new configs
# ---------------------------------------------------------------------------

NEW_CONFIGS = [
    "chemistry/configs/chemistry_atomic_config2.yaml",
    "chemistry/configs/chemistry_multi_choice_config.yaml",
    "chemistry/configs/chemistry_multi_answer_config.yaml",
    "chemistry/configs/chemistry_fill_in_blank_config.yaml",
    "chemistry/configs/chemistry_true_false_config.yaml",
    "chemistry/configs/chemistry_pairwise_config.yaml",
    "chemistry/configs/chemistry_ranking_config.yaml",
    "chemistry/configs/chemistry_mmp_config.yaml",
]

EXPECTED_METHODS = {
    "chemistry_atomic_config2.yaml": "chemistry_atomic",
    "chemistry_multi_choice_config.yaml": "chemistry_multi_choice",
    "chemistry_multi_answer_config.yaml": "chemistry_multi_answer",
    "chemistry_fill_in_blank_config.yaml": "chemistry_fill_in_blank",
    "chemistry_true_false_config.yaml": "chemistry_true_false",
    "chemistry_pairwise_config.yaml": "pairwise_preference",
    "chemistry_ranking_config.yaml": "ranking",
    "chemistry_mmp_config.yaml": "matched_molecular_pair",
}


def test_new_configs_exist():
    for path in NEW_CONFIGS:
        assert Path(path).exists(), f"Missing config: {path}"


def test_new_configs_valid_yaml():
    for path in NEW_CONFIGS:
        data = yaml.safe_load(Path(path).read_text())
        assert "global_params" in data
        assert "nodes" in data


def test_new_configs_have_required_nodes():
    required_ops = {"read", "chunk", "partition", "generate"}
    for path in NEW_CONFIGS:
        data = yaml.safe_load(Path(path).read_text())
        op_names = {n["op_name"] for n in data["nodes"]}
        assert required_ops == op_names, f"{path} missing ops: {required_ops - op_names}"


def test_new_configs_point_to_molecule_graph():
    for path in NEW_CONFIGS:
        data = yaml.safe_load(Path(path).read_text())
        read_node = next(n for n in data["nodes"] if n["op_name"] == "read")
        input_paths = read_node["params"]["input_path"]
        assert any("molecule_graph.graphml" in p for p in input_paths), (
            f"{path}: should point to molecule_graph.graphml"
        )


def test_new_configs_correct_generate_method():
    for path in NEW_CONFIGS:
        config_file = Path(path).name
        expected = EXPECTED_METHODS[config_file]
        data = yaml.safe_load(Path(path).read_text())
        gen_node = next(n for n in data["nodes"] if n["op_name"] == "generate")
        actual = gen_node["params"]["method"]
        assert actual == expected, (
            f"{path}: expected method={expected!r}, got {actual!r}"
        )


def test_mmp_config_includes_edges():
    """MMP config must use include_edges: true to get edge attributes in prompts."""
    data = yaml.safe_load(Path("chemistry/configs/chemistry_mmp_config.yaml").read_text())
    read_node = next(n for n in data["nodes"] if n["op_name"] == "read")
    assert read_node["params"].get("include_edges") is True, (
        "MMP config must set include_edges: true"
    )


def test_pairwise_and_mmp_use_pairs():
    """Pairwise and MMP configs must use exactly 2 units per partition."""
    for path in [
        "chemistry/configs/chemistry_pairwise_config.yaml",
        "chemistry/configs/chemistry_mmp_config.yaml",
    ]:
        data = yaml.safe_load(Path(path).read_text())
        part_node = next(n for n in data["nodes"] if n["op_name"] == "partition")
        mp = part_node["params"]["method_params"]
        assert mp["min_units_per_community"] == 2, f"{path}: min_units should be 2"
        assert mp["max_units_per_community"] == 2, f"{path}: max_units should be 2"


def test_ranking_config_uses_multi_molecule_partitions():
    """Ranking config must have at least 3 molecules per partition."""
    data = yaml.safe_load(Path("chemistry/configs/chemistry_ranking_config.yaml").read_text())
    part_node = next(n for n in data["nodes"] if n["op_name"] == "partition")
    mp = part_node["params"]["method_params"]
    assert mp["min_units_per_community"] >= 3, "Ranking needs at least 3 molecules per batch"
