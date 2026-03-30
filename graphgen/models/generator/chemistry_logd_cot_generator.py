import re
from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates.generation.logd_cot_generation import LOGD_COT_PROMPT
from graphgen.utils import logger


class ChemistryLogdCotGenerator(BaseGenerator):
    """
    Generates chain-of-thought (CoT) logD prediction QA pairs.

    Question contains ONLY the SMILES string.
    Answer follows a 4-step reasoning chain:
      1. SMILES parsing — identify key structural features
      2. Functional group contributions — explain each FG's effect on logD
      3. Descriptor analysis — interpret MW, LogP, TPSA, HBD, HBA, RotBonds
      4. LogD prediction — final logD value, bin (low/mid/high), PK implication

    Designed for single-molecule partitions (max_units=1) on the KG2 molecule graph.
    """

    @staticmethod
    def _build_context(attrs: dict) -> str:
        """Extract all available molecular data into a context string for answer generation."""
        lines = []
        for key in [
            "smiles", "logd_exp", "logd_bin", "functional_groups", "scaffold",
            "logp", "mw", "hbd", "hba", "tpsa", "rotbonds",
        ]:
            val = attrs.get(key, "")
            if val:
                lines.append(f"{key}: {val}")
        desc = attrs.get("content", "")
        if desc:
            lines.append(f"description: {desc}")
        return "\n".join(lines)

    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        nodes, _ = batch
        if not nodes:
            return ""
        _, attrs = nodes[0]
        smiles = attrs.get("smiles", "")
        if not smiles:
            content = attrs.get("content", "")
            m = re.search(r"(?:^|[\s|])smiles:\s*(\S+)", content)
            smiles = m.group(1).rstrip("|").strip() if m else ""
        if not smiles:
            logger.warning("ChemistryLogdCotGenerator: no SMILES found in node attrs")
            return ""
        context = ChemistryLogdCotGenerator._build_context(attrs)
        return LOGD_COT_PROMPT["en"].format(smiles=smiles, context=context)

    async def generate(self, batch) -> list[dict]:
        prompt = self.build_prompt(batch)
        if not prompt:
            return []
        response = await self.llm_client.generate_answer(prompt)
        return self.parse_response(response)

    @staticmethod
    def parse_response(response: str) -> list[dict]:
        question_match = re.search(r"<question>(.*?)</question>", response, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if question_match and answer_match:
            question = question_match.group(1).strip().strip('"').strip("'")
            answer = answer_match.group(1).strip().strip('"').strip("'")
            return [{"question": question, "answer": answer}]
        logger.warning("Failed to parse logD CoT response: %s", response)
        return []
