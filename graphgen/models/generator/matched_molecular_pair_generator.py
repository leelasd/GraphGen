import re
from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates.generation.matched_molecular_pair_generation import (
    MATCHED_MOLECULAR_PAIR_GENERATION_PROMPT,
)
from graphgen.utils import logger


class MatchedMolecularPairGenerator(BaseGenerator):
    """
    Generates a SAR QA from a matched molecular pair (MMP): two molecules that
    differ at exactly one structural position.  Explains the mechanistic reason
    for the delta in logD (or other property).

    Requires KG2 with include_edges: true so that MMP/Tanimoto edges are loaded,
    and partitions with min_units_per_community=2, max_units_per_community=2
    so each batch is exactly one pair of molecules.
    """

    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        nodes, edges = batch
        context = ""
        for node in nodes:
            desc = node[1].get("description") or node[1].get("content", "")
            context += f"- {node[0]}: {desc}\n"
        for edge in edges:
            desc = edge[2].get("description") or edge[2].get("content", f"{edge[0]} -> {edge[1]}")
            tanimoto = edge[2].get("tanimoto", "")
            shared_fg = edge[2].get("shared_fg", "")
            scaffold = edge[2].get("scaffold", "")
            extra = ""
            if tanimoto:
                extra += f" | tanimoto: {tanimoto}"
            if shared_fg:
                extra += f" | shared_fg: {shared_fg}"
            if scaffold:
                extra += f" | scaffold: {scaffold}"
            context += f"  relationship: {edge[0]} -- {edge[1]}: {desc}{extra}\n"
        prompt = MATCHED_MOLECULAR_PAIR_GENERATION_PROMPT["en"].format(context=context)
        return prompt

    @staticmethod
    def parse_response(response: str) -> list[dict]:
        question_match = re.search(r"<question>(.*?)</question>", response, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

        if question_match and answer_match:
            question = question_match.group(1).strip().strip('"').strip("'")
            answer = answer_match.group(1).strip().strip('"').strip("'")
        else:
            logger.warning("Failed to parse MMP response: %s", response)
            return []

        return [{"question": question, "answer": answer}]
