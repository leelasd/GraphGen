import re
from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates.generation.ranking_generation import RANKING_GENERATION_PROMPT
from graphgen.utils import logger


class RankingGenerator(BaseGenerator):
    """
    Generates a ranking QA: given N molecules, order them by a chemical property
    (e.g. logD) with mechanistic justification.

    Works best with partitions of min_units_per_community=3, max_units_per_community=5
    so that each batch contains enough molecules to rank meaningfully.
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
            context += f"  relationship: {edge[0]} -- {edge[1]}: {desc}\n"
        prompt = RANKING_GENERATION_PROMPT["en"].format(context=context)
        return prompt

    @staticmethod
    def parse_response(response: str) -> list[dict]:
        question_match = re.search(r"<question>(.*?)</question>", response, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

        if question_match and answer_match:
            question = question_match.group(1).strip().strip('"').strip("'")
            answer = answer_match.group(1).strip().strip('"').strip("'")
        else:
            logger.warning("Failed to parse ranking response: %s", response)
            return []

        return [{"question": question, "answer": answer}]
