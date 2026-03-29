import re
from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates.generation.pairwise_preference_generation import (
    PAIRWISE_PREFERENCE_GENERATION_PROMPT,
)
from graphgen.utils import logger


class PairwisePreferenceGenerator(BaseGenerator):
    """
    Generates a pairwise comparison QA: given two molecules, which is preferred
    for a chemical property (e.g. logD, solubility) and why.

    Requires partitions with min_units_per_community >= 2 so that at least
    two molecules appear in each batch.
    """

    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        nodes, edges = batch
        labels = {node[0]: f"Molecule {chr(65 + i)}" for i, node in enumerate(nodes)}
        context = ""
        for node in nodes:
            label = labels[node[0]]
            desc = node[1].get("description") or node[1].get("content", "")
            context += f"- {label}: {desc}\n"
        for edge in edges:
            src = labels.get(edge[0], edge[0])
            tgt = labels.get(edge[1], edge[1])
            desc = edge[2].get("description") or edge[2].get("content", "")
            context += f"  relationship: {src} -- {tgt}: {desc}\n"
        prompt = PAIRWISE_PREFERENCE_GENERATION_PROMPT["en"].format(context=context)
        return prompt

    @staticmethod
    def parse_response(response: str) -> list[dict]:
        question_match = re.search(r"<question>(.*?)</question>", response, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

        if question_match and answer_match:
            question = question_match.group(1).strip().strip('"').strip("'")
            answer = answer_match.group(1).strip().strip('"').strip("'")
        else:
            logger.warning("Failed to parse pairwise preference response: %s", response)
            return []

        return [{"question": question, "answer": answer}]
