import re
from typing import Any

from graphgen.bases import BaseGenerator
from graphgen.templates.generation.chemistry_atomic_generation import (
    CHEMISTRY_ATOMIC_GENERATION_PROMPT,
)
from graphgen.utils import logger


class ChemistryAtomicGenerator(BaseGenerator):
    @staticmethod
    def build_prompt(
        batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        nodes, edges = batch
        context = ""
        for node in nodes:
            desc = node[1].get("description") or node[1].get("content", "")
            context += f"- {desc}\n"
        for edge in edges:
            desc = edge[2].get("description") or edge[2].get("content", "")
            context += f"- {desc}\n"
        prompt = CHEMISTRY_ATOMIC_GENERATION_PROMPT["en"].format(context=context)
        return prompt

    @staticmethod
    def parse_response(response: str) -> list[dict]:
        question_match = re.search(r"<question>(.*?)</question>", response, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

        if question_match and answer_match:
            question = question_match.group(1).strip().strip('"').strip("'")
            answer = answer_match.group(1).strip().strip('"').strip("'")
        else:
            logger.warning("Failed to parse chemistry atomic response: %s", response)
            return []

        return [{"question": question, "answer": answer}]
