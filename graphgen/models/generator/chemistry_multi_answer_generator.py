from typing import Any

from graphgen.models.generator.multi_answer_generator import MultiAnswerGenerator
from graphgen.templates.generation.chemistry_multi_answer_generation import (
    CHEMISTRY_MAQ_GENERATION_PROMPT,
)
from graphgen.utils import logger


class ChemistryMultiAnswerGenerator(MultiAnswerGenerator):
    def build_prompt(
        self, batch: tuple[list[tuple[str, dict]], list[tuple[Any, Any, dict]]]
    ) -> str:
        nodes, edges = batch
        context = ""
        for node in nodes:
            desc = node[1].get("description") or node[1].get("content", "")
            context += f"- {node[0]}: {desc}\n"
        for edge in edges:
            desc = edge[2].get("description") or edge[2].get("content", f"{edge[0]} -> {edge[1]}")
            context += f"- {edge[0]} - {edge[1]}: {desc}\n"
        prompt = CHEMISTRY_MAQ_GENERATION_PROMPT["en"].format(
            context=context,
            num_of_questions=self.num_of_questions,
        )
        return prompt
