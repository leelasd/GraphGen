from typing import Any

from graphgen.models.generator.multi_choice_generator import MultiChoiceGenerator
from graphgen.templates.generation.chemistry_multi_choice_generation import (
    CHEMISTRY_MCQ_GENERATION_PROMPT,
)
from graphgen.utils import logger


class ChemistryMultiChoiceGenerator(MultiChoiceGenerator):
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
        prompt = CHEMISTRY_MCQ_GENERATION_PROMPT["en"].format(
            context=context,
            num_of_questions=self.num_of_questions,
        )
        return prompt
