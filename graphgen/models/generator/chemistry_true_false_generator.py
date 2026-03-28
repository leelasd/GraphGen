from typing import Any

from graphgen.models.generator.true_false_generator import TrueFalseGenerator
from graphgen.templates.generation.chemistry_true_false_generation import (
    CHEMISTRY_TF_GENERATION_PROMPT,
)
from graphgen.utils import logger


class ChemistryTrueFalseGenerator(TrueFalseGenerator):
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
        prompt = CHEMISTRY_TF_GENERATION_PROMPT["en"].format(
            context=context,
            num_of_questions=self.num_of_questions,
        )
        return prompt
