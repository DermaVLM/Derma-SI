import logging
from typing import Dict, List, Union

LOGGER = logging.getLogger(__name__)


class ClassificationIdentifier:
    """
    A class for identifying whether instructions represent classification tasks,
    implementing Stage 2 of the Self-Instruct pipeline.
    """

    def __init__(self, task_generator):
        """
        Initialize with a TaskGenerator that provides the LLM interface.

        Parameters:
        - task_generator: TaskGenerator instance with generate_with_* methods
        """
        self.task_generator = task_generator

    def identify_classification_tasks(
        self, instructions: Union[List[str], str]
    ) -> Dict[str, bool]:
        """
        Identify which instructions represent classification tasks.

        Parameters:
        - instructions: Either a single instruction string or a list of instructions

        Returns:
        - Dictionary mapping instructions to boolean (True if classification)
        """
        if isinstance(instructions, str):
            instructions = [instructions]

        classification_results = {}

        for instruction in instructions:
            is_classification = self._identify_single_task(instruction)
            classification_results[instruction] = is_classification

        return classification_results

    def _identify_single_task(self, instruction: str) -> bool:
        """
        Identify if a single instruction represents a classification task.

        Parameters:
        - instruction: Instruction string to analyze

        Returns:
        - Boolean indicating if it's a classification task
        """
        # First, try a heuristic approach based on keywords
        if self._is_likely_classification_by_heuristics(instruction):
            return True

        # If heuristics are inconclusive, use the LLM
        prompt = self._create_classification_identification_prompt(instruction)

        if self.task_generator.generation_method == "local":
            response = self.task_generator._generate_with_local_model(
                prompt, max_tokens=5, temperature=0
            )
        else:
            response = self.task_generator._generate_with_openai(
                prompt, max_tokens=5, temperature=0
            )

        return self._parse_classification_response(response)

    def _is_likely_classification_by_heuristics(self, instruction: str) -> bool:
        """
        Use heuristics to identify likely classification tasks.

        Parameters:
        - instruction: Instruction to analyze

        Returns:
        - Boolean indicating if it's likely a classification task, or None if inconclusive
        """
        instruction_lower = instruction.lower()

        # Strong indicators of classification tasks
        classification_indicators = [
            "classify",
            "categorize",
            "choose between",
            "select from",
            "decide whether",
            "is it a",
            "determine if it is",
            "identify which category",
            "pick the correct",
            "multiple choice",
            "yes or no",
            "true or false",
        ]

        # Strong indicators of non-classification tasks
        non_classification_indicators = [
            "explain",
            "describe",
            "elaborate",
            "write about",
            "provide details",
            "discuss",
            "analyze",
            "summarize",
            "how would you",
        ]

        # Check for classification indicators
        for indicator in classification_indicators:
            if indicator in instruction_lower:
                return True

        # Check for non-classification indicators
        for indicator in non_classification_indicators:
            if indicator in instruction_lower:
                return False

        # If no strong indicators, return None (inconclusive)
        return None

    def _create_classification_identification_prompt(self, instruction: str) -> str:
        """
        Create a prompt for identifying if an instruction is a classification task.

        Parameters:
        - instruction: Instruction to analyze

        Returns:
        - Prompt string for the LLM
        """
        prompt = (
            "Determine whether this dermatology instruction represents a classification task "
            "with a limited set of possible answers or categories.\n\n"
            "Answer 'Yes' if the task involves classifying, categorizing, or selecting from specific options.\n"
            "Answer 'No' if the task requires open-ended explanation, description, or analysis.\n\n"
            "Examples:\n"
            "Instruction: Decide whether the given statement is a myth or a fact.\n"
            "Is it classification? Yes\n\n"
            "Instruction: Describe standard aftercare instructions following a cosmetic skin procedure.\n"
            "Is it classification? No\n\n"
            "Instruction: Classify whether the described lesion is likely benign or malignant.\n"
            "Is it classification? Yes\n\n"
            "Instruction: Provide a brief historical note about a dermatological disease or treatment.\n"
            "Is it classification? No\n\n"
            f"Instruction: {instruction}\n"
            "Is it classification? "
        )

        return prompt

    def _parse_classification_response(self, response: str) -> bool:
        """
        Parse the LLM's response to determine if it's a classification task.

        Parameters:
        - response: LLM output string

        Returns:
        - Boolean indicating if it's a classification task
        """
        return "yes" in response.lower()
