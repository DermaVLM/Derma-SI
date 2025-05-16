import json
import os
import logging
from typing import List, Dict, Any, Optional
from uuid import uuid4
from random import sample, shuffle

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DermCase:
    """Class to represent a dermatology case with standardized structure."""

    def __init__(
        self,
        case_id: str,
        disease: str,
        base_information: Dict[str, Any],
        questions_and_answers: List[Dict[str, Any]],
        final_diagnosis: str,
        treatment_of_choice: str,
        additional_info: Optional[str] = None,
        origin: str = "human-generated",
    ):
        self.case_id = case_id
        self.disease = disease
        self.base_information = base_information
        self.questions_and_answers = questions_and_answers
        self.final_diagnosis = final_diagnosis
        self.treatment_of_choice = treatment_of_choice
        self.additional_info = additional_info
        self.origin = origin

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DermCase":
        """Create a DermCase instance from a dictionary."""
        return cls(
            case_id=data.get("case_id", f"case_{uuid4().hex[:8]}"),
            disease=data.get("disease", ""),
            base_information=data.get("base_information", {}),
            questions_and_answers=data.get("questions_and_answers", []),
            final_diagnosis=data.get("final_diagnosis", ""),
            treatment_of_choice=data.get("treatment_of_choice", ""),
            additional_info=data.get("any_important_additional_information", ""),
            origin=data.get("origin", "human-generated"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the DermCase to a dictionary."""
        result = {
            "case_id": self.case_id,
            "disease": self.disease,
            "base_information": self.base_information,
            "questions_and_answers": self.questions_and_answers,
            "final_diagnosis": self.final_diagnosis,
            "treatment_of_choice": self.treatment_of_choice,
        }

        # Include origin as metadata for the pipeline, but not visible in the output format
        result["origin"] = self.origin

        return result

    def __str__(self) -> str:
        """String representation of the case."""
        return (
            f"DermCase(id={self.case_id}, disease={self.disease}, origin={self.origin})"
        )


def load_seed_cases(file_path: str) -> List[DermCase]:
    """Load seed cases from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different possible structures
        if isinstance(data, list):
            cases = data
        elif isinstance(data, dict) and "cases" in data:
            cases = data["cases"]
        else:
            cases = [data]  # Single case

        loaded_cases = [DermCase.from_dict(case) for case in cases]
        logger.info(f"Successfully loaded {len(loaded_cases)} cases from {file_path}")
        return loaded_cases
    except Exception as e:
        logger.error(f"Error loading seed cases from {file_path}: {e}")
        return []


def save_cases(cases: List[DermCase], output_path: str) -> None:
    """Save cases to a JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([case.to_dict() for case in cases], f, indent=2)
        logger.info(f"Saved {len(cases)} cases to {output_path}")
    except Exception as e:
        logger.error(f"Error saving cases: {e}")


def sample_cases(
    cases: List[DermCase], num_human: int, num_model: int
) -> List[List[DermCase]]:
    """
    Sample cases using a round-robin approach, ensuring a mix of human and model-generated cases.

    Args:
        cases: List of all available cases
        num_human: Number of human-generated cases to include per prompt
        num_model: Number of model-generated cases to include per prompt

    Returns:
        List of case groups, each containing a mix of human and model cases
    """
    human_cases = [case for case in cases if case.origin == "human-generated"]
    model_cases = [case for case in cases if case.origin == "model-generated"]

    # Make sure we have at least 100 model cases, not using same cases at first generations.
    if len(model_cases) < 100:
        logger.warning(
            f"Insufficient model generated cases ({len(model_cases)})."
            f" Using 0 model-generated cases."
        )
        num_model = 0

    # If we don't have any model cases, use only human cases
    if num_model == 0:
        logger.warning(
            "No model-generated cases available. Using only human-generated cases."
        )
        if len(human_cases) >= num_human + num_model:
            # If we have enough human cases, sample them
            shuffle(human_cases)
            return [[human_cases[i] for i in range(num_human + num_model)]]
        else:
            # If we don't have enough human cases, use all of them
            logger.warning(
                f"Not enough human cases. Using all {len(human_cases)} available."
            )
            return [human_cases]

    shuffle(human_cases)
    shuffle(model_cases)

    # Sample cases for each prompt
    sampled_prompts = []
    human_sample = sample(human_cases, min(num_human, len(human_cases)))
    model_sample = sample(model_cases, min(num_model, len(model_cases)))
    sampled_prompts.append(human_sample + model_sample)

    return sampled_prompts
