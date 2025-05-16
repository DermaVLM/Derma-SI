import os
import json
import glob
import logging
from typing import List, Dict, Any

from case_structure import DermCase, save_cases

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_single_case(file_path: str) -> Dict[str, Any]:
    """Process a single case file and return it as a dictionary."""
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Try to parse as JSON
        try:
            data = json.loads(content)
            logger.info(f"Successfully loaded {file_path} as JSON")
            return data
        except json.JSONDecodeError:
            logger.warning(f"Could not parse {file_path} as JSON, treating as raw text")

            # If it's not valid JSON, try to parse the structure from raw text
            # This is a simplified example - you might need more robust parsing
            lines = content.strip().split("\n")

            case = {
                "case_id": os.path.basename(file_path).replace(".txt", ""),
                "disease": "Unknown",  # Default value
                "base_information": {},
                "questions_and_answers": [],
                "final_diagnosis": "",
                "treatment_of_choice": "",
                "any_important_additional_information": "",
                "origin": "human-generated",
            }

            # TODO: Very simple parsing logic, might need adapting to data format
            current_section = None
            current_qa = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("CASE ID:"):
                    case["case_id"] = line.replace("CASE ID:", "").strip()
                elif line.startswith("DISEASE:"):
                    case["disease"] = line.replace("DISEASE:", "").strip()
                elif line.startswith("FINAL DIAGNOSIS:"):
                    case["final_diagnosis"] = line.replace(
                        "FINAL DIAGNOSIS:", ""
                    ).strip()
                elif line.startswith("TREATMENT OF CHOICE:"):
                    case["treatment_of_choice"] = line.replace(
                        "TREATMENT OF CHOICE:", ""
                    ).strip()
                # More parsing rules as needed

            return case
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None


def prepare_seed_data(input_dir: str, output_path: str) -> None:
    """
    Prepare seed data from individual case files.

    Args:
        input_dir: Directory containing individual case files
        output_path: Path to save the combined seed data
    """
    # Find all JSON case files
    file_paths = glob.glob(os.path.join(input_dir, "*.json"))

    logger.info(f"Found {len(file_paths)} potential case files")

    # Process each file
    seed_cases = []
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                case_dict = json.load(f)

            # Ensure the case has a case_id
            if "case_id" not in case_dict:
                # Extract case ID from filename (e.g., '1-comodonal_acne.json' -> 'com_acne_001')
                filename = os.path.basename(file_path)
                case_number = filename.split("-")[0].strip()
                condition = (
                    filename.split("-")[1].split(".")[0].strip().replace(" ", "_")
                )
                case_id = f"{condition[:3]}_{case_number.zfill(3)}"
                case_dict["case_id"] = case_id

            # Ensure origin is set to human-generated
            case_dict["origin"] = "human-generated"

            case = DermCase.from_dict(case_dict)
            seed_cases.append(case)
            logger.info(f"Processed {file_path} -> {case.case_id}")

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

    logger.info(f"Successfully processed {len(seed_cases)} cases")

    # Save the combined seed data
    save_cases(seed_cases, output_path)
    logger.info(f"Saved {len(seed_cases)} seed cases to {output_path}")


def prepare_seed_from_json(json_file: str, output_path: str) -> None:
    """
    Prepare seed data from a single JSON file containing multiple cases.

    Args:
        json_file: Path to JSON file with cases
        output_path: Path to save the processed seed data
    """
    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        # Handle different possible structures
        if isinstance(data, list):
            cases = data
        elif isinstance(data, dict) and "cases" in data:
            cases = data["cases"]
        elif isinstance(data, dict):
            # Single case
            cases = [data]
        else:
            logger.error(f"Unrecognized data structure in {json_file}")
            return

        seed_cases = [DermCase.from_dict(case) for case in cases]
        logger.info(f"Extracted {len(seed_cases)} cases from {json_file}")

        # Save processed cases
        save_cases(seed_cases, output_path)
        logger.info(f"Saved {len(seed_cases)} seed cases to {output_path}")

    except Exception as e:
        logger.error(f"Error processing {json_file}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare seed data for DermSelfInstruct"
    )
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument(
        "--type",
        choices=["dir", "file"],
        default="file",
        help="Input type: directory of cases or single JSON file",
    )

    args = parser.parse_args()

    if args.type == "dir":
        prepare_seed_data(args.input, args.output)
    else:
        prepare_seed_from_json(args.input, args.output)
