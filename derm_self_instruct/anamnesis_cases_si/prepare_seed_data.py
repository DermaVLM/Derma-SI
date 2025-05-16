import os
import json
import logging
from case_structure import DermCase, save_cases

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_seed_data():
    input_dir = "mnt/SI_3/o1 - anamnesis cases"
    output_path = "mnt/SI_3/seed_data_anamnesis.json"

    # Find all JSON case files
    file_paths = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".json")
    ]

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

                # Handle filenames with more than one dash
                condition_part = "-".join(filename.split("-")[1:]).split(".")[0].strip()
                condition = condition_part.replace(" ", "_")

                # Create a case_id using the first 3 letters of the condition and the number
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


if __name__ == "__main__":
    prepare_seed_data()
