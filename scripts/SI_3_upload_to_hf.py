import os
import json
import logging
from datasets import Dataset
from huggingface_hub import HfApi, login

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_dataset(input_path):
    """
    Prepare the dataset for uploading to Hugging Face.

    Args:
        input_path: Path to JSON file containing all cases

    Returns:
        Dataset object ready for uploading
    """
    try:
        # Load JSON data
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} cases from {input_path}")

        # Convert the data structure to a format better suited for Hugging Face datasets
        dataset_dict = {
            "case_id": [],
            "disease": [],
            "age": [],
            "gender": [],
            "medical_history": [],
            "questions_and_answers": [],
            "final_diagnosis": [],
            "treatment": [],
            "additional_info": [],
            "origin": [],
        }

        for case in data:
            dataset_dict["case_id"].append(case.get("case_id", ""))
            dataset_dict["disease"].append(case.get("disease", ""))

            # Extract base information
            base_info = case.get("base_information", {})
            dataset_dict["age"].append(base_info.get("age", 0))
            dataset_dict["gender"].append(base_info.get("gender", ""))
            dataset_dict["medical_history"].append(
                base_info.get("other_relevant_history", "")
            )

            # Store questions and answers as JSON string
            dataset_dict["questions_and_answers"].append(
                json.dumps(case.get("questions_and_answers", []))
            )

            dataset_dict["final_diagnosis"].append(case.get("final_diagnosis", ""))
            dataset_dict["treatment"].append(case.get("treatment_of_choice", ""))
            dataset_dict["additional_info"].append(
                case.get("any_important_additional_information", "")
            )
            dataset_dict["origin"].append(case.get("origin", "human-generated"))

        # Create Hugging Face dataset
        dataset = Dataset.from_dict(dataset_dict)
        logger.info(f"Created dataset with {len(dataset)} examples")

        return dataset

    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        return None


def upload_to_huggingface(dataset, repo_name):
    """
    Upload the dataset to Hugging Face.

    Args:
        dataset: Dataset object
        repo_name: Name for the repository (e.g., 'username/dataset-name')
    """
    try:

        # Push dataset to the hub
        dataset.push_to_hub(repo_name)

        logger.info(f"Successfully uploaded dataset to {repo_name}")

        # Create a README file with dataset information
        api = HfApi()

        readme_content = f"""# Dermatology Case Studies Dataset

## Dataset Description

This dataset contains {len(dataset)} dermatology case studies in a question-answer format, 
simulating the interaction between a dermatologist and a patient. Each case includes detailed 
patient history, physical examination findings, diagnostic considerations, and treatment plans.

## Features

- `case_id`: Unique identifier for each case
- `disease`: The dermatological condition covered in the case
- `age`: Patient age
- `gender`: Patient gender
- `medical_history`: Relevant medical history information
- `questions_and_answers`: Structured Q&A between doctor and patient (stored as JSON)
- `final_diagnosis`: The confirmed diagnosis
- `treatment`: Recommended treatment approach
- `additional_info`: Any additional relevant information
- `origin`: Whether the case is human-generated or model-generated

## Usage

This dataset can be used for training medical question-answering systems, clinical decision support tools,
or educational resources for dermatology training.

## Citation

If you use this dataset in your research, please cite:

```
todo
```
"""

        # Upload README
        # api.upload_file(
        #    path_or_fileobj=readme_content.encode(),
        #    path_in_repo="README.md",
        #    repo_id=repo_name,
        # should spesifyt repo type too
        # )

        # logger.info(f"Created README for {repo_name}")

        # Provide a direct link to the dataset
        return f"https://huggingface.co/datasets/{repo_name}"

    except Exception as e:
        logger.error(f"Error uploading to Hugging Face: {e}")
        return None


def main():
    input_path = "mnt/SI_3/output/run_20250315_123633/generated_cases.json"
    repo_name = "DermaVLM/anamnesis_cases_1k"

    # Prepare dataset
    dataset = prepare_dataset(input_path)

    if dataset:
        # Upload to Hugging Face
        dataset_url = upload_to_huggingface(dataset, repo_name)

        if dataset_url:
            logger.info(f"dataset is now available at: {dataset_url}")
        else:
            logger.error("Failed to upload dataset to Hugging Face")
    else:
        logger.error("Failed to prepare dataset")


if __name__ == "__main__":
    main()
