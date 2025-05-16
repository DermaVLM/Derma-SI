import os
import logging

from dotenv import load_dotenv
from derm_self_instruct import DermDatasetGenerator

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required")

    # Paths to input files
    kb_path = "mnt/processed_dermatology_kb.json"
    templates_path = "mnt/disease_seed_tasks_eg.json"

    # Create dataset generator
    generator = DermDatasetGenerator(
        api_key=api_key,
        gemini_model="gemini-2.0-flash",
        output_dir="mnt/output_SI_2",
    )

    # Generate the dataset
    generator.generate_dataset(
        kb_path=kb_path,
        templates_path=templates_path,
        # None values indicate no limits:
        max_diseases=None,
        max_tasks_per_disease=None,
        skip_diseases=1010,
        checkpoint_path="mnt/output_SI_2/derm_dataset_intermediate_960_diseases.json",
    )

    logger.info("Dataset generation completed successfully")
