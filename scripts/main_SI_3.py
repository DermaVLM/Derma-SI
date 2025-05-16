import os
import json
import logging
from datetime import datetime

import dotenv
from derm_self_instruct import DermCaseSelfInstruct

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"derm_self_instruct_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main():
    dotenv.load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    seed_data_path = "mnt/SI_3/seed_data_anamnesis.json"
    output_dir = "mnt/SI_3/output"
    num_cases = 20000  # Number of new cases to generate
    num_human = 2  # Number of human-generated cases to use per prompt
    num_model = 2  # Number of model-generated cases to use per prompt
    gemini_model = "gemini-2.0-flash"
    save_intermediate = True

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)

    # Save the run configuration
    config = {
        "timestamp": timestamp,
        "seed_data": seed_data_path,
        "num_cases": num_cases,
        "num_human": num_human,
        "num_model": num_model,
        "gemini_model": gemini_model,
        "save_intermediate": save_intermediate,
    }

    with open(os.path.join(run_output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Starting Dermatology Self-Instruct run with config: {config}")

    # Initialize the pipeline
    pipeline = DermCaseSelfInstruct(
        api_key=api_key, output_dir=run_output_dir, gemini_model=gemini_model
    )

    # Run the pipeline
    all_cases = pipeline.run(
        seed_data_path=seed_data_path,
        output_path=os.path.join(run_output_dir, "all_cases.json"),
        num_cases=num_cases,
        num_human=num_human,
        num_model=num_model,
        save_intermediate=save_intermediate,
    )

    # Analyze and log results
    if all_cases:
        human_cases = [case for case in all_cases if case.origin == "human-generated"]
        model_cases = [case for case in all_cases if case.origin == "model-generated"]

        logger.info(f"Pipeline completed successfully")
        logger.info(f"Total cases: {len(all_cases)}")
        logger.info(f"Human-generated cases: {len(human_cases)}")
        logger.info(f"Model-generated cases: {len(model_cases)}")

        # Create a summary report
        summary = {
            "timestamp": timestamp,
            "total_cases": len(all_cases),
            "human_cases": len(human_cases),
            "model_cases": len(model_cases),
            "config": config,
            "diseases": list({case.disease for case in all_cases}),
        }

        with open(os.path.join(run_output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Results saved to {run_output_dir}")
    else:
        logger.error("Pipeline failed to generate cases")


if __name__ == "__main__":
    main()
