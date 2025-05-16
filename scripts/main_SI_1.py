import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from derm_self_instruct import SelfInstructPipeline

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        print("Error: Gemini API key not found.")
        sys.exit(1)

    start_time = datetime.now()

    # Parameters for the pipeline
    model = "gemini"
    device = 0
    gemini_model = "gemini-2.0-flash"

    target_size = 100000
    batch_size = 8
    similarity_threshold = 0.95

    # Create output directory
    output_dir = "mnt"
    os.makedirs(output_dir, exist_ok=True)

    # Check for seed data
    # Seed data (input) file
    seed_data_path = os.path.join(output_dir, "task_pool_checkpoint_filtered.json")
    # Output dataset
    output_path = os.path.join(
        output_dir,
        f"dermatology_SI_dataset_{start_time.strftime('%Y%m%d_%H%M%S')}.json",
    )

    if not os.path.exists(seed_data_path):
        print(f"Error: Seed data file not found at {seed_data_path}")
        sys.exit(1)

    pipeline = SelfInstructPipeline(
        generation_method=model,
        device=device,
        api_key=api_key,
        gemini_model=gemini_model,
        output_dir=output_dir,
        similarity_threshold=similarity_threshold,
    )

    # Run the pipeline
    model_description = {
        "openai": "OpenAI",
        "gemini": f"Google Gemini ({gemini_model})",
        "local": "Local model",
    }

    print(f"Starting Self-Instruct pipeline using {model_description[model]}")
    print(f"Target size: {target_size} tasks")

    task_pool = pipeline.run(
        seed_data_path=seed_data_path,
        output_path=output_path,
        target_size=target_size,
        batch_size=batch_size,
    )

    end_time = datetime.now()
    duration = end_time - start_time

    # Print summary
    print("\n" + "=" * 60)
    print(f"Dataset generation complete!")
    print(f"Model: {model_description[model]}")
    print(f"Generated {len(task_pool)} tasks in {duration}")
    print(f"Output saved to: {output_path}")
    print("=" * 60)
