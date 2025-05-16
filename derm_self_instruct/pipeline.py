import os
import sys
import json
import logging
import time
from tqdm import tqdm
from typing import Dict, List, Any, Tuple, Optional

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from derm_self_instruct import (
    load_seed_data,
    save_task_pool,
    TaskGenerator,
    filter_tasks,
)

LOGGER = logging.getLogger(__name__)


class SelfInstructPipeline:
    """
    Main pipeline class that integrates all components of the Self-Instruct process
    for dermatology data generation.
    """

    def __init__(
        self,
        generation_method="gemini",
        device=0,
        api_key=None,
        gemini_model="gemini-2.0-flash",
        output_dir="mnt",
        similarity_threshold=0.7,
        save_intermediate=True,
        save_raw_outputs=True,
        log_level=logging.INFO,
    ):
        """
        Initialize the Self-Instruct pipeline with multiple model support.

        Parameters:
        - generation_method: "local", "openai", or "gemini"
        - device: GPU device index for local models
        - api_key: API key for OpenAI or Gemini
        - gemini_model: Gemini model name to use (default: "gemini-2.0-flash")
        - output_dir: Directory for saving data and logs
        - similarity_threshold: Threshold for filtering similar tasks
        - save_intermediate: Whether to save intermediate results
        - save_raw_outputs: Whether to save raw model outputs for debugging
        - log_level: Logging level
        """
        self.output_dir = output_dir
        self.similarity_threshold = similarity_threshold
        self.save_intermediate = save_intermediate
        self.save_raw_outputs = save_raw_outputs

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging
        setup_logging(output_dir, log_level, save_raw_outputs)

        # Initialize components
        self.task_generator = TaskGenerator(
            generation_method=generation_method,
            device=device,
            api_key=api_key,
            gemini_model=gemini_model,
        )

    def run(
        self,
        seed_data_path: str,
        output_path: str,
        target_size: int,
        batch_size: int = 8,
    ):
        """
        Run the complete Self-Instruct pipeline to generate a dataset.

        Parameters:
        - seed_data_path: Path to the seed data file
        - output_path: Path to save the final dataset
        - target_size: Target number of tasks to generate
        - batch_size: Number of instructions to generate per iteration

        Returns:
        - Final task pool
        """
        # Load seed data
        LOGGER.info(f"Loading seed data from {seed_data_path}")
        task_pool = load_seed_data(seed_data_path)

        if not task_pool:
            LOGGER.error("No seed data found. Exiting.")
            return None

        LOGGER.info(f"Loaded {len(task_pool)} seed tasks")

        # Start iterative generation
        current_size = len(task_pool)
        iteration = 0

        progress_bar = tqdm(total=target_size, initial=current_size)
        progress_bar.set_description("Generating tasks")

        while current_size < target_size:
            iteration += 1
            LOGGER.info(f"Starting iteration {iteration}")

            # Run one iteration of the pipeline
            new_tasks = self._run_iteration(
                task_pool=task_pool,
                iteration=iteration,
                batch_size=batch_size,
            )

            # Add filtered tasks to the pool
            task_pool.extend(new_tasks)
            current_size = len(task_pool)

            # Update progress
            progress_bar.update(len(new_tasks))
            progress_bar.set_description(
                f"Iteration {iteration}: {current_size}/{target_size} tasks"
            )

            # Save intermediate task pool
            if self.save_intermediate and iteration % 5 == 0:
                intermediate_path = os.path.join(
                    self.output_dir, f"task_pool_iter_{iteration}.json"
                )
                save_task_pool(intermediate_path, task_pool)
                LOGGER.info(f"Saved intermediate task pool to {intermediate_path}")

        # Save final task pool
        LOGGER.info(
            f"Saving final task pool with {len(task_pool)} tasks to {output_path}"
        )
        save_task_pool(output_path, task_pool)
        progress_bar.close()

        # Analyze the dataset
        self._analyze_dataset(task_pool)

        return task_pool

    def _run_iteration(
        self,
        task_pool: List[Dict[str, Any]],
        iteration: int,
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Run one iteration of the Self-Instruct pipeline.

        Parameters:
        - task_pool: Current pool of tasks
        - iteration: Current iteration number
        - batch_size: Number of instructions to generate

        Returns:
        - List of new filtered tasks
        """
        # Stage 1+2: Generate instructions with classification status in one step
        LOGGER.info("Stage 1+2: Generating instructions with classification status")

        # Generate instructions
        instructions_with_classification = (
            self.task_generator.generate_instructions_with_classification(
                task_pool, num_human=4, num_model=4, batch_size=batch_size
            )
        )

        # Log results
        classification_count = sum(
            1 for v in instructions_with_classification.values() if v
        )
        LOGGER.info(
            f"Generated {len(instructions_with_classification)} instructions "
            f"({classification_count} classification, "
            f"{len(instructions_with_classification) - classification_count} non-classification)"
        )

        # Save intermediate results
        if self.save_intermediate:
            instructions_path = os.path.join(
                self.output_dir,
                f"iter_{iteration}_instructions_with_classification.json",
            )
            self._save_json(
                instructions_path,
                {
                    "instructions_with_classification": {
                        k: str(v) for k, v in instructions_with_classification.items()
                    }
                },
            )
            LOGGER.info(f"Saved instructions to {instructions_path}")

        # Stage 3: Generate instances for each instruction
        LOGGER.info("Stage 3: Generating instances")
        tasks_with_instances = self.task_generator.generate_instances(
            instructions_with_classification
        )

        # Log results
        LOGGER.info(f"Generated instances for {len(tasks_with_instances)} tasks")

        # Save intermediate results
        if self.save_intermediate:
            instances_path = os.path.join(
                self.output_dir, f"iter_{iteration}_with_instances.json"
            )
            self._save_json(
                instances_path,
                {"tasks": tasks_with_instances},
            )
            LOGGER.info(f"Saved tasks with instances to {instances_path}")

        # Stage 4: Filter tasks
        LOGGER.info("Stage 4: Filtering tasks")
        filtered_tasks = filter_tasks(
            task_pool, tasks_with_instances, self.similarity_threshold
        )

        # Log results
        LOGGER.info(
            f"Filtered down to {len(filtered_tasks)} tasks "
            f"(filtered out {len(tasks_with_instances) - len(filtered_tasks)} tasks)"
        )

        # Save intermediate results
        if self.save_intermediate:
            filtered_path = os.path.join(
                self.output_dir, f"iter_{iteration}_filtered.json"
            )
            self._save_json(
                filtered_path,
                {"filtered_tasks": filtered_tasks},
            )
            LOGGER.info(f"Saved filtered tasks to {filtered_path}")

        return filtered_tasks

    def _analyze_dataset(self, task_pool: List[Dict[str, Any]]):
        """
        Analyze the generated dataset and save statistics.

        Parameters:
        - task_pool: The complete task pool to analyze
        """
        # Count task types
        classification_count = sum(
            1 for task in task_pool if task.get("is_classification", False)
        )
        non_classification_count = len(task_pool) - classification_count

        # Count tasks with non-empty input
        tasks_with_input = sum(
            1
            for task in task_pool
            for instance in task.get("instances", [])
            if instance.get("input", "").strip()
        )

        # Calculate average lengths
        avg_instruction_len = sum(
            len(task.get("instruction", "")) for task in task_pool
        ) / max(1, len(task_pool))

        all_instances = [
            instance for task in task_pool for instance in task.get("instances", [])
        ]

        non_empty_inputs = [
            instance.get("input", "")
            for instance in all_instances
            if instance.get("input", "").strip()
        ]

        avg_input_len = sum(len(input_text) for input_text in non_empty_inputs) / max(
            1, len(non_empty_inputs)
        )

        avg_output_len = sum(
            len(instance.get("output", "")) for instance in all_instances
        ) / max(1, len(all_instances))

        # Origin statistics
        origin_counts = {}
        for task in task_pool:
            origin = task.get("origin", "unknown")
            origin_counts[origin] = origin_counts.get(origin, 0) + 1

        # Create analysis object
        analysis = {
            "total_tasks": len(task_pool),
            "classification_tasks": classification_count,
            "non_classification_tasks": non_classification_count,
            "tasks_with_input": tasks_with_input,
            "tasks_without_input": len(task_pool) - tasks_with_input,
            "avg_instruction_length": round(avg_instruction_len, 1),
            "avg_input_length": round(avg_input_len, 1),
            "avg_output_length": round(avg_output_len, 1),
            "origin_distribution": origin_counts,
            "llm_provider": self.task_generator.generation_method,
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        LOGGER.info(f"Dataset Analysis: {json.dumps(analysis, indent=2)}")

        # Save analysis
        analysis_path = os.path.join(self.output_dir, "dataset_analysis.json")
        self._save_json(analysis_path, analysis)
        LOGGER.info(f"Saved dataset analysis to {analysis_path}")

    def _save_json(self, path: str, data: Any):
        """
        Save data as JSON.

        Parameters:
        - path: Path to save the JSON file
        - data: Data to save
        """
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            LOGGER.error(f"Error saving JSON to {path}: {e}")


def setup_logging(
    output_dir: str, log_level: int = logging.INFO, save_raw_outputs: bool = True
):
    """
    Set up logging with improved configuration for better debugging.

    Parameters:
    - output_dir: Directory for log files
    - log_level: Logging level (default: INFO)
    - save_raw_outputs: Whether to save raw model outputs for debugging (default: True)

    Returns:
    - Logger instance
    """
    # Create logging directory
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create directory for saving raw outputs if needed
    if save_raw_outputs:
        raw_outputs_dir = os.path.join(output_dir, "raw_outputs")
        os.makedirs(raw_outputs_dir, exist_ok=True)

    # Generate timestamp for log files
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"self_instruct_{timestamp}.log")
    )
    file_handler.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log basic info
    logger.info(f"Logging initialized with level {logging.getLevelName(log_level)}")
    logger.info(f"Log files will be saved to: {log_dir}")
    if save_raw_outputs:
        logger.info(f"Raw model outputs will be saved to: {raw_outputs_dir}")

    return logger


def save_raw_output(
    output_dir: str, model_type: str, stage: str, iteration: int, raw_output: str
):
    """
    Save raw model output to file for debugging purposes.

    Parameters:
    - output_dir: Base directory for saving outputs
    - model_type: Type of model used (openai, gemini, local)
    - stage: Pipeline stage (instructions, instances, etc.)
    - iteration: Current iteration number
    - raw_output: Raw text output from the model
    """
    raw_outputs_dir = os.path.join(output_dir, "raw_outputs")
    os.makedirs(raw_outputs_dir, exist_ok=True)

    filename = f"{model_type}_{stage}_iter{iteration}_{time.strftime('%H%M%S')}.txt"
    filepath = os.path.join(raw_outputs_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(raw_output)

    logging.getLogger().info(
        f"Saved raw {model_type} output for {stage} (iteration {iteration}) to {filepath}"
    )


class OutputFormatter:
    """
    Helper class to format model outputs for better logging and debugging.
    """

    @staticmethod
    def format_for_log(text: str, max_length: int = 100) -> str:
        """
        Format text for logging by truncating and cleaning.

        Parameters:
        - text: Text to format
        - max_length: Maximum length to show in logs

        Returns:
        - Formatted text string
        """
        if not text:
            return "[EMPTY]"

        # Replace newlines with visible markers for logging
        formatted = text.replace("\n", "\\n")

        # Truncate if needed
        if len(formatted) > max_length:
            return formatted[:max_length] + "..."

        return formatted

    @staticmethod
    def summarize_parsing_results(parsed_data, data_type: str) -> str:
        """
        Create a summary of parsing results for logging.

        Parameters:
        - parsed_data: The parsed data (instructions or instances)
        - data_type: Type of data being parsed

        Returns:
        - Summary string
        """
        if isinstance(parsed_data, dict):
            count = len(parsed_data)
            if data_type == "instructions":
                classification_count = sum(1 for v in parsed_data.values() if v)
                return f"Parsed {count} instructions ({classification_count} classification, {count - classification_count} non-classification)"
        elif isinstance(parsed_data, list):
            return f"Parsed {len(parsed_data)} {data_type}"

        return f"No {data_type} successfully parsed"
