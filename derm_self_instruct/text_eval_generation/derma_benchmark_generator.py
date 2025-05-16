import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from uuid import uuid4

from google import genai
from google.genai import types as genai_types
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DermatologyBenchmarkGenerator:
    """Generator for dermatology benchmark questions using Gemini."""

    def __init__(
        self,
        api_key: str,
        output_dir: str = "benchmark_output",
        gemini_model: str = "gemini-2.5-pro-exp-03-25",
        temperature: float = 0.2,
    ):
        """
        Initialize the Dermatology Benchmark Generator.

        Args:
            api_key: Gemini API key
            output_dir: Directory to save outputs
            gemini_model: Gemini model to use
            temperature: Temperature for generation (lower for more factual content)
        """
        self.output_dir = output_dir
        self.gemini_model = gemini_model
        self.temperature = temperature

        # Initialize Gemini client
        self.genai_client = genai.Client(api_key=api_key)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(
            f"Initialized DermatologyBenchmarkGenerator with model {gemini_model}"
        )

    def generate_benchmark_question(self, prompt: str) -> Optional[str]:
        """
        Generate a benchmark question using Gemini API.

        Args:
            prompt: The formatted prompt to send to Gemini

        Returns:
            The raw generated text or None if generation failed
        """
        try:
            # Create configuration for generation
            config = genai_types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=8192,
                top_p=0.95,
            )

            # Generate content
            response = self.genai_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=config,
            )

            # Extract the generated text
            generated_text = response.text.strip()

            # Log a preview of the generated text
            logger.info(f"Generated text preview: {generated_text[:100]}...")

            return generated_text

        except Exception as e:
            logger.error(f"Error generating benchmark question: {e}")
            return None

    def run_generation(
        self,
        prompts_file: str,
        save_interval: int = 300,
        max_questions: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate benchmark questions from the provided prompts.

        Args:
            prompts_file: Path to the JSON file containing prompts
            save_interval: Number of questions to generate before saving
            max_questions: Maximum number of questions to generate (None for all)

        Returns:
            List of generated benchmark questions with their prompts and responses
        """
        # Load prompts
        logger.info(f"Loading prompts from {prompts_file}")
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts_data = json.load(f)

        prompts = prompts_data.get("prompts", [])
        logger.info(f"Loaded {len(prompts)} prompts")

        # Limit number of questions if specified
        if max_questions is not None:
            prompts = prompts[:max_questions]
            logger.info(f"Limited to generating {max_questions} questions")

        # Initialize results storage
        results = []

        # Generate questions
        logger.info(f"Generating {len(prompts)} benchmark questions")
        progress_bar = tqdm(total=len(prompts))

        for i, prompt_item in enumerate(prompts):
            prompt_text = prompt_item.get("prompt", "")
            metadata = prompt_item.get("metadata", {})

            # Generate benchmark question
            generated_text = self.generate_benchmark_question(prompt_text)

            if generated_text:
                # Create result entry with unique ID
                result = {
                    "question_id": f"derm_bench_{uuid4().hex[:8]}",
                    "prompt": prompt_text,
                    "response": generated_text,
                    "metadata": metadata,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

                # Add to results
                results.append(result)

                logger.info(
                    f"Generated question {i+1}/{len(prompts)}: {result['question_id']}"
                )

                # Save intermediate results every save_interval questions
                if (i + 1) % save_interval == 0:
                    self._save_intermediate_results(results, i + 1)
            else:
                logger.warning(f"Failed to generate question {i+1}/{len(prompts)}")

            progress_bar.update(1)

        progress_bar.close()

        # Save final results
        self._save_final_results(results)

        return results

    def _save_intermediate_results(self, results: List[Dict[str, Any]], count: int):
        """Save intermediate results to a JSON file."""
        intermediate_path = os.path.join(
            self.output_dir, f"benchmark_questions_intermediate_{count}.json"
        )

        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump(
                {"generated_count": len(results), "results": results}, f, indent=2
            )

        logger.info(
            f"Saved intermediate results ({len(results)} questions) to {intermediate_path}"
        )

    def _save_final_results(self, results: List[Dict[str, Any]]):
        """Save final results to a JSON file."""
        output_path = os.path.join(
            self.output_dir, f"benchmark_questions_final_{len(results)}.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"total_questions": len(results), "results": results}, f, indent=2
            )

        logger.info(f"Saved final results ({len(results)} questions) to {output_path}")

        # Save statistics
        self._generate_statistics(results)

    def _generate_statistics(self, results: List[Dict[str, Any]]):
        """Generate and save statistics about the benchmark questions."""
        stats = {
            "total_questions": len(results),
            "categories": {},
            "difficulties": {},
            "question_types": {},
        }

        # Collect statistics
        for result in results:
            metadata = result.get("metadata", {})

            # Count categories
            category = metadata.get("category", "unknown")
            stats["categories"][category] = stats["categories"].get(category, 0) + 1

            # Count difficulties
            difficulty = metadata.get("difficulty", "unknown")
            stats["difficulties"][difficulty] = (
                stats["difficulties"].get(difficulty, 0) + 1
            )

            # Count question types
            question_type = metadata.get("question_type", "unknown")
            stats["question_types"][question_type] = (
                stats["question_types"].get(question_type, 0) + 1
            )

        # Save statistics
        stats_path = os.path.join(self.output_dir, "benchmark_statistics.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved benchmark statistics to {stats_path}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        exit(1)

    generator = DermatologyBenchmarkGenerator(api_key=api_key)

    # Run generation
    generator.run_generation(
        prompts_file="mnt/text_benchmark_prompts.json",
        save_interval=25,
        max_questions=None,
    )
