import json
import logging
import os
import re

from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
from uuid import uuid4

import torch
from google import genai
from google.genai import types as genai_types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)


class DermDatasetGenerator:
    """Generate a dermatology dataset using self-instruct methodology"""

    def __init__(
        self,
        api_key: str = None,
        gemini_model: str = "gemini-2.0-pro",
        output_dir: str = "output",
    ):
        """
        Initialize the DermDatasetGenerator

        Parameters:
        - api_key: str, API key for Gemini
        - gemini_model: str, Gemini model to use
        - output_dir: str, Directory to save generated datasets
        """
        self.gemini_model = gemini_model
        self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Gemini client
        if not api_key:
            raise ValueError("API key must be provided for Gemini API")
        self.gemini_client = genai.Client(api_key=api_key)

    def load_knowledge_base(self, kb_path: str) -> List[Dict[str, Any]]:
        """
        Load the dermatology knowledge base

        Parameters:
        - kb_path: str, Path to the knowledge base JSON file

        Returns:
        - List of disease dictionaries
        """
        LOGGER.info(f"Loading knowledge base from {kb_path}")
        with open(kb_path, "r", encoding="utf-8") as f:
            kb = json.load(f)
        LOGGER.info(f"Loaded {len(kb)} diseases from knowledge base")
        return kb

    def load_task_templates(self, templates_path: str) -> List[Dict[str, Any]]:
        """
        Load the task templates

        Parameters:
        - templates_path: str, Path to the task templates JSON file

        Returns:
        - List of task template dictionaries
        """
        LOGGER.info(f"Loading task templates from {templates_path}")
        with open(templates_path, "r", encoding="utf-8") as f:
            templates_data = json.load(f)
            templates = templates_data.get("seed_tasks_for_disease", [])
        LOGGER.info(f"Loaded {len(templates)} task templates")
        return templates

    def load_checkpoint(self, checkpoint_path: str) -> List[Dict[str, Any]]:
        """
        Load an existing dataset from a checkpoint file

        Parameters:
        - checkpoint_path: str, Path to the checkpoint file

        Returns:
        - List of examples from the checkpoint
        """
        LOGGER.info(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        LOGGER.info(f"Loaded {len(dataset)} examples from checkpoint")
        return dataset

    def prepare_disease_context(self, disease_entry: Dict[str, Any]) -> str:
        """
        Prepare the disease context by concatenating all sections of knowledge

        Parameters:
        - disease_entry: Dict, Entry from the knowledge base for a specific disease

        Returns:
        - str: Concatenated disease information
        """
        disease_name = disease_entry.get("name", "Unknown Disease")
        sections = disease_entry.get("sections", {})

        # Concatenate all sections into a single context string
        context_parts = [f"# {disease_name}"]

        for section_name, section_content in sections.items():
            # Format the section name for better readability
            formatted_section = section_name.replace("_", " ").title()
            context_parts.append(f"## {formatted_section}")
            context_parts.append(section_content.strip())

        return "\n\n".join(context_parts)

    def generate_task_examples(
        self,
        disease_entry: Dict[str, Any],
        task_templates: List[Dict[str, Any]],
        max_tasks: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate task examples for a specific disease using the task templates

        Parameters:
        - disease_entry: Dict, Entry from the knowledge base for a specific disease
        - task_templates: List, List of task template dictionaries
        - max_tasks: int, Maximum number of tasks to generate (None for all)

        Returns:
        - List of generated task examples
        """
        disease_name = disease_entry.get("name", "Unknown Disease")
        LOGGER.info(f"Generating tasks for disease: {disease_name}")

        # Prepare the disease context
        disease_context = self.prepare_disease_context(disease_entry)

        # Limit the number of tasks if specified
        if max_tasks is not None and max_tasks > 0:
            templates_to_use = task_templates[:max_tasks]
        else:
            templates_to_use = task_templates

        # Generate examples for each template
        examples = []
        for i, template in enumerate(
            tqdm(templates_to_use, desc=f"Tasks for {disease_name}", leave=False)
        ):
            example = self._generate_task_example(
                disease_name, disease_context, template
            )
            examples.append(example)

            # Log progress every 10 tasks
            if (i + 1) % 10 == 0 or (i + 1) == len(templates_to_use):
                LOGGER.info(
                    f"Generated {i+1}/{len(templates_to_use)} tasks for {disease_name}"
                )

        return examples

    def _generate_task_example(
        self,
        disease_name: str,
        disease_context: str,
        template: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a single task example for a specific disease

        Parameters:
        - disease_name: str, Name of the disease
        - disease_context: str, Concatenated disease information
        - template: Dict, Task template dictionary

        Returns:
        - Generated task example dictionary
        """
        task_id = template.get("id", "unknown")
        title = template.get("title", "Unknown Task")
        prompt_template = template.get("prompt", "")
        description = template.get("description", "")

        # Replace placeholder with disease name
        prompt = prompt_template.replace("<DISEASE_NAME>", disease_name)

        # Generate answer using the Gemini API
        answer = self._generate_answer(
            disease_name, disease_context, prompt, title, description
        )

        # Create example object
        example = {
            "id": f"{task_id}_{uuid4().hex[:8]}",
            "disease_name": disease_name,
            "task_title": title,
            "task_description": description,
            "question": prompt,
            "answer": answer,
        }

        return example

    def _generate_answer(
        self,
        disease_name: str,
        disease_context: str,
        question: str,
        task_title: str,
        task_description: str,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> str:
        """
        Generate an answer for a specific question about a disease

        Parameters:
        - disease_name: str, Name of the disease
        - disease_context: str, Concatenated disease information
        - question: str, Question to answer
        - task_title: str, Title of the task
        - task_description: str, Description of the task
        - temperature: float, Temperature parameter for generation
        - max_tokens: int, Maximum number of tokens to generate

        Returns:
        - str: Generated answer
        """
        try:
            # Create the prompt for Gemini
            prompt = f"""You are a dermatology expert specializing in skin diseases and conditions, answering questions about {disease_name}.

CONTEXT INFORMATION (for your knowledge only - DO NOT reference this context in your answer):
{disease_context}

TASK: {task_title}
TASK DESCRIPTION: {task_description}

QUESTION: {question}

Answer the question as an authoritative dermatology expert. Write in a direct, confident style without referencing any source materials or phrases like "based on the information" or "according to the context". Never indicate that you're using specific information provided to you. 

Instead, speak authoritatively as if you personally have expertise on {disease_name}. If the context doesn't contain sufficient information for certain aspects, use general dermatological knowledge to provide a complete answer.

Your answer should be:
1. Factually accurate and clinically sound
2. Well-structured with appropriate headings and organization
3. Comprehensive, covering all relevant aspects of the question
4. Practical for clinical application

ANSWER:"""

            config = genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            # Generate content
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=config,
            )

            # Extract the answer
            answer = response.text.strip()
            return answer

        except Exception as e:
            LOGGER.error(
                f"Error generating answer for {disease_name}, task {task_title}: {e}"
            )
            return f"[Error generating answer: {str(e)}]"

    def generate_dataset(
        self,
        kb_path: str,
        templates_path: str,
        max_diseases: int = None,
        max_tasks_per_disease: int = None,
        skip_diseases: int = 0,
        checkpoint_path: str = None,
    ) -> None:
        """
        Generate the complete dataset

        Parameters:
        - kb_path: str, Path to the knowledge base JSON file
        - templates_path: str, Path to the task templates JSON file
        - max_diseases: int, Maximum number of diseases to process (None for all)
        - max_tasks_per_disease: int, Maximum number of tasks per disease (None for all)
        - skip_diseases: int, Number of diseases to skip from the beginning (for continuing)
        - checkpoint_path: str, Path to a checkpoint file to continue from (optional)
        """
        # Load knowledge base and task templates
        kb = self.load_knowledge_base(kb_path)
        task_templates = self.load_task_templates(templates_path)

        # Initialize dataset from checkpoint if provided
        dataset = []
        if checkpoint_path and os.path.exists(checkpoint_path):
            dataset = self.load_checkpoint(checkpoint_path)
            LOGGER.info(
                f"Continuing from checkpoint with {len(dataset)} existing examples"
            )

        # Skip the specified number of diseases
        if skip_diseases > 0:
            if skip_diseases >= len(kb):
                LOGGER.warning(
                    f"Skip diseases count ({skip_diseases}) exceeds knowledge base size ({len(kb)})"
                )
                return

            LOGGER.info(f"Skipping first {skip_diseases} diseases")
            kb = kb[skip_diseases:]

        # Limit the number of diseases if specified
        if max_diseases is not None and max_diseases > 0:
            kb = kb[:max_diseases]

        # Calculate actual disease count for logging
        starting_disease_num = skip_diseases

        # Process each disease
        for i, disease in enumerate(tqdm(kb, desc="Processing diseases")):
            actual_disease_num = starting_disease_num + i + 1

            disease_examples = self.generate_task_examples(
                disease, task_templates, max_tasks=max_tasks_per_disease
            )
            dataset.extend(disease_examples)

            # Log progress
            LOGGER.info(
                f"Processed disease {i+1}/{len(kb)} (overall: {actual_disease_num}): {disease.get('name')} - Generated {len(disease_examples)} examples"
            )

            # Save intermediate results every 10 diseases
            if (i + 1) % 10 == 0 or (i + 1) == len(kb):
                self._save_intermediate_dataset(dataset, actual_disease_num)

        # Save the final dataset
        self._save_dataset(dataset)

        LOGGER.info(f"Dataset generation complete. Total examples: {len(dataset)}")

    def _save_intermediate_dataset(
        self, dataset: List[Dict[str, Any]], disease_count: int
    ) -> None:
        """
        Save an intermediate version of the dataset

        Parameters:
        - dataset: List of generated examples
        - disease_count: Number of diseases processed so far
        """
        output_path = (
            self.output_dir / f"derm_dataset_intermediate_{disease_count}_diseases.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        LOGGER.info(
            f"Saved intermediate dataset with {len(dataset)} examples after processing {disease_count} diseases"
        )

    def _save_dataset(self, dataset: List[Dict[str, Any]]) -> None:
        """
        Save the final dataset

        Parameters:
        - dataset: List of generated examples
        """
        output_path = self.output_dir / "derm_dataset_final.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        LOGGER.info(
            f"Saved final dataset with {len(dataset)} examples to {output_path}"
        )
