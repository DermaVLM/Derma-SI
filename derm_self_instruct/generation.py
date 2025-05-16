import json
import re
from uuid import uuid4
import logging
from typing import Dict, List

import torch
import transformers
from google import genai
from google.genai import types as genai_types
from openai import OpenAI

from derm_self_instruct.sampling import sample_tasks_round_robin
from derm_self_instruct.config import MODEL_ID_LLM

LOGGER = logging.getLogger(__name__)


class TaskGenerator:
    def __init__(
        self,
        generation_method="openai",
        device=0,
        api_key=None,
        gemini_model="gemini-2.0-flash",
    ):
        """
        Initialize the TaskGenerator class with support for multiple LLM providers.

        Parameters:
        - generation_method: str, "local", "openai", or "gemini"
        - device: int, GPU device index for local LLM
        - api_key: str, API key for OpenAI or Gemini
        - gemini_model: str, Gemini model to use (default: "gemini-2.0-flash")
        """
        self.generation_method = generation_method
        self.device = device
        self.gemini_model = gemini_model

        if generation_method == "local":
            self.pipeline = self._get_local_pipeline()
            # Ensure api_key is None for local model, as it's not needed so there is no confusion
            if api_key:
                LOGGER.warning(
                    "API key is not needed for local model and will be ignored."
                )
            self.openai_client = None
            self.gemini_client = None

        elif generation_method == "openai":
            if not api_key:
                raise ValueError("API key must be provided for OpenAI API.")
            self.openai_client = OpenAI(api_key=api_key)
            self.pipeline = None
            self.gemini_client = None

        elif generation_method == "gemini":
            self.gemini_client = genai.Client(api_key=api_key)
            self.pipeline = None
            self.openai_client = None

        else:
            raise ValueError(
                "Invalid generation method. Use 'local', 'openai', or 'gemini'."
            )

    def _get_local_pipeline(self):
        """Initialize the local Hugging Face pipeline for text generation."""
        return transformers.pipeline(
            "text-generation",
            model=MODEL_ID_LLM,
            model_kwargs={"torch_dtype": torch.float16},
            device=self.device,
        )

    def generate_instructions_with_classification(
        self,
        task_pool,
        num_human,
        num_model,
        batch_size=8,
    ):
        """
        Generate new instructions and identify if they're classification tasks in a single step.

        Parameters:
        - task_pool: list, existing tasks to sample prompts from
        - num_human: int, number of human-written tasks per prompt
        - num_model: int, number of model-generated tasks per prompt
        - batch_size: int, number of instructions to generate at once

        Returns:
        - dict: mapping instruction strings to boolean (True if classification)
        """
        sampled_prompts = sample_tasks_round_robin(task_pool, num_human, num_model)
        instructions_with_classification = {}

        for prompt_tasks in sampled_prompts:
            prompt = self._create_combined_instruction_generation_prompt(
                prompt_tasks,
                num_instructions=batch_size,
            )

            if self.generation_method == "local":
                generated_text = self._generate_with_local_model(
                    prompt, max_tokens=2048, temperature=0.7
                )
            elif self.generation_method == "openai":
                generated_text = self._generate_with_openai(
                    prompt, max_tokens=2048, temperature=0.7
                )
            elif self.generation_method == "gemini":
                generated_text = self._generate_with_gemini(
                    prompt, max_tokens=2048, temperature=0.7
                )

            # Parse instructions and classification status
            parsed_results = self._parse_instructions_with_classification(
                generated_text
            )
            instructions_with_classification.update(parsed_results)

        return instructions_with_classification

    def generate_instances(self, instructions_with_classification):
        """
        Generate instances for each instruction based on whether it's a classification task.

        Parameters:
        - instructions_with_classification: dict mapping instructions to classification status

        Returns:
        - tasks_with_instances: list of task dictionaries with instances
        """
        tasks_with_instances = []

        for idx, (instruction, is_classification) in enumerate(
            instructions_with_classification.items()
        ):
            instances = []

            if is_classification:
                # Output-first approach for classification tasks
                prompt = self._create_classification_instance_prompt(instruction)

                if self.generation_method == "local":
                    response = self._generate_with_local_model(
                        prompt, max_tokens=2048, temperature=0.7
                    )
                elif self.generation_method == "openai":
                    response = self._generate_with_openai(
                        prompt, max_tokens=2048, temperature=0.7
                    )
                elif self.generation_method == "gemini":
                    response = self._generate_with_gemini(
                        prompt, max_tokens=2048, temperature=0.7
                    )

                # Parse classification instances (class label first, then input)
                instances = self._parse_classification_instances(response)
            else:
                # Input-first approach for non-classification tasks
                # Update prompt to explicitly request the exact number of instances
                prompt = self._create_non_classification_instance_prompt(instruction)

                if self.generation_method == "local":
                    response = self._generate_with_local_model(
                        prompt, max_tokens=2048, temperature=0.7
                    )
                elif self.generation_method == "openai":
                    response = self._generate_with_openai(
                        prompt, max_tokens=2048, temperature=0.7
                    )
                elif self.generation_method == "gemini":
                    response = self._generate_with_gemini(
                        prompt, max_tokens=2048, temperature=0.7
                    )

                # Parse non-classification instances (input first, then output)
                instances = self._parse_non_classification_instances(response)

            if instances:
                task = {
                    "id": f"generated_task_{uuid4().hex}",
                    "name": "generated_task",
                    "instruction": instruction,
                    "instances": instances,
                    "is_classification": is_classification,
                    "origin": "model-generated",
                }
                tasks_with_instances.append(task)

            # Save every 1000 tasks to avoid losing progress
            if idx % 1000 == 0:
                LOGGER.info(f"Generated instances for {idx} tasks")
                with open(f"temp_tasks_with_instances_{idx}.json", "w") as f:
                    json.dump(tasks_with_instances, f, indent=2)

        LOGGER.info("Completed generating instances for all tasks.")

        return tasks_with_instances

    def generate_tasks(self, task_pool, num_human, num_model, batch_size=8):
        """
        Complete Self-Instruct pipeline: generates instructions with classification status,
        generates instances, and prepares final task objects.

        Parameters:
        - task_pool: list, existing tasks to sample prompts from
        - num_human: int, number of human-written tasks per prompt
        - num_model: int, number of model-generated tasks per prompt
        - batch_size: int, number of tasks to generate

        Returns:
        - generated_tasks: list of task objects with instances
        """
        # Stage 1+2: Generate instructions with classification status
        LOGGER.info("Stage 1+2: Generating instructions with classification status...")
        instructions_with_classification = (
            self.generate_instructions_with_classification(
                task_pool, num_human, num_model, batch_size
            )
        )

        classification_count = sum(
            1 for v in instructions_with_classification.values() if v
        )
        LOGGER.info(
            f"Generated {len(instructions_with_classification)} instructions "
            f"({classification_count} classification, "
            f"{len(instructions_with_classification) - classification_count} non-classification)"
        )

        # Stage 3: Generate instances
        LOGGER.info("Stage 3: Generating instances...")
        tasks_with_instances = self.generate_instances(instructions_with_classification)
        LOGGER.info(f"Generated instances for {len(tasks_with_instances)} tasks")

        return tasks_with_instances

    def _create_combined_instruction_generation_prompt(
        self, example_tasks, num_instructions=5
    ):
        """Create a prompt for generating unique, factually correct dermatology instructions."""
        prompt = (
            "You are a board-certified dermatologist with expertise in clinical dermatology, dermatopathology, and "
            "procedural dermatology. Generate unique, factually correct dermatology instructions that represent "
            "realistic clinical scenarios.\n\n"
            "REQUIREMENTS FOR INSTRUCTIONS:\n"
            "1. Each instruction must be medically accurate and reflect current dermatological practice\n"
            "2. Create diverse instructions covering various aspects: diagnosis, treatment planning, patient education, "
            "procedural techniques, dermoscopic interpretation, and differential diagnosis\n"
            "3. Include specific dermatologic conditions (e.g., actinic keratosis, SCC, melanoma, psoriasis, atopic dermatitis, "
            "hidradenitis suppurativa, etc.) rather than generic skin problems\n"
            "4. Incorporate instructions addressing special populations (pediatric, geriatric, immunocompromised, skin of color, etc.)\n"
            "5. Include both common and uncommon/rare dermatologic conditions\n\n"
            "FORMAT REQUIREMENTS:\n"
            "1. Each instruction must be prefixed with 'Instruction N:' where N is a number\n"
            "2. Immediately after each instruction, include a line 'Is classification task: Yes/No'\n"
            "3. Classification tasks require selecting from specific options or categories\n"
            "4. Non-classification tasks require detailed analysis, explanation, or planning\n"
            "5. Instructions should be 1-3 sentences, clear, and specific\n\n"
            "Examples:\n\n"
        )

        for i, task in enumerate(example_tasks, 1):
            prompt += f"Instruction {i}: {task['instruction']}\n"
            prompt += f"Is classification task: {'Yes' if task.get('is_classification', False) else 'No'}\n\n"

        prompt += (
            f"Now, generate {num_instructions} NEW dermatology instructions. Ensure they are UNIQUE from the examples "
            f"and from each other. Make them factually correct, clinically relevant, and clearly indicate if each is a "
            f"classification task.\n\n"
        )

        return prompt

    def _create_classification_instance_prompt(self, instruction):
        """Create a prompt for generating a single classification instance."""
        prompt = (
            "You are a dermatology expert assistant. Generate one realistic example "
            "for the following dermatology classification task.\n\n"
            f"Task: {instruction}\n\n"
            "Requirements:\n"
            "1. First provide a specific class label that answers the task\n"
            "2. Then provide a detailed, realistic dermatology scenario that would result in that classification\n"
            "3. Include relevant clinical details like symptoms, appearance, etc.\n"
            "4. Be medically accurate and clinically sound\n\n"
            "Format your response exactly like this:\n"
            "Class label: [specific, concise classification label]\n"
            "Input: [detailed clinical scenario with relevant history and findings]"
        )
        return prompt

    def _create_non_classification_instance_prompt(self, instruction):
        """Create a prompt for generating a single non-classification instance."""
        prompt = (
            "You are a dermatology expert assistant. Generate one realistic example "
            "for the following dermatology analytical task.\n\n"
            f"Task: {instruction}\n\n"
            "Requirements:\n"
            "1. First provide a detailed, realistic dermatology scenario as input\n"
            "2. Then provide a thorough, medically accurate response as output\n"
            "3. Include relevant clinical details in both the input and output\n"
            "4. Make the output comprehensive, educational and clinically sound\n\n"
            "Format your response exactly like this:\n"
            "Input: [detailed clinical scenario with relevant history and findings]\n"
            "Output: [comprehensive, medically accurate response]"
        )
        return prompt

    def _generate_with_local_model(self, prompt, max_tokens=128, temperature=0.7):
        """
        Generate text using the local model pipeline.
        NOTE: NOT TESTED
        """
        try:
            inputs = self.pipeline.tokenizer(prompt, return_tensors="pt").to(
                self.pipeline.device
            )
            outputs = self.pipeline.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=0.9,
            )
            return self.pipeline.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()
        except Exception as e:
            LOGGER.error(f"Local model generation error: {e}")
            return ""

    def _generate_with_openai(self, prompt, max_tokens=128, temperature=0.7):
        """
        Generate text using OpenAI's Chat Completions API.
        NOTE: NOT TESTED
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a specialized dermatology assistant. When generating content, "
                            "you must follow the exact format specifications in the user's prompt. "
                            "Never deviate from the required formatting patterns."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            LOGGER.error(f"OpenAI API error: {e}")
            return ""

    def _generate_with_gemini(self, prompt, max_tokens=128, temperature=0.7):
        """Generate text using Google's Gemini API."""
        try:
            # Create configuration for generation
            config = genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            prompt = (
                "You are a specialized dermatology assistant. When generating content, "
                "you must follow the exact format specifications in this prompt.\n\n"
                f"{prompt}"
            )

            # Generate content
            response = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config=config,
            )

            # Return the generated text
            return response.text.strip()
        except Exception as e:
            LOGGER.error(f"Gemini API error: {e}")
            return ""

    def _parse_instructions_with_classification(self, generated_text):
        """
        Parse the generated text to extract instructions and their classification status.
        This version enforces strict format and does not attempt to salvage malformed outputs.
        """
        instructions_with_classification = {}

        # Use a more precise regex pattern to match the instruction format
        pattern = r"Instruction\s+\d+:\s*(.*?)\nIs classification task:\s*(Yes|No)"
        matches = re.finditer(pattern, generated_text, re.IGNORECASE | re.DOTALL)

        for match in matches:
            instruction = match.group(1).strip()
            classification_status = match.group(2).strip().lower()

            # Only add properly formatted instructions with sufficient length
            if len(instruction) > 10:
                is_classification = classification_status == "yes"
                instructions_with_classification[instruction] = is_classification

        if not instructions_with_classification:
            LOGGER.warning(
                f"Failed to parse any instructions from generated text. Text: {generated_text[:100]}..."
            )

        return instructions_with_classification

    def _parse_classification_instances(self, generated_text):
        """
        Parse instances from classification instance generation using strict format requirements.
        Only extracts properly formatted class label and input pairs.
        """
        instances = []

        # Match class label and input pairs with exact format
        pattern = r"Class label:\s*(.*?)\nInput:\s*(.*?)(?=Class label:|$)"
        matches = re.finditer(pattern, generated_text, re.DOTALL)

        for match in matches:
            class_label = match.group(1).strip()
            input_text = match.group(2).strip()

            # Only include pairs where both class label and input exist
            if class_label and input_text:
                instances.append({"input": input_text, "output": class_label})

        if not instances:
            LOGGER.warning(
                f"Failed to parse any classification instances from generated text. Text: {generated_text[:100]}..."
            )

        return instances

    def _parse_non_classification_instances(self, generated_text):
        """
        Parse instances from non-classification instance generation with improved robustness.
        Handles multiple formats including with or without Example prefix.

        Parameters:
        - generated_text: Text generated by the model containing instances

        Returns:
        - List of instances with input and output fields
        """
        instances = []

        # Try several different regex patterns to handle various formats
        # Log each attempt for better debugging

        # Pattern 1: Standard format with Example prefix
        # Example 1:
        # Input: ...
        # Output: ...
        pattern1 = (
            r"Example\s+\d+:?\s*\nInput:\s*(.*?)\nOutput:\s*(.*?)(?=Example\s+\d+:|$)"
        )
        matches1 = list(re.finditer(pattern1, generated_text, re.DOTALL))

        if matches1:
            LOGGER.debug(
                f"Found {len(matches1)} instances using standard Example-Input-Output format"
            )
            for match in matches1:
                input_text = match.group(1).strip()
                output_text = match.group(2).strip()

                # Output must exist, input can be empty
                if output_text:
                    instances.append({"input": input_text, "output": output_text})

        # Pattern 2: Direct Input-Output format without Example prefix
        # Input: ...
        # Output: ...
        if not instances:
            pattern2 = r"Input:\s*(.*?)\nOutput:\s*(.*?)(?=Input:|$)"
            matches2 = list(re.finditer(pattern2, generated_text, re.DOTALL))

            if matches2:
                LOGGER.debug(
                    f"Found {len(matches2)} instances using direct Input-Output format"
                )
                for match in matches2:
                    input_text = match.group(1).strip()
                    output_text = match.group(2).strip()

                    # Output must exist, input can be empty
                    if output_text:
                        instances.append({"input": input_text, "output": output_text})

        # Pattern 3: Last resort - look for anything that resembles Input-Output structure
        # This is more permissive and only used if the previous patterns fail
        if not instances:
            pattern3 = r"(?:Example.*?|^)(?:Input|Case)?\s*:?\s*(.*?)(?:Output|Assessment|Diagnosis|Plan)\s*:?\s*(.*?)(?=Example|\n\s*(?:Input|Case)|$)"
            matches3 = list(re.finditer(pattern3, generated_text, re.DOTALL))

            if matches3:
                LOGGER.debug(
                    f"Found {len(matches3)} instances using permissive pattern matching"
                )
                for match in matches3:
                    input_text = match.group(1).strip()
                    output_text = match.group(2).strip()

                    # Both input and output should exist and be reasonably lengthy
                    if output_text and len(output_text) > 5:
                        instances.append({"input": input_text, "output": output_text})

        if not instances:
            # Add the full text in debug logs, added this because this parsing was the most difficult
            LOGGER.warning(
                f"Failed to parse any non-classification instances from generated text."
            )
            LOGGER.debug(f"Full text that couldn't be parsed: {generated_text}")
            # Only show a truncated version in the regular warning log
            LOGGER.warning(f"First 100 chars: {generated_text[:100]}...")

        return instances
