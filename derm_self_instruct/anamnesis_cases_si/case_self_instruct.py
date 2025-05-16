import os
import json
import logging
import time
import re
from typing import List, Dict, Any, Optional
from uuid import uuid4

from google import genai
from google.genai import types as genai_types
from tqdm import tqdm

from .case_structure import DermCase, load_seed_cases, save_cases, sample_cases

# Setup logging
logger = logging.getLogger(__name__)


class DermCaseSelfInstruct:
    """Pipeline for generating synthetic dermatology cases."""

    def __init__(
        self,
        api_key: str,
        output_dir: str = "output",
        gemini_model: str = "gemini-2.0-flash",
    ):
        """
        Initialize the DermCaseSelfInstruct pipeline.

        Args:
            api_key: Gemini API key
            output_dir: Directory to save outputs
            gemini_model: Gemini model to use
        """
        self.output_dir = output_dir
        self.gemini_model = gemini_model

        # Initialize Gemini client
        self.genai_client = genai.Client(api_key=api_key)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Initialized DermCaseSelfInstruct with model {gemini_model}")

    def format_case_for_prompt(self, case: DermCase) -> str:
        """Format a case for inclusion in the prompt template."""
        formatted_case = f"CASE ID: {case.case_id}\n"
        formatted_case += f"DISEASE: {case.disease}\n"
        formatted_case += "BASE INFORMATION:\n"

        for key, value in case.base_information.items():
            formatted_case += f"  {key}: {value}\n"

        formatted_case += "QUESTIONS AND ANSWERS:\n"
        for qa in case.questions_and_answers:
            section = qa.get("section", "")
            subsection = qa.get("subsection", "")
            question = qa.get("question", "")
            answer = qa.get("answer", "")
            answer_status = qa.get("answer_status", "")

            section_header = f"{section}"
            if subsection:
                section_header += f" > {subsection}"

            formatted_case += f"  {section_header}\n"
            formatted_case += f"    Q: {question}\n"

            if answer_status == "answered":
                formatted_case += f"    A: {answer}\n"
            elif answer_status == "not_applicable":
                formatted_case += (
                    f"    A: [NOT APPLICABLE] {qa.get('reason_explanation', '')}\n"
                )
            elif answer_status == "unanswered":
                formatted_case += (
                    f"    A: [UNANSWERED] {qa.get('reason_explanation', '')}\n"
                )

        formatted_case += f"FINAL DIAGNOSIS: {case.final_diagnosis}\n"
        formatted_case += f"TREATMENT OF CHOICE: {case.treatment_of_choice}\n"

        if case.additional_info:
            formatted_case += f"ADDITIONAL INFORMATION: {case.additional_info}\n"

        return formatted_case

    def create_generation_prompt(self, cases: List[DermCase]) -> str:
        """Create a prompt for generating a new case based on examples."""
        prompt = (
            "You are a board-certified dermatologist tasked with creating realistic dermatology case studies "
            "for training medical students. Generate a new, UNIQUE dermatology case that follows the same format "
            "as the examples below. Ensure medical accuracy and realism.\n\n"
            "REQUIREMENTS:\n"
            "1. Follow the EXACT format of the examples, with all sections in the same order\n"
            "2. Generate a completely different dermatological condition than those in the examples\n"
            "3. Include all sections: case_id, disease, base_information, questions_and_answers, final_diagnosis, "
            "treatment_of_choice, and any_important_additional_information\n"
            "4. For 'questions_and_answers', include EXACTLY the same sections, subsections, and questions as the examples\n"
            "5. For each question, include:\n"
            "   - section: The major category (e.g., 'Main Complaint', 'History of Present Complaint')\n"
            "   - subsection: A subcategory if applicable (e.g., 'Location', 'Onset')\n"
            "   - question: The exact text of the question\n"
            "   - answer: Patient's response\n"
            "   - answer_status: One of 'answered', 'not_applicable', or 'unanswered'\n"
            "6. If answer_status is 'not_applicable' or 'unanswered', include a 'reason_explanation'\n"
            "7. Ensure 'base_information' includes age, gender, and other_relevant_history\n"
            "8. Make the case medically accurate, realistic, and educational\n\n"
            "EXAMPLE CASES:\n\n"
        )

        # Add examples
        for i, case in enumerate(cases, 1):
            prompt += f"Example Case {i}:\n```json\n"
            prompt += json.dumps(case.to_dict(), indent=2)
            prompt += "\n```\n\n"

        prompt += (
            "Now, generate a NEW, UNIQUE case following the EXACT same structure. "
            "The 'questions_and_answers' section should have the same structure, sections, and questions "
            "as the examples, but with different, medically accurate answers. "
            "Ensure it represents a different dermatological condition than the examples, but maintain "
            "the exact same format including the question and section organization.\n\n"
            "Output ONLY the new case in valid JSON format without any additional text or explanation."
        )

        return prompt

    def generate_case(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Generate a new case using Gemini API."""
        try:
            # Create configuration for generation
            config = genai_types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=16384,
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

            # Try multiple approaches to extract JSON

            # Approach 1: Look for JSON between triple backticks
            json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", generated_text)
            if json_match:
                json_str = json_match.group(1).strip()
                try:
                    new_case = json.loads(json_str)
                    logger.info("Successfully extracted JSON from code block")
                    return new_case
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse JSON from code block, trying other methods"
                    )

            # Approach 2: Look for the outermost JSON object
            json_start = generated_text.find("{")
            json_end = generated_text.rfind("}")

            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = generated_text[json_start : json_end + 1]
                try:
                    new_case = json.loads(json_str)
                    logger.info("Successfully extracted JSON using braces matching")
                    return new_case
                except json.JSONDecodeError as je:
                    logger.warning(f"Failed to parse JSON using braces: {je}")

            # Approach 3: Try to clean up the text and extract JSON
            # Remove any non-JSON text at the beginning and end
            lines = generated_text.split("\n")
            json_lines = []
            in_json = False
            open_braces = 0

            for line in lines:
                if not in_json and "{" in line:
                    in_json = True

                if in_json:
                    json_lines.append(line)
                    open_braces += line.count("{") - line.count("}")

                    if open_braces == 0 and in_json:
                        break

            if json_lines:
                json_str = "\n".join(json_lines)
                try:
                    new_case = json.loads(json_str)
                    logger.info(
                        "Successfully extracted JSON using line-by-line parsing"
                    )
                    return new_case
                except json.JSONDecodeError:
                    pass

            # If all extraction methods fail, log the error and return None
            logger.error("Failed to extract valid JSON from generated text")
            logger.debug(f"Full generated text: {generated_text}")

            # Save the failed output for debugging
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            error_file = os.path.join(
                self.output_dir, f"failed_generation_{timestamp}.txt"
            )
            with open(error_file, "w") as f:
                f.write(generated_text)
            logger.info(f"Saved failed generation to {error_file}")

            return None

        except Exception as e:
            logger.error(f"Error generating case: {e}")
            return None

    def run(
        self,
        seed_data_path: str,
        output_path: str,
        num_cases: int = 10,
        num_human: int = 2,
        num_model: int = 2,
        save_intermediate: bool = True,
    ):
        """
        Run the DermCaseSelfInstruct pipeline to generate new cases.

        Args:
            seed_data_path: Path to the seed data file
            output_path: Path to save the final dataset
            num_cases: Number of new cases to generate
            num_human: Number of human cases to use per prompt
            num_model: Number of model cases to use per prompt
            save_intermediate: Whether to save intermediate results

        Returns:
            List of all cases (seed + generated)
        """
        # Load seed cases
        logger.info(f"Loading seed cases from {seed_data_path}")
        seed_cases = load_seed_cases(seed_data_path)

        if not seed_cases:
            logger.error("No seed cases found. Exiting.")
            return None

        logger.info(f"Loaded {len(seed_cases)} seed cases")

        # Initialize task pool with seed cases
        all_cases = seed_cases.copy()
        generated_cases = []

        # Generate new cases
        logger.info(f"Generating {num_cases} new cases")
        progress_bar = tqdm(total=num_cases)

        for i in range(num_cases):
            # Sample cases for this generation
            sampled_cases = sample_cases(all_cases, num_human, num_model)[0]

            # Create prompt
            prompt = self.create_generation_prompt(sampled_cases)

            # Generate new case
            new_case_dict = self.generate_case(prompt)

            if new_case_dict:
                # Add origin and ensure case_id is unique
                new_case_dict["origin"] = "model-generated"
                new_case_dict["case_id"] = f"synthetic_{uuid4().hex[:8]}"

                # Create DermCase object
                new_case = DermCase.from_dict(new_case_dict)

                # Add to collections
                all_cases.append(new_case)
                generated_cases.append(new_case)

                logger.info(
                    f"Generated case {i+1}/{num_cases}: {new_case.case_id} - {new_case.disease}"
                )

                # Save intermediate results
                if save_intermediate and (i + 1) % 2000 == 0:
                    intermediate_path = os.path.join(
                        self.output_dir, f"generated_cases_intermediate_{i+1}.json"
                    )
                    save_cases(generated_cases, intermediate_path)
                    logger.info(f"Saved intermediate results to {intermediate_path}")
            else:
                logger.warning(f"Failed to generate case {i+1}/{num_cases}")

            progress_bar.update(1)

        progress_bar.close()

        # Save all cases
        logger.info(f"Generated {len(generated_cases)} new cases")
        logger.info(f"Saving all {len(all_cases)} cases to {output_path}")
        save_cases(all_cases, output_path)

        # Save only generated cases
        generated_path = os.path.join(self.output_dir, "generated_cases.json")
        save_cases(generated_cases, generated_path)
        logger.info(f"Saved {len(generated_cases)} generated cases to {generated_path}")

        return all_cases
