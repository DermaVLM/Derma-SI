from typing import Dict, List, Union, Any, Optional
import json
import random
from pathlib import Path
import logging
from collections import defaultdict

LOGGER = logging.getLogger(__name__)


class DermatologyTextBenchmark:
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        difficulty_levels: List[str] = ["medium", "hard"],
        question_types: List[str] = [
            "diagnosis",
            "treatment",
            "pathophysiology",
            "epidemiology",
            "clinical_features",
        ],
    ):
        """
        Initialize the text benchmark generator for dermatology.

        Args:
            dataset_path: Optional path to additional dataset if needed
            difficulty_levels: Difficulty levels to generate
            question_types: Types of questions to generate
        """
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.difficulty_levels = difficulty_levels
        self.question_types = question_types

        # Load the dermatology disease hierarchy for generating balanced questions
        self.dermatological_diseases = self._load_disease_hierarchy()

        # Track coverage to ensure balanced benchmark
        self.coverage_tracker = defaultdict(int)

    def _load_disease_hierarchy(self) -> Dict:
        """
        Load the dermatology disease hierarchy from local data.
        """
        # TODO: Load from json
        return {
            "1_most_frequent_dermatological_diseases": {
                "1_acne_rosacea": {
                    "acne": [
                        "Comedonal acne",
                        "Papulopustular acne",
                        "Nodulocystic acne",
                        "Acne conglobata",
                    ],
                    "rosacea": [
                        "Erythematotelangiectatic rosacea",
                        "Papulopustular rosacea",
                        "Phymatous rosacea",
                        "Ocular rosacea",
                    ],
                },
                "2_seborrheic_dermatitis": [
                    "Scalp seborrheic dermatitis",
                    "Facial seborrheic dermatitis",
                    "Intertriginous seborrheic dermatitis",
                    "Seborrheic dermatitis in immunocompromised patients",
                ],
                "3_parasytes": [
                    "Typical (classic) scabies",
                    "Crusted (Norwegian) scabies",
                    "Nodular scabies",
                    "Scabies incognito",
                    "Lice Infestation (Pediculosis)",
                    "Cutenous Larva Migrans",
                ],
                "4_allergic_skin_conditions": {
                    "irritant_contact_dermatitis": [
                        "Hand dermatitis",
                        "Foot dermatitis",
                        "Airborne irritant dermatitis",
                        "Diaper dermatitis",
                    ],
                    "allergic_contact_dermatitis": ["Allergic Contact Dermatitis"],
                    "urticaria": [
                        "Acute urticaria",
                        "Chronic urticaria",
                        "Physical urticarias",
                    ],
                    "angioedema": ["Hereditary angioedema", "Acquired Angioedema"],
                    "drug_reactions": [
                        "Exanthematous (Maculopapular) Drug Eruption",
                        "Urticarial Drug Reaction",
                        "Fixed Drug Eruption",
                        "Photosensitive Drug Eruption",
                        "Stevens-Johnson Syndrome (SJS) & Toxic Epidermal Necrolysis (TEN)",
                        "Drug Reaction with Eosinophilia and Systemic Symptoms (DRESS)",
                        "Acute Generalized Exanthematous Pustulosis (AGEP)",
                        "Bullous/Pustular Drug Eruptions",
                    ],
                },
                "5_warts": [
                    "Common warts (verruca vulgaris)",
                    "Plantar warts (verruca plantaris)",
                    "Flat warts (verruca plana)",
                    "Filiform warts",
                    "Genital warts (condyloma acuminata)",
                ],
                "6_fungal_infections": {
                    "dermatophyte_infections": [
                        "Tinea pedis",
                        "Tinea cruris",
                        "Tinea corporis",
                        "Tinea faciei",
                        {"Tinea capitis": ["Superficial", "Kerion", "Favus"]},
                        "Tinea unguium (onychomycosis)",
                        "Tinea indotinea",
                    ],
                    "candidiasis": ["Skin folds", "Mucous membranes"],
                    "pityriasis_versicolor": ["Pityriasis (Tinea) versicolor"],
                },
            },
            "2_second_most_frequent_dermatological_diseases": {
                "1_inflammatory": {
                    "psoriasis": [
                        "Plaque psoriasis (psoriasis vulgaris)",
                        "Guttate psoriasis",
                        "Inverse (flexural) psoriasis",
                        "Pustular psoriasis",
                        "Erythrodermic psoriasis",
                        "Nail psoriasis",
                        "Arthropathic",
                    ],
                    "lichen": ["Lichen planus", "Lichen Striatus"],
                },
                "4_alopecia_areata": [
                    "Patchy alopecia areata",
                    "Alopecia totalis",
                    "Alopecia universalis",
                    "Ophiasis",
                ],
                "5_vitiligo": [
                    "Non-segmental vitiligo (generalized)",
                    "Segmental vitiligo",
                    "Focal vitiligo",
                    "Universal vitiligo",
                ],
                "6_1_atopic_dermatitis": [
                    "Infantile atopic dermatitis",
                    "Childhood atopic dermatitis",
                    "Adult atopic dermatitis",
                ],
                "6_2_eczema": [
                    "Nummular eczema",
                    "Eczema herpeticum",
                    "Palmoplantar eczema",
                ],
                "7_neoplastic_skin_conditions": {
                    "benign": [
                        "Seborrheic Keratosis",
                        "Dermatofibroma",
                        "Hemangioma",
                        "Lyphangioma",
                        "Nevus",
                        "Pyogenic Granuloma",
                        "Angiokeratoma",
                    ],
                    "malignant": [
                        {
                            "Basal cell carcinoma (BCC)": [
                                "Nodular BCC",
                                "Superficial BCC",
                                "Morpheaform (sclerosing) BCC",
                            ]
                        },
                        {
                            "Squamous cell carcinoma (SCC)": [
                                "In situ (Bowen's disease)",
                                "Invasive SCC",
                            ]
                        },
                        "Actinic keratosis",
                        "Melanoma",
                        "Kaposi Sarcoma",
                        "Merkel Cell Carcinoma",
                    ],
                },
                "8_1_bacterial_infections": [
                    "Impetigo (bullous and non-bullous)",
                    "Cellulitis, erysipelas",
                    "Folliculitis, furuncles, carbuncles",
                ],
                "8_2_viral_infections": [
                    "Herpes simplex (oral, genital)",
                    "Herpes zoster (shingles)",
                    "Molluscum contagiosum",
                    "Hand-foot-and-mouth disease",
                    "Measles (Skin rash manifestation)",
                ],
                "9_sexually_transmitted_diseases": [
                    "Syphilis",
                    "Gonorrhea",
                    "Chlamydia",
                    "Genital herpes (HSV-2, sometimes HSV-1)",
                    "Human papillomavirus (HPV)",
                    "HIV",
                ],
                "10_autoimmune_skin_disease": [
                    "Lupus erythematosus (Discoid, systemic)",
                    "Dermatomyositis",
                    "Bullous pemphigoid",
                    "Pemphigus vulgaris",
                ],
                "11_genetic_hereditary_conditions": [
                    "Ichthyosis Vulgaris",
                    "Epidermolysis Bullosa",
                    "Neurofibromatosis",
                    "Tuberous Sclerosis",
                    "Xeroderma Pigmentosum",
                ],
                "12_sweat_gland_disorders": [
                    "Hyperhidrosis",
                    "Miliaria (Heat Rash)",
                    "Hidradenitis Suppurativa",
                ],
                "13_skin_injuries": [
                    "Abrasions (Scrapes)",
                    "Lacerations (Cuts)",
                    "Avulsions",
                    "Puncture Wounds",
                    "Contusions (Bruises)",
                    "Friction Blisters",
                    "Burns -- thermal",
                    "Burns -- chemical",
                    "Burns -- electrical",
                    "Burns - radiation",
                    "Frostbite",
                    "Pressure (Decubitus) Ulcers",
                    "Shear and Friction Injuries",
                    "Crush Injuries",
                    "Skin Tears",
                    "Radiation Dermatitis (Non-burn Radiation Injury)",
                ],
                "14_other_skin_diseases": [
                    "Keratosis Pilaris",
                    "Prurigo Nodularis",
                    "Pityriasis Rosea",
                ],
            },
        }

    def prepare_knowledge_benchmark_prompt(
        self,
        category: str,
        subcategory: str,
        condition: str,
        difficulty: str,
        question_type: str,
    ) -> str:
        """
        Generate a prompt for knowledge-based dermatology questions.

        Args:
            category: Major disease category (e.g., "acne_rosacea")
            subcategory: Subcategory if applicable (e.g., "acne")
            condition: Specific condition to focus on (e.g., "Comedonal acne")
            difficulty: Desired difficulty level ("easy", "medium", "hard")
            question_type: Type of question (diagnosis, treatment, etc.)

        Returns:
            Formatted prompt for generating a knowledge benchmark question
        """
        # Update coverage tracker
        self.coverage_tracker[f"{category}_{subcategory}_{condition}"] += 1

        prompt = (
            f"***** INSTRUCTIONS *****\n"
            f"You are an AI assistant tasked with generating a challenging multiple-choice question for a dermatology knowledge benchmark.\n"
            f"Generate ONE {difficulty}-level multiple-choice question with four options (A, B, C, D) about {condition}.\n"
            f"The question should focus on {question_type} aspects of this condition.\n\n"
            f"Your question should:\n"
            f"- Be at {difficulty} difficulty level for dermatology specialists\n"
            f"- Focus specifically on {question_type} aspects\n"
            f"- Include plausible distractors that test understanding of dermatological concepts\n"
            f"- Be clinically relevant for practicing dermatologists\n\n"
            f"***** CONTEXT *****\n"
            f"Dermatological Category: {category}\n"
            f"Subcategory: {subcategory}\n"
            f"Specific Condition: {condition}\n"
            f"Question Type: {question_type}\n"
            f"Difficulty Level: {difficulty}\n\n"
            f"***** RESPONSE FORMAT *****\n"
            f"{{\n"
            f'  "question_metadata": {{\n'
            f'    "category": "{category}",\n'
            f'    "subcategory": "{subcategory}",\n'
            f'    "condition": "{condition}",\n'
            f'    "difficulty": "{difficulty}",\n'
            f'    "question_type": "{question_type}"\n'
            f"  }},\n"
            f'  "question": "<dermatology_question>",\n'
            f'  "options": {{\n'
            f'    "A": "<option_A>",\n'
            f'    "B": "<option_B>",\n'
            f'    "C": "<option_C>",\n'
            f'    "D": "<option_D>"\n'
            f"  }},\n"
            f'  "correct_answer": "<A, B, C, or D>",\n'
            f'  "explanation": "<detailed_explanation_of_why_the_correct_answer_is_right_and_others_are_wrong>",\n'
            f'  "clinical_relevance": "<brief_explanation_of_the_clinical_relevance_of_this_knowledge>"\n'
            f"}}"
        )

        return prompt

    def prepare_differential_diagnosis_prompt(
        self, conditions: List[str], difficulty: str
    ) -> str:
        """
        Generate a prompt for differential diagnosis questions.

        Args:
            conditions: List of related conditions to differentiate between
            difficulty: Desired difficulty level

        Returns:
            Formatted prompt for generating a differential diagnosis question
        """
        conditions_str = ", ".join(conditions)
        # Properly format conditions as JSON array
        conditions_json = json.dumps(conditions)

        prompt = (
            f"***** INSTRUCTIONS *****\n"
            f"You are an AI assistant tasked with generating a differential diagnosis question for a dermatology benchmark.\n"
            f"Generate ONE {difficulty}-level multiple-choice question with four options (A, B, C, D) that tests the ability to differentiate between: {conditions_str}.\n\n"
            f"Your question should:\n"
            f"- Present a clinical scenario where differential diagnosis is challenging\n"
            f"- Include key distinguishing features that hint at the correct diagnosis\n"
            f"- Test the ability to recognize subtle diagnostic differences\n"
            f"- Be at {difficulty} difficulty level for dermatology specialists\n\n"
            f"***** CONTEXT *****\n"
            f"Conditions to Differentiate: {conditions_str}\n"
            f"Difficulty Level: {difficulty}\n\n"
            f"***** RESPONSE FORMAT *****\n"
            f"{{\n"
            f'  "question_metadata": {{\n'
            f'    "question_type": "differential_diagnosis",\n'
            f'    "conditions_covered": {conditions_json},\n'
            f'    "difficulty": "{difficulty}"\n'
            f"  }},\n"
            f'  "clinical_scenario": "<detailed_patient_scenario>",\n'
            f'  "question": "What is the most likely diagnosis?",\n'
            f'  "options": {{\n'
            f'    "A": "<option_A>",\n'
            f'    "B": "<option_B>",\n'
            f'    "C": "<option_C>",\n'
            f'    "D": "<option_D>"\n'
            f"  }},\n"
            f'  "correct_answer": "<A, B, C, or D>",\n'
            f'  "explanation": "<detailed_explanation_of_differential_diagnosis_features>",\n'
            f'  "key_distinguishing_features": ["<feature_1>", "<feature_2>", "<feature_3>"]\n'
            f"}}"
        )

        return prompt

    def prepare_therapeutic_management_prompt(
        self, condition: str, difficulty: str
    ) -> str:
        """
        Generate a prompt for therapeutic management questions.

        Args:
            condition: Specific condition to focus on
            difficulty: Desired difficulty level

        Returns:
            Formatted prompt for generating a treatment management question
        """
        prompt = (
            f"***** INSTRUCTIONS *****\n"
            f"You are an AI assistant tasked with generating a therapeutic management question for a dermatology benchmark.\n"
            f"Generate ONE {difficulty}-level multiple-choice question with four options (A, B, C, D) about the treatment of {condition}.\n\n"
            f"Your question should:\n"
            f"- Present a specific clinical scenario requiring treatment decisions\n"
            f"- Focus on evidence-based treatment approaches\n"
            f"- Consider factors like patient demographics, comorbidities, or treatment history\n"
            f"- Be at {difficulty} difficulty level for dermatology specialists\n\n"
            f"***** CONTEXT *****\n"
            f"Condition to Treat: {condition}\n"
            f"Difficulty Level: {difficulty}\n\n"
            f"***** RESPONSE FORMAT *****\n"
            f"{{\n"
            f'  "question_metadata": {{\n'
            f'    "question_type": "therapeutic_management",\n'
            f'    "condition": "{condition}",\n'
            f'    "difficulty": "{difficulty}"\n'
            f"  }},\n"
            f'  "clinical_scenario": "<detailed_patient_scenario>",\n'
            f'  "question": "<specific_treatment_question>",\n'
            f'  "options": {{\n'
            f'    "A": "<option_A>",\n'
            f'    "B": "<option_B>",\n'
            f'    "C": "<option_C>",\n'
            f'    "D": "<option_D>"\n'
            f"  }},\n"
            f'  "correct_answer": "<A, B, C, or D>",\n'
            f'  "explanation": "<detailed_explanation_of_treatment_rationale>",\n'
            f'  "evidence_basis": "<brief_summary_of_evidence_supporting_this_approach>"\n'
            f"}}"
        )

        return prompt

    def prepare_histopathology_prompt(self, condition: str, difficulty: str) -> str:
        """
        Generate a prompt for histopathology-based questions.

        Args:
            condition: Specific condition to focus on
            difficulty: Desired difficulty level

        Returns:
            Formatted prompt for generating a histopathology question
        """
        prompt = (
            f"***** INSTRUCTIONS *****\n"
            f"You are an AI assistant tasked with generating a histopathology question for a dermatology benchmark.\n"
            f"Generate ONE {difficulty}-level multiple-choice question with four options (A, B, C, D) about the histopathological features of {condition}.\n\n"
            f"Your question should:\n"
            f"- Describe specific histopathological findings\n"
            f"- Test knowledge of characteristic microscopic features\n"
            f"- Include relevant staining or special techniques if applicable\n"
            f"- Be at {difficulty} difficulty level for dermatology specialists\n\n"
            f"***** CONTEXT *****\n"
            f"Condition: {condition}\n"
            f"Difficulty Level: {difficulty}\n\n"
            f"***** RESPONSE FORMAT *****\n"
            f"{{\n"
            f'  "question_metadata": {{\n'
            f'    "question_type": "histopathology",\n'
            f'    "condition": "{condition}",\n'
            f'    "difficulty": "{difficulty}"\n'
            f"  }},\n"
            f'  "histopathology_description": "<detailed_description_of_microscopic_findings>",\n'
            f'  "question": "<specific_histopathology_question>",\n'
            f'  "options": {{\n'
            f'    "A": "<option_A>",\n'
            f'    "B": "<option_B>",\n'
            f'    "C": "<option_C>",\n'
            f'    "D": "<option_D>"\n'
            f"  }},\n"
            f'  "correct_answer": "<A, B, C, or D>",\n'
            f'  "explanation": "<detailed_explanation_of_histopathological_features>",\n'
            f'  "key_microscopic_features": ["<feature_1>", "<feature_2>", "<feature_3>"]\n'
            f"}}"
        )

        return prompt

    def generate_balanced_benchmark(
        self, num_questions: int, output_path: str
    ) -> List[Dict]:
        """
        Generate a balanced set of benchmark questions across categories and difficulties.

        Args:
            num_questions: Number of questions to generate
            output_path: Path to save the questions

        Returns:
            List of generated question prompts
        """
        all_prompts = []

        # Calculate distribution across categories to ensure coverage
        category_allocation = self._calculate_distribution(num_questions)

        # Generate prompts according to the distribution
        for category, subcategory, condition, count in category_allocation:
            for _ in range(count):
                # Randomly select difficulty and question type
                difficulty = random.choice(self.difficulty_levels)
                question_type = random.choice(self.question_types)

                # Generate appropriate prompt based on question type
                if question_type == "differential_diagnosis":
                    # Get related conditions for differential diagnosis
                    related_conditions = self._get_related_conditions(
                        category, subcategory, condition, 3
                    )
                    prompt = self.prepare_differential_diagnosis_prompt(
                        [condition] + related_conditions, difficulty
                    )
                elif question_type == "therapeutic_management":
                    prompt = self.prepare_therapeutic_management_prompt(
                        condition, difficulty
                    )
                elif question_type == "histopathology":
                    prompt = self.prepare_histopathology_prompt(condition, difficulty)
                else:
                    # Default to knowledge-based questions
                    prompt = self.prepare_knowledge_benchmark_prompt(
                        category, subcategory, condition, difficulty, question_type
                    )

                all_prompts.append(
                    {
                        "prompt": prompt,
                        "metadata": {
                            "category": category,
                            "subcategory": subcategory,
                            "condition": condition,
                            "difficulty": difficulty,
                            "question_type": question_type,
                        },
                    }
                )

        # Save prompts to output file
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {"total_prompts": len(all_prompts), "prompts": all_prompts},
                f,
                indent=2,
            )

        print(f"Generated {len(all_prompts)} prompts and saved to {output_path}")
        return all_prompts

    def _calculate_distribution(self, num_questions: int) -> List[tuple]:
        """
        Calculate a balanced distribution of questions across the disease hierarchy.

        Args:
            num_questions: Total number of questions to generate

        Returns:
            List of (category, subcategory, condition, count) tuples
        """
        # Extract all conditions from the hierarchy
        all_conditions = []

        # Helper function to recursively traverse the hierarchy
        def extract_conditions(data, category="", subcategory=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key.startswith("1_") or key.startswith("2_"):
                        # This is a main category
                        new_category = key
                        extract_conditions(value, new_category, "")
                    else:
                        # This is a subcategory
                        new_subcategory = key
                        extract_conditions(value, category, new_subcategory)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Handle nested structure like Tinea capitis
                        extract_conditions(item, category, subcategory)
                    else:
                        # This is a specific condition
                        all_conditions.append((category, subcategory, item))

        # Process the disease hierarchy
        extract_conditions(self.dermatological_diseases)

        # Calculate weights - for now using equal weights
        # In a more sophisticated implementation, you could weight by prevalence
        total_conditions = len(all_conditions)
        distribution = []

        # Simple approach: distribute questions evenly, with at least 1 question per condition
        base_questions_per_condition = max(1, num_questions // total_conditions)
        extra_questions = num_questions - (
            base_questions_per_condition * total_conditions
        )

        # Assign base questions to all conditions
        for condition_info in all_conditions:
            distribution.append((*condition_info, base_questions_per_condition))

        # Distribute any extra questions
        # Prioritize more common or important conditions
        for i in range(extra_questions):
            # For now, just assign to the first N conditions
            # In a real implementation, you might use a more sophisticated approach
            if i < len(distribution):
                category, subcategory, condition, count = distribution[i]
                distribution[i] = (category, subcategory, condition, count + 1)

        return distribution

    def _get_related_conditions(
        self, category: str, subcategory: str, condition: str, num_related: int
    ) -> List[str]:
        """
        Find related conditions for differential diagnosis questions.

        Args:
            category: The condition's category
            subcategory: The condition's subcategory
            condition: The specific condition
            num_related: Number of related conditions to find

        Returns:
            List of related conditions
        """
        related_conditions = []

        # Helper function to collect all conditions
        def collect_conditions(data):
            conditions = []
            if isinstance(data, dict):
                for key, value in data.items():
                    conditions.extend(collect_conditions(value))
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        conditions.extend(collect_conditions(item))
                    else:
                        conditions.append(item)
            return conditions

        # Get all conditions
        all_conditions = collect_conditions(self.dermatological_diseases)

        # First, try to get conditions from the same subcategory (most related)
        same_subcategory = []
        for cat, subcat, cond in [
            (c, s, i)
            for c, s, i in [
                (c, s, self._extract_condition(i))
                for c, s, i in self._extract_all_conditions(
                    self.dermatological_diseases
                )
            ]
        ]:
            if cat == category and subcat == subcategory and cond != condition:
                same_subcategory.append(cond)

        # Then, try conditions from the same category (somewhat related)
        same_category = []
        for cat, subcat, cond in [
            (c, s, i)
            for c, s, i in [
                (c, s, self._extract_condition(i))
                for c, s, i in self._extract_all_conditions(
                    self.dermatological_diseases
                )
            ]
        ]:
            if cat == category and subcat != subcategory:
                same_category.append(cond)

        # Finally, use any other conditions (less related)
        other_conditions = [
            c
            for c in all_conditions
            if c != condition and c not in same_subcategory and c not in same_category
        ]

        # Combine to get the most related conditions first
        all_potential_related = same_subcategory + same_category + other_conditions

        # Take the first num_related unique conditions
        seen = set()
        for cond in all_potential_related:
            if cond not in seen and len(related_conditions) < num_related:
                related_conditions.append(cond)
                seen.add(cond)

        return related_conditions

    def _extract_all_conditions(self, data, category="", subcategory=""):
        """
        Extract all conditions with their categories and subcategories.
        """
        result = []
        if isinstance(data, dict):
            for key, value in data.items():
                if key.startswith("1_") or key.startswith("2_"):
                    # This is a main category
                    result.extend(self._extract_all_conditions(value, key, ""))
                else:
                    # This is a subcategory
                    result.extend(self._extract_all_conditions(value, category, key))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Handle nested structure
                    result.extend(
                        self._extract_all_conditions(item, category, subcategory)
                    )
                else:
                    # This is a specific condition
                    result.append((category, subcategory, item))
        return result

    def _extract_condition(self, condition_text):
        """Extract just the condition name without parenthetical explanations."""
        if "(" in condition_text:
            return condition_text.split("(")[0].strip()
        return condition_text


if __name__ == "__main__":
    benchmark_generator = DermatologyTextBenchmark()
    prompts = benchmark_generator.generate_balanced_benchmark(
        num_questions=2000,
        output_path="mnt/text_benchmark_prompts.json",
    )
