# %%
import os
import json
import re

import pandas as pd
from datasets import Dataset

pd.set_option("display.max_colwidth", 555)
pd.set_option("display.max_columns", 555)
# %%
benchmark = "derm_self_instruct\\text_eval_generation\\benchmark_output\\benchmark_questions_final_1998.json"
with open(benchmark, "r") as f:
    benchmark = json.load(f)

# %%

# Sample of the benchmark data
"""
{'total_questions': 1998,
 'results': [{'question_id': 'derm_bench_b4877fc7',
   'prompt': '***** INSTRUCTIONS *****\nYou are an AI assistant tasked w...ion_of_the_clinical_relevance_of_this_knowledge>"\n}',
   'response': '```json\n{\n  "question_metadata": {\n    "category": "1_acne_rosacea",\n    "subcategory": "acne",\n    "condition": "Comedonal acne",\n    "difficulty": "hard",\n    "question_type": "pathophysiology"\n  },\n  "question": "Regarding the initial pathophysiological events leading to microcomedone formation in comedonal acne, which statement most accurately describes the primary driving mechanism?",\n  "options": {\n    "A": "Direct stimulation of keratinocyte proliferation by *Cutibacterium acnes* lipase activity hydrolyzing triglycerides into pro-proliferative free fatty acids.",\n    "B": "Androgen-mediated hypertrophy of sebaceous glands leading primarily to excessive sebum *volume* physically obstructing the follicular ostium.",\n    "C": "Aberrant terminal differentiation and increased cohesiveness of infundibular keratinocytes, influenced by alterations in the local sebaceous lipid profile, particularly reduced linoleic acid levels.",\n    "D": "Upregulation of Toll-like receptor 2 (TLR2) on perifollicular macrophages, triggering a cytokine cascade that secondarily induces follicular hyperkeratosis."\n  },\n  "correct_answer": "C",\n  "explanation": "The formation of the microc...nt very early.",\n  "clinical_relevance": "Understanding that altered ...dones."\n}\n```',
   'metadata': {'category': '1_acne_rosacea',
    'subcategory': 'acne',
    'condition': 'Comedonal acne',
    'difficulty': 'hard',
    'question_type': 'pathophysiology'},
   'timestamp': '2025-03-28 17:11:41'},
"""
# MMLU benchmark columns: question, subject, choices, answer


def extract_QA_from_response(response):
    """Extract the JSON content from the response"""
    # Find the content between ```json and ``` markers
    pattern = r"```(?:json)?\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)

    if not match:
        return None

    json_str = match.group(1)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


# %%
# Transform data to match MMLU format
transformed_data = []

for item in benchmark["results"]:
    question_id = item["question_id"]
    parsed_response = extract_QA_from_response(item["response"])

    if not parsed_response:
        raise ValueError(
            f"Failed to parse JSON from response for question ID {question_id}"
        )

        # Extract the question data
    question = parsed_response.get("question", "")
    options = parsed_response.get("options", {})
    correct_answer = parsed_response.get("correct_answer", "")
    explanation = parsed_response.get("explanation", "")
    clinical_relevance = parsed_response.get("clinical_relevance", "")

    # Format choices as a list for MMLU format
    choices = []
    for key in sorted(options.keys()):
        choices.append(options[key])

    # Get metadata
    category = item["metadata"]["category"]
    subcategory = item["metadata"]["subcategory"]
    condition = item["metadata"]["condition"]
    difficulty = item["metadata"]["difficulty"]
    question_type = item["metadata"]["question_type"]

    # Create entry in MMLU-like format
    entry = {
        "question_id": question_id,
        "question": question,
        "choices": choices,
        "answer": correct_answer,
        "explanation": explanation,
        "clinical_relevance": clinical_relevance,
        "subject": category,
        "subsubject": subcategory,
        "condition": condition,
        "difficulty": difficulty,
        "question_type": question_type,
    }

    if all(
        entry.get(key) is None
        for key in [
            "question",
            "choices",
            "answer",
            "explanation",
            "clinical_relevance",
            "difficulty",
        ]
    ):
        raise ValueError(
            f"Missing required fields in entry for question ID {question_id}"
        )

    else:
        transformed_data.append(entry)

# %%
df = pd.DataFrame(transformed_data)
dataset = Dataset.from_pandas(df)
dataset.push_to_hub("DermaVLM/eval3-text-benchmark")
print("Dataset pushed to the Hub successfully.")

# %%
