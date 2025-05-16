# %%
import json
import pandas as pd
from datasets import Dataset

with open("mnt/TASK_POOL_FINAL_at_07-03-25.json", "r") as f:
    data = json.load(f)

# %%
# Transform data to the required format
transformed_data = []

for item in data:
    instruction = item["instruction"]
    is_classification = item.get("is_classification")
    origin = item.get("origin")

    for instance in item["instances"]:
        # Ensure is_classification is a boolean, not a string
        is_classification_bool = is_classification
        if isinstance(is_classification, str):
            is_classification_bool = is_classification.lower() == "true"

        transformed_data.append(
            {
                "instruction": instruction,
                "input": instance["input"],
                "output": instance["output"],
                "is_classification": is_classification_bool,
                "origin": origin,
            }
        )

# Convert to pandas DataFrame
df = pd.DataFrame(transformed_data)

# %%
# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Push to the Hub
dataset.push_to_hub("DermaVLM/self_instruct_1-default_derma")

print(
    f"Successfully uploaded {len(transformed_data)} samples to DermaVLM/self_instruct_1-default_derma"
)
# %%
