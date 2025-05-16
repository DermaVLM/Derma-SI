# %%
import json
import pandas as pd
from datasets import Dataset

with open("mnt\\datasets\\SI_2_KB_cleaned.json", "r") as f:
    data = json.load(f)

# %%
# Transform data to the required format
transformed_data = []

for item in data:
    disease_name = item["disease_name"]
    question = item["question"]
    answer = item["answer"]

    transformed_data.append(
        {
            "disease_name": disease_name,
            "question": question,
            "answer": answer,
        }
    )

# Convert to pandas DataFrame
df = pd.DataFrame(transformed_data)

# %%
# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Push to the Hub
dataset.push_to_hub("DermaVLM/self_instruct_2-kb_derma")

print(
    f"Successfully uploaded {len(transformed_data)} samples to DermaVLM/self_instruct_2-kb_derma"
)
print(f"Dataset has {len(dataset)} records.")
# %%
