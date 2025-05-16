# Derm-SI

Derm-SI (Dermatology Self Instruct) is a research toolkit for generating high-quality synthetic dermatological data using self-instructed large language models. The toolkit enables the creation of dermatology knowledge bases, clinical cases, and evaluation benchmarks to improve medical AI training and evaluation.

## Overview

This repository contains a collection of tools for generating diverse dermatological content:

1. **Knowledge Base Generation** - Create comprehensive dermatology knowledge bases with structured information about skin conditions
2. **Clinical Case Generation** - Generate realistic patient anamnesis cases with detailed Q&A between doctor and patient
3. **Evaluation Benchmark Creation** - Develop dermatology text benchmarks for evaluating medical AI systems

## Repository Structure

```
derm_self_instruct/
├── derm_self_instruct/
│   ├── anamnesis_cases_si/          # Clinical case generation pipeline
│   ├── knowledge_base_utils/        # Knowledge base generation utilities
│   ├── text_eval_generation/        # Evaluation benchmark generation tools
│   └── ...
├── mnt/                             # Data mount directory
│   ├── SI_3/                        # Self-instruct run outputs
│   └── seed_tasks...
├── main_SI_2.py                     # Knowledge base generation entry point
└── SI_3_upload_to_hf.py             # Hugging Face dataset upload scripts
```

## Modules

### Knowledge Base Generation (`knowledge_base_utils`)

The `DermDatasetGenerator` class creates comprehensive dermatology knowledge using a structured template system:

```python
generator = DermDatasetGenerator(
    api_key=api_key,
    gemini_model="gemini-2.0-flash",
    output_dir="mnt/output_SI_2",
)

generator.generate_dataset(
    kb_path="mnt/processed_dermatology_kb.json",
    templates_path="mnt/disease_seed_tasks_eg.json"
)
```

### Clinical Case Generation (`anamnesis_cases_si`)

The `DermCaseSelfInstruct` class generates realistic dermatology cases with detailed patient-doctor interactions:

```python
case_generator = DermCaseSelfInstruct(
    api_key=api_key,
    output_dir="output",
    gemini_model="gemini-2.0-flash"
)

cases = case_generator.run(
    seed_data_path="mnt/SI_3/seed_data_anamnesis.json",
    output_path="mnt/SI_3/output/all_cases.json",
    num_cases=1000
)
```

### Evaluation Benchmark Generation (`text_eval_generation`)

The `DermatologyTextBenchmark` and `DermatologyBenchmarkGenerator` classes create dermatology knowledge benchmarks for evaluating LLMs:

```python
benchmark_generator = DermatologyTextBenchmark()
prompts = benchmark_generator.generate_balanced_benchmark(
    num_questions=2000,
    output_path="mnt/text_benchmark_prompts.json",
)
```

## Data

The repository works with several types of dermatology data:

1. **Knowledge Base** - Structured information about skin diseases and conditions
2. **Clinical Cases** - Patient cases with detailed Q&A structures
3. **Evaluation Benchmarks** - Multiple choice questions testing dermatology knowledge

## Hugging Face Integration

Generated datasets can be uploaded to Hugging Face using the `SI_3_upload_to_hf.py` script:

```python
dataset = prepare_dataset("mnt/SI_3/output/run_20250315_123633/generated_cases.json")
dataset_url = upload_to_huggingface(dataset, "DermaVLM/anamnesis_cases_1k")
```
