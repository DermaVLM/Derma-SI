import json


def load_seed_data(filepath):
    """Load initial seed data from a JSON file."""
    try:
        with open(filepath, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []


def save_task_pool(filepath, task_pool):
    """Save the updated task pool to a JSON file."""
    with open(filepath, "w") as file:
        json.dump(task_pool, file, indent=4)
