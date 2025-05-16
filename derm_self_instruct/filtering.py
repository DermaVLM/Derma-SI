import logging
import json

from sklearn.metrics.pairwise import cosine_similarity
from derm_self_instruct.models import get_embedding_model

LOGGER = logging.getLogger(__name__)

embedding_model = get_embedding_model()


def get_task_text(task):
    """
    Get concatenated text representation of a task including instruction, input, and output.
    This provides a more comprehensive basis for similarity comparison.
    """
    instruction = task.get("instruction", "")

    # Get the first instance's input and output
    input_text = ""
    output_text = ""
    if task.get("instances") and len(task["instances"]) > 0:
        input_text = task["instances"][0].get("input", "")
        output_text = task["instances"][0].get("output", "")

    # Concatenate with separators
    return f"{instruction} [SEP] {input_text} [SEP] {output_text}"


def calculate_similarity(text1, text2):
    """Calculate cosine similarity between two texts."""
    embeddings = embedding_model.encode([text1, text2], show_progress_bar=False)
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]


def filter_tasks(task_pool, new_tasks, similarity_threshold=0.7):
    """
    Filter tasks based on heuristics and similarity with batch embedding
    using concatenated instruction + input + output.
    """
    # Apply basic heuristic filtering first
    basic_filtered_tasks = []

    for new_task in new_tasks:
        instruction = new_task["instruction"]

        # Get output from first instance if available
        output_text = ""
        if new_task.get("instances") and len(new_task["instances"]) > 0:
            output_text = new_task["instances"][0].get("output", "")

        # Heuristic Filtering
        if len(instruction) < 10:
            LOGGER.debug(f"Filtered out task with short instruction: '{instruction}'")
            continue
        if len(output_text) < 1:
            LOGGER.debug(f"Filtered out task with empty output")
            continue
        if any(
            keyword in instruction.lower()
            for keyword in ["image", "graph", "figure", "images"]
        ):
            LOGGER.debug(f"Filtered out task with unsupported keyword: '{instruction}'")
            continue
        if instruction == output_text:
            LOGGER.debug(
                f"Filtered out task where instruction equals output: '{instruction}'"
            )
            continue

        basic_filtered_tasks.append(new_task)

    # If no tasks passed basic filtering, return empty list
    if not basic_filtered_tasks:
        return []

    # Get concatenated text representations for all tasks
    pool_texts = [get_task_text(task) for task in task_pool]
    new_texts = [get_task_text(task) for task in basic_filtered_tasks]

    if pool_texts:  # Only if there are existing tasks
        # Compute embeddings for all tasks in one batch
        all_texts = pool_texts + new_texts
        all_embeddings = embedding_model.encode(all_texts, show_progress_bar=True)

        pool_embeddings = all_embeddings[: len(pool_texts)]
        new_embeddings = all_embeddings[len(pool_texts) :]

        # Check similarity against existing pool
        similarity_filtered_tasks = []
        for i, (task, embedding) in enumerate(
            zip(basic_filtered_tasks, new_embeddings)
        ):
            # Calculate similarities with existing pool
            similarities = cosine_similarity([embedding], pool_embeddings)[0]
            if not any(sim > similarity_threshold for sim in similarities):
                similarity_filtered_tasks.append(task)

        # Check similarity within the new batch itself
        final_filtered_tasks = []
        final_embeddings = []

        for i, (task, embedding) in enumerate(
            zip(
                similarity_filtered_tasks,
                new_embeddings[-len(similarity_filtered_tasks) :],
            )
        ):
            # Compare with already selected tasks
            if final_embeddings:
                similarities = cosine_similarity([embedding], final_embeddings)[0]
                if any(sim > similarity_threshold for sim in similarities):
                    continue

            final_filtered_tasks.append(task)
            final_embeddings.append(embedding)
    else:
        # If no existing tasks, just check similarity within the new batch
        new_embeddings = embedding_model.encode(new_texts, show_progress_bar=True)
        final_filtered_tasks = []
        final_embeddings = []

        for i, (task, embedding) in enumerate(
            zip(basic_filtered_tasks, new_embeddings)
        ):
            # Compare with already selected tasks
            if final_embeddings:
                similarities = cosine_similarity([embedding], final_embeddings)[0]
                if any(sim > similarity_threshold for sim in similarities):
                    continue

            final_filtered_tasks.append(task)
            final_embeddings.append(embedding)

    LOGGER.info(
        f"Filtered {len(new_tasks) - len(final_filtered_tasks)} tasks out of {len(new_tasks)}."
    )

    return final_filtered_tasks


def filter_large_dataset(
    input_file, output_file, similarity_threshold=0.7, batch_size=1000
):
    """
    Filter a large dataset by processing it in batches to avoid memory issues.

    Parameters:
    - input_file: Path to input JSON file containing tasks
    - output_file: Path to save filtered tasks
    - similarity_threshold: Threshold for similarity filtering
    - batch_size: Number of tasks to process in each batch
    """
    # Load the input dataset
    print(f"Loading dataset from {input_file}...")
    with open(input_file, "r") as f:
        tasks = json.load(f)

    if isinstance(tasks, dict) and "tasks" in tasks:
        tasks = tasks["tasks"]

    total_tasks = len(tasks)
    print(f"Loaded {total_tasks} tasks")

    # Initialize with empty task pool
    filtered_pool = []

    # Process in batches
    for i in range(0, total_tasks, batch_size):
        batch = tasks[i : min(i + batch_size, total_tasks)]
        print(
            f"Processing batch {i//batch_size + 1}/{(total_tasks-1)//batch_size + 1} ({len(batch)} tasks)"
        )

        # Filter current batch against the accumulated filtered pool
        newly_filtered = filter_tasks(filtered_pool, batch, similarity_threshold)
        filtered_pool.extend(newly_filtered)

        print(f"Batch filtered: {len(newly_filtered)}/{len(batch)} tasks accepted")
        print(
            f"Total filtered so far: {len(filtered_pool)}/{min(i+batch_size, total_tasks)} tasks"
        )

    # Save final result
    print(f"Saving {len(filtered_pool)} filtered tasks to {output_file}")
    with open(output_file, "w") as f:
        json.dump(filtered_pool, f, indent=2)

    print(f"Filtering complete: {len(filtered_pool)}/{total_tasks} tasks retained")
    return filtered_pool


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    input_file = "mnt/task_pool_checkpoint.json"
    output_file = "mnt/task_pool_checkpoint_filtered.json"
    similarity_threshold = 0.95
    batch_size = 2000

    # Run the filtering
    filter_large_dataset(
        input_file=input_file,
        output_file=output_file,
        similarity_threshold=similarity_threshold,
        batch_size=batch_size,
    )
