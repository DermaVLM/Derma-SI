from random import shuffle as random_shuffle


def sample_tasks_round_robin(task_pool, num_human, num_model):
    """Sample tasks round-robin style, ensuring every task is used in the iteration."""
    human_tasks = [
        task for task in task_pool if task.get("origin") == "human-generated"
    ]
    model_tasks = [
        task for task in task_pool if task.get("origin") == "model-generated"
    ]

    # Shuffle the tasks to ensure randomness
    random_shuffle(human_tasks)
    random_shuffle(model_tasks)

    total_tasks = len(task_pool)
    samples_per_prompt = num_human + num_model
    num_prompts = -(-total_tasks // samples_per_prompt)  # Ceiling division

    sampled_prompts = []
    human_index = 0
    model_index = 0

    for _ in range(num_prompts):
        prompt_sample = []

        for _ in range(num_human):
            if human_index < len(human_tasks):
                prompt_sample.append(human_tasks[human_index])
                human_index += 1
            elif model_index < len(model_tasks):
                prompt_sample.append(model_tasks[model_index])
                model_index += 1

        for _ in range(num_model):
            if model_index < len(model_tasks):
                prompt_sample.append(model_tasks[model_index])
                model_index += 1
            elif human_index < len(human_tasks):
                prompt_sample.append(human_tasks[human_index])
                human_index += 1

        sampled_prompts.append(prompt_sample)

    return sampled_prompts
