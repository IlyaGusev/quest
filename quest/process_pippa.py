import json
import random

import fire
from datasets import load_dataset
from tqdm import tqdm


def revert_flattening(records):
    fixed_records = []
    for key, values in records.items():
        if not fixed_records:
            fixed_records = [{} for _ in range(len(values))]
        for i, value in enumerate(values):
            fixed_records[i][key] = value
    return fixed_records


def merge_bot_messages(messages):
    new_messages = []
    prev_role = None
    merge_count = 0
    for m in messages:
        role = "user" if m["is_human"] else "bot"
        if role != "bot" or prev_role != "bot":
            new_messages.append(m)
            merge_count = 0
        else:
            assert new_messages[-1]["is_human"] == False
            assert role == "bot"
            new_messages[-1]["message"] += "\n" + m["message"]
            merge_count += 1
        prev_role = role
    return new_messages


def process_pippa(
    output_path: str,
    dataset_name: str = "PygmalionAI/PIPPA",
    min_messages: int = 4,
    sample_rate: float = 0.01,
    min_last_message_length: int = 150,
):
    records = []
    for row in tqdm(load_dataset(dataset_name, split="train")):
        context = row["bot_description"]
        char_name = row["bot_name"].strip()
        messages = row["conversation"]
        if isinstance(messages, dict):
            messages = revert_flattening(messages)
        messages = merge_bot_messages(messages)
        if len(messages) < min_messages:
            continue

        prompt = row["bot_definitions"]
        prompt = prompt.split("END_OF_DIALOG")[0]
        prompt = prompt.replace("{{user}}", "User")
        prompt = prompt.replace("{{user}", "User")
        prompt = prompt.replace("{{u01}}", "User")
        for i in range(20):
            prompt = prompt.replace("{{random_user_" + str(i) + "}}", "User")
        prompt = prompt.replace("{{char}}", char_name)
        prompt = prompt.strip()

        found = False
        full_prompt = f"{char_name}'s Persona: {context}\n####\n{prompt}\n<START>\n"
        for i, message in enumerate(messages):
            role = "User" if message["is_human"] else char_name
            content = message["message"].strip()
            if content.startswith(char_name + ":"):
                content = content[len(char_name) + 1:].strip()
            if role == char_name and i <= 7 and i >= 3 and len(content) > min_last_message_length:
                if random.random() < sample_rate:
                    full_prompt += f"{role}:"
                    found = True
                    break
            full_prompt += f"{role}: {content}\n\n"
        if found:
            records.append({"prompt": full_prompt, "source": "pippa"})

    with open(output_path, "w") as w:
        for record in records:
            w.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(process_pippa)
