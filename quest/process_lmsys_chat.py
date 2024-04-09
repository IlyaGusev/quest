import random
import json

import fire
from datasets import load_dataset
from tqdm import tqdm

from quest.openai_wrapper import openai_completion


PROMPT = """Determine whether the prompt below is about creative writing or roleplay, and the completion to this prompt should be a medium-sized or long text.

If it is about creative writing or roleplay, return a JSON: {{"category": "creative_writing_or_roleplay"}}.
If it is is not about that, return a JSON: {{"category": "other"}}

###
Prompt: {task}
###
"""


def parse_json_output(output):
    start_index = output.find("{")
    end_index = output.rfind("}")
    text = output[start_index : end_index + 1]
    text = text.strip()
    record = json.loads(text)
    return record


def process_lmsys_chat(
    output_path: str,
    nrows: int = 200,
    sample_rate: float = 0.2,
    model_name: str = "gpt-4-turbo-preview"
):
    results = dict()
    count = 0
    with open(output_path, "w") as w:
        for row in tqdm(load_dataset("lmsys/lmsys-chat-1m", streaming=True, split="train")):
            if random.random() > sample_rate:
                continue
            if row["language"] != "English":
                continue
            prompt = row["conversation"][0]["content"]
            answer = row["conversation"][1]["content"]
            if prompt in results:
                continue
            if len(prompt) < 100 or len(prompt) > 1500:
                continue
            if "NAME" in prompt:
                continue
            if len(answer) < 100:
                continue
            if any(ss in prompt for ss in ("|", "GPT")):
                continue
            if len(prompt.split("\n")) > 30:
                continue
            response = openai_completion(
                [{"role": "user", "content": PROMPT.format(task=prompt)}],
                model_name=model_name
            )
            response = parse_json_output(response)
            category = response["category"]
            results[prompt] = category
            if "other" in category:
                continue
            print(prompt)
            count += 1
            w.write(json.dumps({
                "source": "lmsys",
                "prompt": prompt
            }, ensure_ascii=False) + "\n")
            if count == nrows:
                break


if __name__ == "__main__":
    fire.Fire(process_lmsys_chat)
