import json
import random

import fire


def process_gpteacher(
    input_path: str,
    output_path: str,
    nrows: int = 200,
    min_output_length: int = 300,
):
    with open(input_path) as r:
        records = json.load(r)

    records = [r for r in records if len(r["output"]) >= min_output_length]

    random.shuffle(records)
    records = records[:nrows]
    clean_records = []
    with open(output_path, "w") as w:
        for record in records:
            prompt = record["instruction"]
            inp = record["input"]
            if inp:
                prompt += f"\n\nUser: {inp}\nCharacter:"
            w.write(json.dumps({"prompt": prompt, "source": "gpteacher_v2"}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(process_gpteacher)
