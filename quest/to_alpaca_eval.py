import json
from typing import Optional

import fire

from quest.utils import read_jsonl


def postprocess(record):
    source = record["source"]
    output = record["output"]
    if source == "pippa":
        output = output.split("User:")[0]
        output = output.strip()
        if len(output) > 500:
            output = output.split("\n")[0]
            output = output.strip()
    record["output"] = output
    return record


def to_alpaca_eval(records):
    records = [postprocess(r) for r in records]
    records = [{
        "instruction": r["prompt"],
        "output": r["output"],
        "generator": r["config_name"].replace(".json", ""),
    } for r in records]
    return records


def to_alpaca_eval_main(
    input_path: str,
    output_path: str,
    nrows: Optional[int] = None,
    source: Optional[str] = None
):
    records = list(read_jsonl(input_path))
    if source:
        records = [r for r in records if r["source"] == source]
    if nrows:
        records = records[:nrows]
    records = to_alpaca_eval(records)
    with open(output_path, "w") as w:
        json.dump(records, w, indent=4)


if __name__ == "__main__":
    fire.Fire(to_alpaca_eval)
