import json

import fire


def convert(input_path, output_path):
    with open(input_path) as r, open(output_path, "w") as w:
        for sample_id, prompt_data in json.load(r).items():
            system_prompt = "You are a talented creative writer of compelling, original prose.\n\n"
            prompt = system_prompt + prompt_data["writing_prompt"]
            for num_iteration, seed in enumerate(prompt_data["seed_modifiers"]):
                seeded_prompt = prompt.replace("<SEED>", seed)
                w.write(json.dumps({
                    "source": "eq_bench_creative_writing",
                    "prompt": seeded_prompt,
                    "sample_id": int(sample_id),
                    "num_iteration": num_iteration,
                    "seed_modifier": seed,
                    "reference_output": prompt_data["reference_output"],
                    "judging_criteria": prompt_data["judging_criteria"]
                }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(convert)
