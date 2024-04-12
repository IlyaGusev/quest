import os
import re
import json
import shutil

import fire

from quest.utils import encode_prompt, read_jsonl
from quest.anthropic_wrapper import anthropic_completion


CRITERIA_TO_IGNORE = [
    "Appropriate Length",
    "Unearned Resolution: Characters' disagreements or tensions are too quickly or easily resolved, without exploring the depth or implications of the conflict.",
    "Melodramatic",
    "Clever / Witty",
    "Gripping",
    "Effective Use of Tropes: If applicable, common narrative tropes are employed thoughtfully and subverted, deconstructed, or used in service of the story's themes and character",
    "Correct Spelling & Grammar"
]

NEG_CRITERIA= [
    "melodramatic",
    "shallow resolution",
    "unearned resolution",  # old naming
    "simplistic moralizing",
    "shallow optimism",
    "forced optimism",  # old naming
    "trite",
    "overwrought",
    "amateurish",
    "contrived",
    "uninspiring",
    "characters are too good",
    "incongruent ending positivity",
    "unearned transformations",
    "profundity over-reach",
    "amateurish descriptives",
    "clunky asides and interruptive sentence structures",
    "stilted dialogue",
    "tit-for-tat dialogue"
]


def create_judging_prompt(criteria, writing_prompt, reference_output, test_model_response):
    criteria_str = "\n".join(criteria)
    prompt = encode_prompt(
        "quest/eq_bench_cw_judge_prompt.jinja",
        writing_prompt=writing_prompt,
        reference_output=reference_output,
        test_model_response=test_model_response,
        criteria_str=criteria_str
    )
    return prompt


def parse_scores(judge_model_response):
    scores = {}
    score_pattern = r'(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)'
    matches = re.findall(score_pattern, judge_model_response)
    for m in matches:
        metric_name = m[0].strip()
        score = float(m[1])
        scores[metric_name] = score
    return scores


def calc_prompt_score(scores):
    assert scores
    scoresum = 0
    for criteria, score in scores.items():
        criteria_lower = criteria.lower().strip()
        if any(neg_criterion in criteria_lower for neg_criterion in NEG_CRITERIA):
            scoresum += 10 - score
        else:
            scoresum += score
    final_score = scoresum / len(scores)
    print('This question score:', round(10 * final_score))
    return final_score


def process_output(writing_prompt, reference_output, test_model_response, judging_criteria, model_name):
    combined_criteria = []
    for criteria_set in judging_criteria:
        combined_criteria += criteria_set["criteria"]
    combined_criteria = list(reversed(combined_criteria))
    filtered_criteria = [x for x in combined_criteria if x not in CRITERIA_TO_IGNORE]
    judging_prompt = create_judging_prompt(filtered_criteria, writing_prompt, reference_output, test_model_response)
    judge_model_response = anthropic_completion(
        messages=[{"role": "user", "content": judging_prompt}],
        model_name=model_name,
        temperature=0.1
    )
    print(judge_model_response)
    scores = parse_scores(judge_model_response)
    print(scores)
    calc_prompt_score(scores)
    return judge_model_response, scores


def calc_full_score(results):
    prompt_scores = list()
    for _, result in results.items():
        prompt_scores.append(calc_prompt_score(result["scores"]))
    return round(10 * sum(prompt_scores) / len(prompt_scores), 2)


def eval_eq_bench_creative_writing(
    input_path: str,
    results_path: str,
    model_name: str = "claude-3-opus-20240229"
):
    results = dict()
    if os.path.exists(results_path):
        results = read_jsonl(results_path)
        results = {r["full_id"]: r for r in results}

    records = read_jsonl(input_path)
    for record in records:
        config_name = record["config_name"].replace(".json", "")
        sample_id = str(record["sample_id"])
        num_iteration = str(record["num_iteration"])
        full_id = f"{config_name}_{sample_id}_{num_iteration}"
        if full_id in results:
            full_score = calc_full_score(results)
            print("Full score: {}, num samples: {}".format(full_score, len(results)))
            continue
        response, scores = process_output(
            writing_prompt=record["prompt"],
            reference_output=record["reference_output"],
            test_model_response=record["output"],
            judging_criteria=record["judging_criteria"],
            model_name=model_name,
        )
        results[full_id] = {
            "full_id": full_id,
            "config_name": config_name,
            "sample_id": sample_id,
            "num_iteration": num_iteration,
            "response": response,
            "scores": scores
        }
        with open(results_path + "_tmp", "w") as w:
            for record in results.values():
                w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
        shutil.move(results_path + "_tmp", results_path)
        full_score = calc_full_score(results)
        print("Full score: {}, num samples: {}".format(full_score, len(results)))


if __name__ == "__main__":
    fire.Fire(eval_eq_bench_creative_writing)
