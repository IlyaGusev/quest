import os
import json
from typing import List
from pathlib import Path

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from quest.utils import read_jsonl, set_random_seed, gen_batch
from quest.sampler_hijack import hijack_samplers

hijack_samplers()


def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    generation_config: GenerationConfig,
):
    formatted_prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False
    ) for prompt in prompts]
    if tokenizer.pad_token_id is None and generation_config.pad_token_id is not None:
        tokenizer.pad_token_id = generation_config.pad_token_id
    data = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True
    )
    data = {k: v.to(model.device) for k, v in data.items()}
    results = model.generate(
        **data,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True
    )
    output_ids = results.sequences
    logits = torch.stack(results.logits)
    scores = torch.stack(results.scores)

    outputs = []
    metas = []
    for i, (sample_output_ids, sample_input_ids) in enumerate(zip(output_ids, data["input_ids"])):
        sample_output_ids = sample_output_ids[len(sample_input_ids):]
        sample_output = tokenizer.decode(sample_output_ids, skip_special_tokens=True)
        outputs.append(sample_output)
        sample_logits = logits[:, i, :]
        sample_logits_top_values, sample_logits_top_indices = torch.topk(sample_logits, k=30, dim=-1)
        sample_scores = scores[:, i, :]
        sample_scores_top_values, sample_scores_top_indices = torch.topk(sample_scores, k=30, dim=-1)
        metas.append({
            "logits_values": sample_logits_top_values,
            "logits_indices": sample_logits_top_indices,
            "scores_values": sample_scores_top_values,
            "scores_indices": sample_scores_top_indices,
            "output_ids": sample_output_ids
        })

    return outputs, metas


def infer(
    input_path: str,
    output_path: str,
    model_name: str,
    generation_config_path: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    batch_size: int = 3,
    seed: int = 42
):
    set_random_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.eval()
    model = torch.compile(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    orig_generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config_path = Path(generation_config_path)
    generation_config = GenerationConfig.from_pretrained(
        generation_config_path.parent,
        generation_config_path.name
    )
    generation_config.pad_token_id = orig_generation_config.pad_token_id
    generation_config.eos_token_id = orig_generation_config.eos_token_id
    generation_config.bos_token_id = orig_generation_config.bos_token_id
    print("Full generation config:", generation_config)

    records = list(read_jsonl(input_path))
    meta_dir = ".".join(output_path.split(".")[:-1])
    Path(meta_dir).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as w:
        num = 0
        for batch in gen_batch(records, batch_size):
            prompts = [r["prompt"] for r in batch]
            outputs, metas = generate(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                generation_config=generation_config
            )
            for prompt, output in zip(prompts, outputs):
                print()
                print("=========")
                print(prompt)
                print()
                print("OUTPUT:")
                print(output)
                print("=========")
                print()
            for record, output, meta in zip(batch, outputs, metas):
                record["output"] = output
                record["model_name"] = model_name
                record["config_name"] = generation_config_path.name
                w.write(json.dumps(record, ensure_ascii=False) + "\n")
                torch.save(meta, os.path.join(meta_dir, f"{num}.pt"))
                num += 1


if __name__ == "__main__":
    fire.Fire(infer)
