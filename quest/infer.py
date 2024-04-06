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
    data = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True
    )
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config,
        pad_token_id=generation_config.eos_token_id
    )
    outputs = []
    for sample_output_ids, sample_input_ids in zip(output_ids, data["input_ids"]):
        sample_output_ids = sample_output_ids[len(sample_input_ids):]
        sample_output = tokenizer.decode(sample_output_ids, skip_special_tokens=True)
        outputs.append(sample_output)
    return outputs


def infer(
    input_path: str,
    model_name: str,
    generation_config_path: str,
    load_in_8bit: bool = True,
    load_in_4bit: bool = False,
    batch_size: int = 1,
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
    generation_config.eos_token_id = orig_generation_config.eos_token_id
    generation_config.bos_token_id = orig_generation_config.bos_token_id

    records = list(read_jsonl(input_path))
    for batch in gen_batch(records, batch_size):
        prompts = [r["prompt"] for r in batch]
        outputs = generate(
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


if __name__ == "__main__":
    fire.Fire(infer)
