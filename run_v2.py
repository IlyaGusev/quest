import os
import fire
from transformers import AutoModelForCausalLM
import torch

from quest.infer import infer

temperatures = [1.4, 1.0, 1.8, 2.2, 0.6]
top_p = [0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
min_p = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4]


BORING_OUTPUTS = [
    "openchat_temp_20",
    "openchat_temp_20_top_p_98",
    "openchat_temp_20_top_p_95",
    "openchat_temp_20_top_p_90",
    "openchat_temp_20_top_p_80",
    "openchat_temp_20_top_p_70",
    "openchat_temp_20_min_p_02",
    "openchat_temp_20_min_p_05",
    "openchat_temp_20_min_p_10",
    "openchat_temp_20_min_p_20",
    "openchat_temp_20_min_p_30",
    "openchat_temp_20_min_p_40",
    "openchat_temp_60_min_p_20",
    "openchat_temp_60_min_p_30",
    "openchat_temp_60_min_p_40",
    "openchat_temp_60_top_p_60",
    "openchat_temp_60_top_p_70"
]
CRAZY_OUTPUTS = [
    "openchat_temp_140",
    "openchat_temp_180",
    "openchat_temp_220",
    "openchat_temp_140_top_p_70",
    "openchat_temp_140_top_p_80",
    "openchat_temp_140_top_p_90",
    "openchat_temp_140_top_p_95",
    "openchat_temp_140_top_p_98",
    "openchat_temp_180_top_p_60",
    "openchat_temp_180_top_p_70",
    "openchat_temp_180_top_p_80",
    "openchat_temp_180_top_p_90",
    "openchat_temp_180_top_p_95",
    "openchat_temp_180_top_p_98",
    "openchat_temp_180_min_p_02",
    "openchat_temp_180_min_p_05",
    "openchat_temp_220_top_p_60",
    "openchat_temp_220_top_p_70",
    "openchat_temp_220_top_p_80",
    "openchat_temp_220_top_p_90",
    "openchat_temp_220_top_p_95",
    "openchat_temp_220_top_p_98",
    "openchat_temp_220_min_p_02",
    "openchat_temp_220_min_p_05",
    "openchat_temp_220_min_p_10"
]

def get_output_name(model_slug, temperature, top_p, min_p):
    name = [model_slug]
    name.append("temp_{}".format(int(temperature * 100)))
    if top_p is not None:
        name.append("top_p_{:02d}".format(int(top_p * 100)))
    if min_p is not None:
        name.append("min_p_{:02d}".format(int(min_p * 100)))
    return = "_".join(name)


def run_exp(
    model_name: str,
    model_slug: str,
    temperature: float,
    top_p: float = None,
    min_p: float = None,
    model: AutoModelForCausalLM = None
):
    output_name = get_output_name(
        model_slug=model_slug,
        temperature=temperature,
        top_p=top_p,
        min_p=min_p
    )
    if output_name in BORING_OUTPUTS or output_name in CRAZY_OUTPUTS:
        print(f"skipping {output_name}")
        return

    output_path = f"data/outputs/v2/{output_name}.jsonl"
    if os.path.exists(output_path):
        print(f"{output_path} exists, skipping!")
        return
    infer(
        input_path="data/prompts/all_v2.jsonl",
        output_path=output_path,
        model_name=model_name,
        generation_config_path="configs/temp100.json",
        temperature=temperature,
        top_p=top_p,
        min_p=min_p,
        model=model
    )


def run_all_exps(
    model_name: str = "openchat/openchat-3.5-0106",
    model_slug: str = "openchat",
):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    model.eval()
    model = torch.compile(model)

    for temp in temperatures:
        run_exp(temperature=temp, model_name=model_name, model_slug=model_slug, model=model)
        for tp in top_p:
            run_exp(temperature=temp, top_p=tp, model_name=model_name, model_slug=model_slug, model=model)
        for mp in min_p:
            run_exp(temperature=temp, min_p=mp, model_name=model_name, model_slug=model_slug, model=model)


if __name__ == "__main__":
    fire.Fire(run_all_exps)
