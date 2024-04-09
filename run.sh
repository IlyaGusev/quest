#python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_default.jsonl "openchat/openchat-3.5-0106" configs/default.json

#python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_top_p_95.jsonl "openchat/openchat-3.5-0106" configs/top_p_95.json
#python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_min_p_05.jsonl "openchat/openchat-3.5-0106" configs/min_p_05.json

python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_top_p_98.jsonl "openchat/openchat-3.5-0106" configs/top_p_98.json
python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_min_p_02.jsonl "openchat/openchat-3.5-0106" configs/min_p_02.json

python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_top_p_90.jsonl "openchat/openchat-3.5-0106" configs/top_p_90.json
python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_min_p_10.jsonl "openchat/openchat-3.5-0106" configs/min_p_10.json

python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_top_p_80.jsonl "openchat/openchat-3.5-0106" configs/top_p_80.json
python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_min_p_20.jsonl "openchat/openchat-3.5-0106" configs/min_p_20.json
