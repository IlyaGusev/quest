python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_temp80.jsonl "openchat/openchat-3.5-0106" configs/temp80.json

python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_temp80_topp_95.jsonl "openchat/openchat-3.5-0106" configs/temp80_topp_95.json
python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_temp80_minp_05.jsonl "openchat/openchat-3.5-0106" configs/temp80_minp_05.json

python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_temp80_topp_98.jsonl "openchat/openchat-3.5-0106" configs/temp80_topp_98.json
python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_temp80_minp_02.jsonl "openchat/openchat-3.5-0106" configs/temp80_minp_02.json
