python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_temp100.jsonl "openchat/openchat-3.5-0106" configs/temp100.json

python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_temp100_topp_98.jsonl "openchat/openchat-3.5-0106" configs/temp100_topp_98.json
python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_temp100_minp_02.jsonl "openchat/openchat-3.5-0106" configs/temp100_minp_02.json

python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_temp100_topp_95.jsonl "openchat/openchat-3.5-0106" configs/temp100_topp_95.json
python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_temp100_minp_05.jsonl "openchat/openchat-3.5-0106" configs/temp100_minp_05.json

python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_temp100_topp_90.jsonl "openchat/openchat-3.5-0106" configs/temp100_topp_90.json
python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_temp100_minp_10.jsonl "openchat/openchat-3.5-0106" configs/temp100_minp_10.json
