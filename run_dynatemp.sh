python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_dynatemp_50_150_100.jsonl "openchat/openchat-3.5-0106" configs/dynatemp_50_150_100.json

python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_dynatemp_50_150_75_minp_05.jsonl "openchat/openchat-3.5-0106" configs/dynatemp_50_150_75_minp_05.json

python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_dynatemp_100_250_100_minp_10.jsonl "openchat/openchat-3.5-0106" configs/dynatemp_100_250_100_minp_10.json

python3 -m quest.infer data/prompts/all.jsonl data/outputs/7b_dynatemp_50_300_100_minp_20.jsonl "openchat/openchat-3.5-0106" configs/dynatemp_50_300_100_minp_20.json
