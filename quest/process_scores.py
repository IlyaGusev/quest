import os
from collections import Counter

import torch
import fire


def process_scores(input_dir, pad_token_id: int = 0):
    cnt = Counter()
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if not file_path.endswith(".pt"):
            continue
        meta = torch.load(file_path)
        tokens_count = (meta["output_ids"] != 0).sum().item()
        values = meta["scores_values"][:tokens_count]
        sampled_counts = (values != float("-Inf")).sum(dim=-1).tolist()
        cnt.update(sampled_counts)
    print(cnt)


if __name__ == "__main__":
    fire.Fire(process_scores)
