import os
from collections import Counter

import torch
import fire


def process_scores(input_dir, pad_token_id: int = 0):
    possible_cnt = Counter()
    real_cnt = Counter()
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if not file_path.endswith(".pt"):
            continue
        meta = torch.load(file_path)
        tokens_count = (meta["output_ids"] != 0).sum().item()
        values = meta["scores_values"][:tokens_count]
        indices = meta["scores_indices"][:tokens_count]
        sampled_counts = (values != float("-Inf")).sum(dim=-1).tolist()
        for output_id, positions in zip(meta["output_ids"], indices):
            try:
                pos = positions.tolist().index(output_id.item())
            except ValueError:
                pos = 30
            real_cnt[pos] += 1
        possible_cnt.update(sampled_counts)

    print("Tokens count:", sum(possible_cnt.values()))
    print("Possible positions:")
    print(sorted(possible_cnt.items()))

    print("Real positions:")
    print(sorted(real_cnt.items()))


if __name__ == "__main__":
    fire.Fire(process_scores)
