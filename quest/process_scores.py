import os
from collections import Counter

import torch
import fire


def process_scores(input_dir, pad_token_id: int = 0, k: int = 30):
    possible_cnt = Counter()
    real_cnt = Counter()
    all_tokens_count = 0
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if not file_path.endswith(".pt"):
            continue
        meta = torch.load(file_path)
        tokens_count = (meta["output_ids"] != 0).sum().item()
        all_tokens_count += tokens_count
        values = meta["scores_values"][:tokens_count]
        indices = meta["scores_indices"][:tokens_count]
        sampled_counts = (values != float("-Inf")).sum(dim=-1).tolist()
        for output_id, positions in zip(meta["output_ids"], indices):
            try:
                pos = positions.tolist().index(output_id.item()) + 1
            except ValueError:
                pos = k
            real_cnt[pos] += 1
        possible_cnt.update(sampled_counts)

    for idx in range(k, 0, -1):
        possible_cnt[idx] += possible_cnt[idx+1]

    print("Tokens count:", all_tokens_count)
    print("Possible positions:")
    possible_positions = [possible_cnt[idx]/all_tokens_count for idx in range(1, k+1)]
    print(possible_positions)

    print("Real positions:")
    real_positions = [real_cnt[idx]/all_tokens_count for idx in range(1, k+1)]
    print(real_positions)

    return possible_positions, real_positions


if __name__ == "__main__":
    fire.Fire(process_scores)
