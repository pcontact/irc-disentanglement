#!/usr/bin/env python3

import json
import os
import random
from collections import OrderedDict

import torch
from torch.utils.data import Dataset


class PrecomputedDataset(Dataset):
    def __init__(self, precomputed_dir, split, random_sample=None, seed=10, cache_size=4):
        self.precomputed_dir = precomputed_dir
        self.split = split
        self.cache_size = cache_size
        self.cache = OrderedDict()

        manifest_path = os.path.join(precomputed_dir, "manifest.jsonl")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError("Missing manifest at {}".format(manifest_path))

        entries = []
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("split") != split:
                    continue
                path = os.path.join(precomputed_dir, record["path"])
                num_samples = int(record["num_samples"])
                for i in range(num_samples):
                    entries.append((path, i))

        if random_sample:
            random.seed(seed)
            random.shuffle(entries)
            entries = entries[: int(random_sample)]

        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def _load_file(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        data = torch.load(path, map_location="cpu")
        self.cache[path] = data
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return data

    def __getitem__(self, idx):
        path, sample_idx = self.entries[idx]
        data = self._load_file(path)
        sample = data["samples"][sample_idx]
        sample = dict(sample)
        sample["name"] = data["name"]
        return sample


def collate_batch(batch):
    if len(batch) == 0:
        return {}

    feat_dim = batch[0]["features"].shape[1]
    max_cand = max(sample["features"].shape[0] for sample in batch)
    max_q_len = max(sample.get("query_len", 0) for sample in batch)
    max_c_len = 0
    for sample in batch:
        cand_lens = sample.get("cand_lens", [])
        if len(cand_lens) > 0:
            max_c_len = max(max_c_len, max(cand_lens))

    batch_size = len(batch)
    batch_feat = torch.zeros(batch_size, max_cand, feat_dim, dtype=torch.float32)
    batch_q_ids = torch.zeros(batch_size, max_q_len, dtype=torch.long)
    batch_c_ids = torch.zeros(batch_size, max_cand, max_c_len, dtype=torch.long)
    query_lens = torch.zeros(batch_size, dtype=torch.long)
    cand_lens = torch.zeros(batch_size, max_cand, dtype=torch.long)
    cand_mask = torch.zeros(batch_size, max_cand, dtype=torch.float32)
    gold_mask = torch.zeros(batch_size, max_cand, dtype=torch.float32)
    orig_gold_count = torch.zeros(batch_size, dtype=torch.long)
    names = []
    query_indices = []

    for i, sample in enumerate(batch):
        features = sample["features"]
        n_cand = features.shape[0]
        batch_feat[i, :n_cand] = features
        cand_mask[i, :n_cand] = 1.0

        q_ids = sample.get("query_ids", [])
        q_len = sample.get("query_len", len(q_ids))
        query_lens[i] = q_len
        if q_len > 0 and max_q_len > 0:
            batch_q_ids[i, :q_len] = torch.tensor(q_ids, dtype=torch.long)

        c_ids = sample.get("cand_ids", [])
        c_lens = sample.get("cand_lens", [])
        if len(c_lens) > 0:
            cand_lens[i, :n_cand] = torch.tensor(c_lens, dtype=torch.long)
        for j in range(min(n_cand, len(c_ids))):
            if max_c_len > 0 and len(c_ids[j]) > 0:
                length = min(len(c_ids[j]), max_c_len)
                batch_c_ids[i, j, :length] = torch.tensor(c_ids[j][:length], dtype=torch.long)

        for gi in sample.get("gold_indices", []):
            if gi < max_cand:
                gold_mask[i, gi] = 1.0

        orig_gold_count[i] = int(sample.get("orig_gold_count", 0))
        names.append(sample.get("name"))
        query_indices.append(int(sample.get("query_index")))

    return {
        "features": batch_feat,
        "query_ids": batch_q_ids,
        "cand_ids": batch_c_ids,
        "query_lens": query_lens,
        "cand_lens": cand_lens,
        "cand_mask": cand_mask,
        "gold_mask": gold_mask,
        "orig_gold_count": orig_gold_count,
        "names": names,
        "query_indices": query_indices,
    }

