#!/usr/bin/env python3

import json
import os
import pickle
import random
import sys
from collections import OrderedDict, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.common import SCHEMA_VERSION, compute_embedding_hash


class PrecomputedDataset:
    def __init__(
        self,
        precomputed_dir: str,
        split: str,
        max_dist: int,
        test_start: int,
        test_end: int,
        word_vectors: Optional[str] = None,
        random_sample: Optional[str] = None,
        seed: int = 10,
        cache_size: int = 4,
    ):
        self.precomputed_dir = precomputed_dir
        self.split = split
        self.max_dist = max_dist
        self.test_start = test_start
        self.test_end = test_end
        self.word_vectors = word_vectors
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.embedding_hash = compute_embedding_hash(word_vectors)

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
                self._validate_record(record)
                path = os.path.join(precomputed_dir, record["path"])
                num_samples = int(record["num_samples"])
                payload = self._load_file(path)
                samples = payload.get("samples", [])
                if num_samples != len(samples):
                    raise ValueError(
                        "Manifest sample count mismatch for {}: {} vs {}".format(path, num_samples, len(samples))
                    )
                for index, sample in enumerate(samples):
                    entries.append(
                        {
                            "path": path,
                            "sample_index": index,
                            "num_candidates": len(sample["candidate_indices"]),
                            "query_index": sample["query_index"],
                        }
                    )

        if random_sample:
            rng = random.Random(seed)
            rng.shuffle(entries)
            entries = entries[: int(random_sample)]

        self.entries = entries

    def _validate_record(self, record: Dict) -> None:
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                "Incompatible schema version for {}: expected {}, got {}".format(
                    record.get("path"), SCHEMA_VERSION, record.get("schema_version")
                )
            )
        if int(record.get("max_dist")) != int(self.max_dist):
            raise ValueError(
                "Incompatible max_dist for {}: expected {}, got {}".format(
                    record.get("path"), self.max_dist, record.get("max_dist")
                )
            )
        if int(record.get("test_start")) != int(self.test_start) or int(record.get("test_end")) != int(self.test_end):
            raise ValueError(
                "Incompatible test window for {}: expected [{}, {}), got [{}, {})".format(
                    record.get("path"),
                    self.test_start,
                    self.test_end,
                    record.get("test_start"),
                    record.get("test_end"),
                )
            )
        if record.get("embedding_hash") != self.embedding_hash:
            raise ValueError(
                "Embedding hash mismatch for {}: expected {}, got {}".format(
                    record.get("path"), self.embedding_hash, record.get("embedding_hash")
                )
            )

    def _load_file(self, path: str):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        self.cache[path] = payload
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return payload

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        payload = self._load_file(entry["path"])
        sample = payload["samples"][entry["sample_index"]]
        message_token_ids = payload.get("message_token_ids", [])
        candidate_indices = list(sample["candidate_indices"])
        query_index = int(sample["query_index"])
        return {
            "name": payload["name"] + ".annotation.txt",
            "query_index": query_index,
            "candidate_indices": candidate_indices,
            "features": sample["features"],
            "gold_indices": list(sample.get("gold_indices", [])),
            "orig_gold_count": int(sample.get("orig_gold_count", 0)),
            "query_token_ids": message_token_ids[query_index] if message_token_ids else [],
            "candidate_token_ids": [message_token_ids[i] for i in candidate_indices] if message_token_ids else [],
        }

    def num_candidates_at(self, idx: int) -> int:
        return int(self.entries[idx]["num_candidates"])


def bucket_indices(dataset: PrecomputedDataset, indices: Optional[Sequence[int]] = None):
    grouped = defaultdict(list)
    items = indices if indices is not None else range(len(dataset))
    for idx in items:
        grouped[dataset.num_candidates_at(idx)].append(idx)
    return grouped


def iter_batches(
    dataset: PrecomputedDataset,
    batch_size: int,
    shuffle: bool = False,
    seed: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
):
    grouped = bucket_indices(dataset, indices=indices)
    keys = list(grouped.keys())
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(keys)
    else:
        keys.sort()

    for key in keys:
        bucket = list(grouped[key])
        if shuffle:
            rng.shuffle(bucket)
        for start in range(0, len(bucket), batch_size):
            batch_indices = bucket[start : start + batch_size]
            yield [dataset[idx] for idx in batch_indices]
