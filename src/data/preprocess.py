#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import sys
import time
from typing import Iterable, List, Optional, Sequence

import numpy as np

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.common import (
    FEATURES,
    SCHEMA_VERSION,
    clear_feature_cache,
    compute_embedding_hash,
    get_features,
    get_ids,
    get_log_path,
    header,
    load_conversations,
    load_embeddings,
    safe_name,
    write_manifest,
)


def add_shared_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("prefix", help="Start of names for files produced.")

    parser.add_argument("--train", nargs="+", help="Training files, e.g. train/*annotation.txt")
    parser.add_argument("--dev", nargs="+", help="Development files, e.g. dev/*annotation.txt")
    parser.add_argument("--test", nargs="+", help="Test files, e.g. test/*annotation.txt")
    parser.add_argument(
        "--test-start",
        type=int,
        default=1000,
        help="The line to start making predictions from in each test file.",
    )
    parser.add_argument(
        "--test-end",
        type=int,
        default=1000000,
        help="The line to stop making predictions on in each test files.",
    )
    parser.add_argument("--model", help="A file containing a trained model")
    parser.add_argument(
        "--random-sample",
        help="Train on only a random sample of the data with this many examples.",
    )

    parser.add_argument("--hidden", default=512, type=int, help="Number of dimensions in hidden vectors.")
    parser.add_argument("--word-vectors", help="File containing word embeddings.")
    parser.add_argument("--layers", default=2, type=int, help="Number of hidden layers in the model")
    parser.add_argument(
        "--nonlin",
        choices=["tanh", "cube", "logistic", "relu", "elu", "selu", "softsign", "swish", "linear"],
        default="softsign",
        help="Non-linearity type.",
    )

    parser.add_argument(
        "--max-dist",
        default=101,
        type=int,
        help="Maximum number of messages to consider when forming a link (count includes the current message).",
    )
    parser.add_argument(
        "--dynet-autobatch",
        action="store_true",
        help="Legacy alias retained for CLI compatibility.",
    )

    parser.add_argument("--report-freq", default=5000, type=int, help="How frequently to evaluate on the development set.")
    parser.add_argument("--epochs", default=20, type=int, help="Maximum number of epochs.")
    parser.add_argument("--opt", choices=["sgd", "mom", "adam"], default="sgd", help="Optimisation method.")
    parser.add_argument("--seed", default=10, type=int, help="Random seed.")
    parser.add_argument("--weight-decay", default=1e-7, type=float, help="Apply weight decay.")
    parser.add_argument("--learning-rate", default=0.018804, type=float, help="The initial learning rate.")
    parser.add_argument(
        "--learning-decay-rate",
        default=0.103,
        type=float,
        help="The rate at which the learning rate decays.",
    )
    parser.add_argument("--momentum", default=0.1, type=float, help="Hyperparameter for momentum.")
    parser.add_argument("--drop", default=0.0, type=float, help="Dropout, applied to inputs only.")
    parser.add_argument("--clip", default=3.740, type=float, help="Gradient clipping.")

    parser.add_argument(
        "--precomputed-dir",
        default=os.path.join("data", "precomputed_dynet"),
        help="Directory to write/read precomputed DyNet artifacts.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IRC Conversation Disentangler (DyNet Precompute).")
    return add_shared_args(parser)


def split_files(args, split_name: str):
    if split_name == "train":
        return args.train
    if split_name == "dev":
        return args.dev
    if split_name == "test":
        return args.test
    raise ValueError("Unknown split {}".format(split_name))


def split_is_test(split_name: str) -> bool:
    return split_name == "test"


def split_dir(precomputed_dir: str, split_name: str) -> str:
    return os.path.join(precomputed_dir, split_name)


def build_conversation_payload(conv, split_name: str, args, token_to_id, embedding_hash):
    message_token_ids = [get_ids(tokens, token_to_id) for tokens in conv["text_tok"]]
    message_users = [item[0] for item in conv["info"]]
    message_targets = [sorted(item[1]) for item in conv["info"]]
    prev_from_user = [item[6] for item in conv["info"]]
    next_from_user = [item[8] for item in conv["info"]]

    samples = []
    for query, gold_values in conv["links"].items():
        gold = list(gold_values)
        candidate_indices = list(range(query, max(-1, query - args.max_dist), -1))
        if split_name == "train" and len(candidate_indices) == 1:
            continue

        gold_filtered = [value for value in gold if value > query - args.max_dist]
        if split_name == "train" and len(gold_filtered) == 0:
            continue

        features = []
        for candidate in candidate_indices:
            features.append(
                get_features(
                    conv["name"],
                    query,
                    candidate,
                    conv["text_ascii"],
                    conv["info"],
                    conv["target_info"],
                    do_cache=True,
                )
            )

        samples.append(
            {
                "query_index": query,
                "candidate_indices": candidate_indices,
                "features": np.asarray(features, dtype=np.float32),
                "gold_indices": [query - value for value in gold_filtered],
                "orig_gold_count": len(gold),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "split": split_name,
        "name": conv["name"],
        "max_dist": args.max_dist,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "embedding_hash": embedding_hash,
        "feature_dim": FEATURES,
        "message_token_ids": message_token_ids,
        "metadata": {
            "users": message_users,
            "targets": message_targets,
            "prev_from_user": prev_from_user,
            "next_from_user": next_from_user,
        },
        "samples": samples,
    }


def precompute_split(split_name: str, args, token_to_id, embedding_hash, log_file=None) -> List[dict]:
    filenames = split_files(args, split_name)
    if not filenames:
        return []

    conversations = load_conversations(
        filenames,
        is_test=split_is_test(split_name),
        test_start=args.test_start,
        test_end=args.test_end,
    )
    out_dir = split_dir(args.precomputed_dir, split_name)
    os.makedirs(out_dir, exist_ok=True)

    manifest_records = []
    for conv in conversations:
        payload = build_conversation_payload(conv, split_name, args, token_to_id, embedding_hash)
        if len(payload["samples"]) == 0:
            continue

        file_name = safe_name(conv["name"]) + ".pkl"
        file_path = os.path.join(out_dir, file_name)
        with open(file_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

        manifest_records.append(
            {
                "path": os.path.relpath(file_path, args.precomputed_dir),
                "split": split_name,
                "name": conv["name"],
                "num_samples": len(payload["samples"]),
                "max_dist": args.max_dist,
                "test_start": args.test_start,
                "test_end": args.test_end,
                "embedding_hash": embedding_hash,
                "schema_version": SCHEMA_VERSION,
                "created": int(time.time()),
            }
        )
        if log_file is not None:
            print(
                "precomputed {} {} samples -> {}".format(conv["name"], len(payload["samples"]), file_path),
                file=log_file,
            )
    clear_feature_cache()
    return manifest_records


def build_manifest_records(args, required_splits: Optional[Sequence[str]] = None, log_file=None) -> List[dict]:
    token_to_id = None
    embedding_hash = compute_embedding_hash(args.word_vectors)
    if args.word_vectors:
        token_to_id, _, _, embedding_hash = load_embeddings(args.word_vectors)

    manifest_records = []
    split_order = ["train", "dev", "test"]
    wanted = set(required_splits or split_order)
    for split_name in split_order:
        if split_name in wanted:
            manifest_records.extend(precompute_split(split_name, args, token_to_id, embedding_hash, log_file=log_file))
    return manifest_records


def build_required_splits(args) -> List[str]:
    required = []
    if args.train:
        required.append("train")
    if args.dev:
        required.append("dev")
    if args.test:
        required.append("test")
    return required


def _record_matches(record, args, split_name: str, embedding_hash) -> bool:
    return (
        record.get("split") == split_name
        and record.get("schema_version") == SCHEMA_VERSION
        and int(record.get("max_dist")) == int(args.max_dist)
        and int(record.get("test_start")) == int(args.test_start)
        and int(record.get("test_end")) == int(args.test_end)
        and record.get("embedding_hash") == embedding_hash
    )


def _has_compatible_manifest(args, required_splits: Sequence[str]) -> bool:
    manifest_path = os.path.join(args.precomputed_dir, "manifest.jsonl")
    if not os.path.exists(manifest_path):
        return False

    embedding_hash = compute_embedding_hash(args.word_vectors)
    matched_splits = set()
    with open(manifest_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            split_name = record.get("split")
            if split_name not in required_splits:
                continue
            if not _record_matches(record, args, split_name, embedding_hash):
                return False
            path = os.path.join(args.precomputed_dir, record.get("path", ""))
            if not os.path.exists(path):
                return False
            matched_splits.add(split_name)
    return matched_splits.issuperset(required_splits)


def prepare_precomputed_artifacts(args, required_splits: Optional[Sequence[str]] = None, log_file=None) -> str:
    required = list(required_splits or build_required_splits(args))
    if not required:
        return args.precomputed_dir

    if _has_compatible_manifest(args, required):
        if log_file is not None:
            print("using compatible precomputed artifacts from {}".format(args.precomputed_dir), file=log_file)
            log_file.flush()
        return args.precomputed_dir

    manifest_records = build_manifest_records(args, required_splits=required, log_file=log_file)
    if not manifest_records:
        missing = ", ".join(required)
        raise ValueError("No data available to precompute required split(s): {}".format(missing))

    manifest_path = os.path.join(args.precomputed_dir, "manifest.jsonl")
    write_manifest(manifest_path, manifest_records)
    if log_file is not None:
        print("wrote manifest {}".format(manifest_path), file=log_file)
        log_file.flush()
    return args.precomputed_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    log_path = get_log_path(args.prefix, ".precompute.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        header(sys.argv if argv is None else [sys.argv[0]] + list(argv), [log_file, sys.stdout])
        prepare_precomputed_artifacts(args, log_file=log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
