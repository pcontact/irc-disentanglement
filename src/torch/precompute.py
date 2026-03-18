#!/usr/bin/env python3

import argparse
import os
import sys
import time

import torch

from torch_common import (
    FEATURES,
    get_log_path,
    get_features,
    get_ids,
    header,
    load_conversations,
    load_embeddings,
    write_manifest,
)


def build_parser():
    parser = argparse.ArgumentParser(description="IRC Conversation Disentangler (Precompute).")

    # General arguments
    parser.add_argument("prefix", help="Start of names for files produced.")

    # Data arguments
    parser.add_argument("--train", nargs="+", help="Training files, e.g. train/*annotation.txt")
    parser.add_argument("--dev", nargs="+", help="Development files, e.g. dev/*annotation.txt")
    parser.add_argument("--test", nargs="+", help="Test files, e.g. test/*annotation.txt")
    parser.add_argument(
        "--test-start",
        type=int,
        help="The line to start making predictions from in each test file.",
        default=1000,
    )
    parser.add_argument(
        "--test-end",
        type=int,
        help="The line to stop making predictions on in each test files.",
        default=1000000,
    )
    parser.add_argument("--model", help="A file containing a trained model")
    parser.add_argument(
        "--random-sample", help="Train on only a random sample of the data with this many examples."
    )

    # Model arguments
    parser.add_argument("--hidden", default=512, type=int, help="Number of dimensions in hidden vectors.")
    parser.add_argument("--word-vectors", help="File containing word embeddings.")
    parser.add_argument("--layers", default=2, type=int, help="Number of hidden layers in the model")
    parser.add_argument(
        "--nonlin",
        choices=["tanh", "cube", "logistic", "relu", "elu", "selu", "softsign", "swish", "linear"],
        default="softsign",
        help="Non-linearity type.",
    )

    # Inference arguments
    parser.add_argument(
        "--max-dist",
        default=101,
        type=int,
        help="Maximum number of messages to consider when forming a link (count includes the current message).",
    )
    parser.add_argument(
        "--dynet-autobatch",
        action="store_true",
        help="Ignored (kept for CLI compatibility).",
    )

    # Training arguments (unused here, kept for compatibility)
    parser.add_argument(
        "--report-freq", default=5000, type=int, help="How frequently to evaluate on the development set."
    )
    parser.add_argument("--epochs", default=20, type=int, help="Maximum number of epochs.")
    parser.add_argument("--opt", choices=["sgd", "mom"], default="sgd", help="Optimisation method.")
    parser.add_argument("--seed", default=10, type=int, help="Random seed.")
    parser.add_argument("--weight-decay", default=1e-7, type=float, help="Apply weight decay.")
    parser.add_argument("--learning-rate", default=0.018804, type=float, help="The initial learning rate.")
    parser.add_argument(
        "--learning-decay-rate", default=0.103, type=float, help="The rate at which the learning rate decays."
    )
    parser.add_argument("--momentum", default=0.1, type=float, help="Hyperparameter for momentum.")
    parser.add_argument("--drop", default=0.0, type=float, help="Dropout, applied to inputs only.")
    parser.add_argument("--clip", default=3.740, type=float, help="Gradient clipping.")

    # Precompute arguments
    parser.add_argument(
        "--precomputed-dir",
        default=os.path.join("data", "precomputed"),
        help="Directory to write precomputed .pt files and manifest.",
    )

    return parser


def safe_name(name):
    safe = name.replace("\\", "_").replace("/", "_").replace(":", "")
    return safe


def precompute_split(split_name, filenames, args, token_to_id):
    if not filenames:
        return []

    is_test = split_name == "test"
    conversations = load_conversations(filenames, is_test, args.test_start, args.test_end)

    split_dir = os.path.join(args.precomputed_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    manifest_records = []
    for conv in conversations:
        name = conv["name"]
        text_ascii = conv["text_ascii"]
        text_tok = conv["text_tok"]
        info = conv["info"]
        target_info = conv["target_info"]
        links = conv["links"]

        samples = []
        for query, gold in links.items():
            cand_indices = list(range(query, max(-1, query - args.max_dist), -1))
            if split_name == "train" and len(cand_indices) == 1:
                continue
            gold_filtered = [v for v in gold if v > query - args.max_dist]
            if split_name == "train" and len(gold_filtered) == 0:
                continue

            query_ids = []
            cand_ids = []
            cand_lens = []
            if token_to_id is not None:
                query_ids = get_ids(text_tok[query], token_to_id)

            features = []
            for i in cand_indices:
                if token_to_id is not None:
                    ids = get_ids(text_tok[i], token_to_id)
                    cand_ids.append(ids)
                    cand_lens.append(len(ids))
                feats = get_features(
                    name, query, i, text_ascii, text_tok, info, target_info, do_cache=False
                )
                features.append(feats)

            gold_indices = [query - v for v in gold_filtered]

            samples.append(
                {
                    "features": torch.tensor(features, dtype=torch.float32),
                    "query_ids": query_ids,
                    "query_len": len(query_ids),
                    "cand_ids": cand_ids,
                    "cand_lens": cand_lens,
                    "gold_indices": gold_indices,
                    "query_index": query,
                    "orig_gold_count": len(gold),
                }
            )

        if len(samples) == 0:
            continue

        file_id = safe_name(name)
        file_path = os.path.join(split_dir, file_id + ".pt")
        torch.save({"name": name, "samples": samples}, file_path)

        manifest_records.append(
            {
                "path": os.path.relpath(file_path, args.precomputed_dir),
                "split": split_name,
                "num_samples": len(samples),
                "name": name,
                "created": int(time.time()),
            }
        )

    return manifest_records


def main():
    parser = build_parser()
    args = parser.parse_args()

    log_file = open(get_log_path(args.prefix, ".precompute.log"), "w")
    header(sys.argv, [log_file, sys.stdout])

    token_to_id = None
    if args.word_vectors:
        token_to_id, _, _ = load_embeddings(args.word_vectors)

    manifest_records = []
    manifest_records.extend(precompute_split("train", args.train, args, token_to_id))
    manifest_records.extend(precompute_split("dev", args.dev, args, token_to_id))
    manifest_records.extend(precompute_split("test", args.test, args, token_to_id))

    manifest_path = os.path.join(args.precomputed_dir, "manifest.jsonl")
    write_manifest(manifest_path, manifest_records)

    log_file.close()


if __name__ == "__main__":
    main()
