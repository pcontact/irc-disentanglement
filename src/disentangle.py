#!/usr/bin/env python3

import argparse
import math
import random
import sys

import numpy as np

from data.common import get_log_path, header
from data.dataset import PrecomputedDataset
from data.preprocess import add_shared_args, build_required_splits, prepare_precomputed_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IRC Conversation Disentangler.")
    add_shared_args(parser)
    parser.add_argument("--dynet-cpu", action="store_true", help="Force CPU execution instead of DyNet GPU mode.")
    parser.add_argument("--dynet-mem", default=4096, type=int, help="DyNet memory allocation in MB.")
    parser.add_argument("--no-dynet-autobatch", action="store_true", help="Disable DyNet autobatching.")
    parser.add_argument("--batch-size", default=64, type=int, help="Number of queries to process per update.")
    parser.add_argument(
        "--mini-eval-steps",
        type=int,
        default=None,
        help="Run mini dev evaluation every N training batches.",
    )
    parser.add_argument(
        "--mini-eval-size",
        type=int,
        default=2048,
        help="Fixed number of dev queries to use for mini evaluation.",
    )
    parser.add_argument(
        "--speed-profile",
        choices=["full", "balanced", "fast"],
        default="full",
        help="Optional runtime preset; explicit flags still win.",
    )
    return parser


def flag_present(argv, flag: str) -> bool:
    return any(arg == flag or arg.startswith(flag + "=") for arg in argv)


def apply_speed_profile(args, argv):
    if args.speed_profile == "full":
        return

    presets = {
        "balanced": {"max_dist": 64, "hidden": 384, "layers": 2},
        "fast": {"max_dist": 32, "hidden": 256, "layers": 1},
    }
    preset = presets[args.speed_profile]
    if not flag_present(argv, "--max-dist"):
        args.max_dist = preset["max_dist"]
    if not flag_present(argv, "--hidden"):
        args.hidden = preset["hidden"]
    if not flag_present(argv, "--layers"):
        args.layers = preset["layers"]


def resolve_learning_rate(args, argv, log_file):
    if args.opt == "adam" and not flag_present(argv, "--learning-rate"):
        args.learning_rate = 0.001
        print("using Adam default learning rate {:.6f}".format(args.learning_rate), file=log_file)
        log_file.flush()


def resolve_autobatch(argv, args) -> bool:
    enabled = True
    for token in argv:
        if token == "--dynet-autobatch":
            enabled = True
        elif token == "--no-dynet-autobatch":
            enabled = False
    if not flag_present(argv, "--dynet-autobatch") and not flag_present(argv, "--no-dynet-autobatch"):
        enabled = True
    if args.dynet_autobatch:
        enabled = True
    if args.no_dynet_autobatch:
        enabled = False
    return enabled


def build_dataset(args, split_name, random_sample=None):
    return PrecomputedDataset(
        args.precomputed_dir,
        split_name,
        max_dist=args.max_dist,
        test_start=args.test_start,
        test_end=args.test_end,
        word_vectors=args.word_vectors,
        random_sample=random_sample,
        seed=args.seed,
    )


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_speed_profile(args, argv)

    log_path = get_log_path(args.prefix, ".log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        header([sys.argv[0]] + argv, [log_file, sys.stdout])

        resolve_learning_rate(args, argv, log_file)
        if args.mini_eval_steps is None:
            args.mini_eval_steps = max(1, int(math.ceil(float(args.report_freq) / float(max(1, args.batch_size)))))

        random.seed(args.seed)
        np.random.seed(args.seed)

        prepare_precomputed_artifacts(args, required_splits=build_required_splits(args), log_file=log_file)

        train_dataset = build_dataset(args, "train", random_sample=args.random_sample) if args.train else None
        dev_dataset = build_dataset(args, "dev") if args.dev else None
        test_split = "test" if args.test else ("dev" if args.dev else None)
        test_dataset = build_dataset(args, test_split) if test_split else None

        import dynet_config

        autobatch = 1 if resolve_autobatch(argv, args) else 0
        if not args.dynet_cpu:
            dynet_config.set_gpu()
        dynet_config.set(
            mem=args.dynet_mem,
            autobatch=autobatch,
            weight_decay=args.weight_decay,
            random_seed=args.seed,
        )
        import dynet as dy  # noqa: F401

        from model.dynet_model import DyNetModel
        from training.evaluate import predict_dataset
        from training.trainer import build_optimizer, train_model

        model = DyNetModel(args)
        optimizer = build_optimizer(model, args)

        if args.model:
            model.model.populate(args.model)

        prev_best = None
        if train_dataset is not None:
            prev_best = train_model(model, optimizer, train_dataset, dev_dataset, args, log_file)

        if prev_best is not None:
            model.model.populate(args.prefix + ".dy.model")
        elif args.model:
            model.model.populate(args.model)

        if test_dataset is not None:
            for prediction in predict_dataset(model, test_dataset, args.batch_size):
                print("{}:{} {} -".format(prediction["name"], prediction["query_index"], prediction["predicted_link"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
