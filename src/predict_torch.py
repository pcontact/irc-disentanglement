#!/usr/bin/env python3

import argparse
import sys

import torch
from torch.utils.data import DataLoader

from torch_common import header
from torch_dataset import PrecomputedDataset, collate_batch
from torch_model import DisentanglementModel


def build_parser():
    parser = argparse.ArgumentParser(description="IRC Conversation Disentangler (PyTorch Prediction).")

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

    # Torch arguments
    parser.add_argument(
        "--precomputed-dir",
        default="data/precomputed",
        help="Directory containing precomputed .pt files and manifest.",
    )
    parser.add_argument("--batch-size", default=64, type=int, help="Batch size.")
    parser.add_argument("--num-workers", default=0, type=int, help="DataLoader workers.")
    parser.add_argument("--pin-memory", action="store_true", help="Pin CPU memory for faster transfer.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    log_file = open(args.prefix + ".log", "w")
    header(sys.argv, [log_file, sys.stdout])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = PrecomputedDataset(args.precomputed_dir, "test")
    if len(test_dataset) == 0:
        test_dataset = PrecomputedDataset(args.precomputed_dir, "dev")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_batch,
    )

    model = DisentanglementModel(
        word_vectors=args.word_vectors,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        nonlin=args.nonlin,
        dropout=args.drop,
    ).to(device)

    model_path = args.model if args.model else args.prefix + ".pt"
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            query_ids = batch["query_ids"].to(device)
            cand_ids = batch["cand_ids"].to(device)
            query_lens = batch["query_lens"].to(device)
            cand_lens = batch["cand_lens"].to(device)
            cand_mask = batch["cand_mask"].to(device)

            scores = model(features, query_ids, cand_ids, query_lens, cand_lens)
            pred = scores.masked_fill(cand_mask == 0, -1e9).argmax(dim=-1)

            for name, query_idx, pred_idx in zip(
                batch["names"], batch["query_indices"], pred.cpu().tolist()
            ):
                print("{}:{} {} -".format(name, query_idx, query_idx - pred_idx))

    log_file.close()


if __name__ == "__main__":
    main()

