#!/usr/bin/env python3

import argparse
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from torch_common import header
from torch_dataset import PrecomputedDataset, collate_batch
from torch_model import DisentanglementModel


def build_parser():
    parser = argparse.ArgumentParser(description="IRC Conversation Disentangler (PyTorch).")

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

    # Training arguments
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
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size.")
    parser.add_argument("--num-workers", default=0, type=int, help="DataLoader workers.")
    parser.add_argument("--pin-memory", action="store_true", help="Pin CPU memory for faster transfer.")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only).")
    parser.add_argument("--grad-accum", default=1, type=int, help="Gradient accumulation steps.")

    return parser


def masked_log_softmax(scores, cand_mask):
    masked = scores.masked_fill(cand_mask == 0, -1e9)
    return torch.log_softmax(masked, dim=-1)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    match = 0
    total_gold = 0
    query_correct = 0
    query_total = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            query_ids = batch["query_ids"].to(device)
            cand_ids = batch["cand_ids"].to(device)
            query_lens = batch["query_lens"].to(device)
            cand_lens = batch["cand_lens"].to(device)
            cand_mask = batch["cand_mask"].to(device)
            gold_mask = batch["gold_mask"].to(device)
            orig_gold_count = batch["orig_gold_count"]

            scores = model(features, query_ids, cand_ids, query_lens, cand_lens)
            log_probs = masked_log_softmax(scores, cand_mask)

            gold_counts = gold_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            target = gold_mask / gold_counts
            loss_per = -(target * log_probs).sum(dim=-1)

            total_loss += loss_per.sum().item()
            total_steps += loss_per.numel()

            pred = scores.masked_fill(cand_mask == 0, -1e9).argmax(dim=-1)
            matched = gold_mask.gather(1, pred.unsqueeze(1)).squeeze(1)
            match += matched.sum().item()
            total_gold += orig_gold_count.sum().item()
            query_correct += matched.sum().item()
            query_total += matched.numel()

    dacc = match / total_gold if total_gold > 0 else 0.0
    qacc = query_correct / query_total if query_total > 0 else 0.0
    avg_loss = total_loss / max(1, total_steps)
    return avg_loss, dacc, qacc, match, total_gold


def main():
    parser = build_parser()
    args = parser.parse_args()

    log_file = open(args.prefix + ".log", "w")
    header(sys.argv, [log_file, sys.stdout])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = PrecomputedDataset(
        args.precomputed_dir, "train", random_sample=args.random_sample, seed=args.seed
    )
    dev_dataset = PrecomputedDataset(args.precomputed_dir, "dev")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_batch,
    )
    dev_loader = DataLoader(
        dev_dataset,
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

    if args.opt == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    prev_best = None
    step = 0
    seen = 0
    next_report = args.report_freq

    for epoch in range(args.epochs):
        model.train()
        optimizer.param_groups[0]["lr"] = args.learning_rate / (
            1.0 + args.learning_decay_rate * epoch
        )

        loss_sum = 0.0
        loss_steps = 0
        match = 0
        total = 0
        query_correct = 0
        query_total = 0

        optimizer.zero_grad(set_to_none=True)

        for batch in train_loader:
            step += 1
            batch_size = batch["features"].shape[0]
            seen += batch_size

            features = batch["features"].to(device)
            query_ids = batch["query_ids"].to(device)
            cand_ids = batch["cand_ids"].to(device)
            query_lens = batch["query_lens"].to(device)
            cand_lens = batch["cand_lens"].to(device)
            cand_mask = batch["cand_mask"].to(device)
            gold_mask = batch["gold_mask"].to(device)
            orig_gold_count = batch["orig_gold_count"]

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                scores = model(features, query_ids, cand_ids, query_lens, cand_lens)
                log_probs = masked_log_softmax(scores, cand_mask)
                gold_counts = gold_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                target = gold_mask / gold_counts
                loss_per = -(target * log_probs).sum(dim=-1)

                cand_counts = cand_mask.sum(dim=-1)
                train_mask = ((gold_mask.sum(dim=-1) > 0) & (cand_counts > 1)).float()
                denom = train_mask.sum().clamp(min=1.0)
                loss_opt = (loss_per * train_mask).sum() / denom
                loss_opt = loss_opt / max(1, args.grad_accum)

            scaler.scale(loss_opt).backward()

            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_sum += loss_per.sum().item()
            loss_steps += loss_per.numel()

            pred = scores.masked_fill(cand_mask == 0, -1e9).argmax(dim=-1)
            matched = gold_mask.gather(1, pred.unsqueeze(1)).squeeze(1)
            match += matched.sum().item()
            total += orig_gold_count.sum().item()
            query_correct += matched.sum().item()
            query_total += matched.numel()

            if seen >= next_report:
                avg_loss = loss_sum / max(1, loss_steps)
                tacc = match / total if total > 0 else 0.0
                tqacc = query_correct / query_total if query_total > 0 else 0.0

                dev_loss, dacc, dqacc, dev_match, dev_total = evaluate(
                    model, dev_loader, device
                )

                print(
                    "{} tl {:.3f} ta {:.3f} da {:.3f} tq {:.3f} dq {:.3f} from {} {}".format(
                        epoch, avg_loss, tacc, dacc, tqacc, dqacc, dev_match, dev_total
                    ),
                    file=log_file,
                )
                log_file.flush()

                if prev_best is None or prev_best[0] < dacc:
                    prev_best = (dacc, epoch)
                    torch.save(model.state_dict(), args.prefix + ".pt")

                next_report += args.report_freq

        if step % args.grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if prev_best is not None and epoch - prev_best[1] > 5:
            break

    log_file.close()


if __name__ == "__main__":
    main()
