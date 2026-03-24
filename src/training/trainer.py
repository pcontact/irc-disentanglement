#!/usr/bin/env python3

import random

import numpy as np

import dynet as dy

from data.dataset import iter_batches
from training.evaluate import evaluate_dataset


def build_optimizer(model, args):
    if args.opt == "sgd":
        optimizer = dy.SimpleSGDTrainer(model.model, learning_rate=args.learning_rate)
    elif args.opt == "mom":
        optimizer = dy.MomentumSGDTrainer(model.model, learning_rate=args.learning_rate, mom=args.momentum)
    elif args.opt == "adam":
        optimizer = dy.AdamTrainer(model.model, alpha=args.learning_rate)
    else:
        raise ValueError("Unknown optimiser: {}".format(args.opt))
    optimizer.set_clip_threshold(args.clip)
    return optimizer


def sample_dev_indices(dev_dataset, size, seed):
    if dev_dataset is None or len(dev_dataset) == 0 or size <= 0:
        return []
    size = min(size, len(dev_dataset))
    rng = random.Random(seed)
    return rng.sample(list(range(len(dev_dataset))), size)


def train_model(model, optimizer, train_dataset, dev_dataset, args, log_file):
    prev_best = None
    step = 0
    mini_eval_size = min(args.mini_eval_size, len(dev_dataset)) if dev_dataset is not None else 0
    mini_eval_indices = sample_dev_indices(dev_dataset, mini_eval_size, args.seed) if mini_eval_size > 0 else []

    for epoch in range(args.epochs):
        optimizer.learning_rate = args.learning_rate / (1.0 + args.learning_decay_rate * epoch)

        loss_sum = 0.0
        loss_steps = 0
        match = 0
        total = 0

        for batch in iter_batches(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
        ):
            dy.renew_cg()
            losses = []
            logits = []
            for sample in batch:
                loss_expr, logits_expr = model.score_instance(
                    sample["query_token_ids"],
                    sample["candidate_token_ids"],
                    sample["features"],
                    gold_indices=sample.get("gold_indices"),
                    train=True,
                )
                if loss_expr is not None:
                    losses.append(loss_expr)
                logits.append(logits_expr)

            if not losses:
                continue

            step += 1
            batch_loss_expr = dy.esum(losses) * (1.0 / float(len(losses)))
            batch_loss_value = batch_loss_expr.scalar_value()
            predictions = [int(np.argmax(logits_expr.npvalue())) for logits_expr in logits]
            batch_loss_expr.backward()
            optimizer.update()

            loss_sum += batch_loss_value * len(batch)
            loss_steps += len(batch)
            for sample, prediction in zip(batch, predictions):
                if prediction in sample.get("gold_indices", []):
                    match += 1
                total += int(sample.get("orig_gold_count", 0))

            if dev_dataset is not None and args.mini_eval_steps > 0 and step % args.mini_eval_steps == 0 and mini_eval_indices:
                mini_loss, mini_dacc, mini_qacc, mini_match, mini_total = evaluate_dataset(
                    model,
                    dev_dataset,
                    args.batch_size,
                    indices=mini_eval_indices,
                )
                avg_loss = loss_sum / max(1, loss_steps)
                tacc = match / total if total > 0 else 0.0
                print(
                    (
                        "{} step {} tl {:.3f} ta {:.3f} mla {:.3f} mlq {:.3f} from {} {}"
                    ).format(
                        epoch,
                        step,
                        avg_loss,
                        tacc,
                        mini_dacc,
                        mini_qacc,
                        mini_match,
                        mini_total,
                    ),
                    file=log_file,
                )
                log_file.flush()

        if dev_dataset is not None and len(dev_dataset) > 0:
            dev_loss, dacc, dqacc, dev_match, dev_total = evaluate_dataset(model, dev_dataset, args.batch_size)
            avg_loss = loss_sum / max(1, loss_steps)
            tacc = match / total if total > 0 else 0.0
            print(
                "{} tl {:.3f} ta {:.3f} da {:.3f} dq {:.3f} from {} {}".format(
                    epoch,
                    avg_loss,
                    tacc,
                    dacc,
                    dqacc,
                    dev_match,
                    dev_total,
                ),
                file=log_file,
            )
            log_file.flush()

            if prev_best is None or prev_best[0] < dacc:
                prev_best = (dacc, epoch)
                model.model.save(args.prefix + ".dy.model")
        else:
            if prev_best is None:
                prev_best = (0.0, epoch)
                model.model.save(args.prefix + ".dy.model")

        if prev_best is not None and epoch - prev_best[1] > 5:
            break

    return prev_best
