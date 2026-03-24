#!/usr/bin/env python3

import numpy as np

import dynet as dy

from data.dataset import bucket_indices


def _predict_index(logits_expr):
    return int(np.argmax(logits_expr.npvalue()))


def _evaluate_batch(model, batch_samples, train=False):
    losses = []
    logits = []
    for sample in batch_samples:
        loss_expr, logits_expr = model.score_instance(
            sample["query_token_ids"],
            sample["candidate_token_ids"],
            sample["features"],
            gold_indices=sample.get("gold_indices"),
            train=train,
        )
        if loss_expr is not None:
            losses.append(loss_expr)
        logits.append(logits_expr)

    batch_loss = 0.0
    if losses:
        batch_loss = (dy.esum(losses) * (1.0 / float(len(losses)))).scalar_value()

    results = []
    for sample, logits_expr in zip(batch_samples, logits):
        predicted_index = _predict_index(logits_expr)
        matched = predicted_index in sample.get("gold_indices", [])
        results.append((sample, predicted_index, matched))
    return batch_loss, len(losses), results


def evaluate_dataset(model, dataset, batch_size, indices=None):
    total_loss = 0.0
    loss_batches = 0
    match = 0
    total_gold = 0
    query_correct = 0
    query_total = 0

    grouped = bucket_indices(dataset, indices=indices)
    for key in sorted(grouped.keys()):
        bucket = grouped[key]
        for start in range(0, len(bucket), batch_size):
            batch_indices = bucket[start : start + batch_size]
            batch_samples = [dataset[idx] for idx in batch_indices]
            dy.renew_cg()
            batch_loss, batch_loss_count, batch_results = _evaluate_batch(model, batch_samples, train=False)
            if batch_loss_count > 0:
                total_loss += batch_loss * batch_loss_count
                loss_batches += batch_loss_count
            for sample, _predicted_index, matched in batch_results:
                if matched:
                    match += 1
                    query_correct += 1
                total_gold += int(sample.get("orig_gold_count", 0))
                query_total += 1

    avg_loss = total_loss / max(1, loss_batches)
    dacc = match / total_gold if total_gold > 0 else 0.0
    qacc = query_correct / query_total if query_total > 0 else 0.0
    return avg_loss, dacc, qacc, match, total_gold


def predict_dataset(model, dataset, batch_size):
    predictions = {}
    grouped = bucket_indices(dataset)
    for key in sorted(grouped.keys()):
        bucket = grouped[key]
        for start in range(0, len(bucket), batch_size):
            batch_indices = bucket[start : start + batch_size]
            batch_samples = [dataset[idx] for idx in batch_indices]
            dy.renew_cg()
            _batch_loss, _batch_loss_count, batch_results = _evaluate_batch(model, batch_samples, train=False)
            for dataset_index, (sample, predicted_index, _matched) in zip(batch_indices, batch_results):
                candidate_indices = sample["candidate_indices"]
                predictions[dataset_index] = {
                    "name": sample["name"],
                    "query_index": sample["query_index"],
                    "predicted_index": predicted_index,
                    "predicted_link": candidate_indices[predicted_index],
                }
    return [predictions[idx] for idx in range(len(dataset))]
