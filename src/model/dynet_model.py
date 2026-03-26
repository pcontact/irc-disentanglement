#!/usr/bin/env python3

import numpy as np

import dynet as dy

from data.common import FEATURES, load_embeddings


def apply_nonlin(expr, nonlin: str):
    if nonlin == "linear":
        return expr
    if nonlin == "tanh":
        return dy.tanh(expr)
    if nonlin == "cube":
        return dy.cube(expr)
    if nonlin == "logistic":
        return dy.logistic(expr)
    if nonlin == "relu":
        return dy.rectify(expr)
    if nonlin == "elu":
        return dy.elu(expr)
    if nonlin == "selu":
        return dy.selu(expr)
    if nonlin == "softsign":
        dims, _ = expr.dim()
        if len(dims) <= 1:
            return dy.softsign(expr)
        # Older DyNet Python bindings reject matrix-valued softsign inputs.
        flat = dy.reshape(expr, (int(np.prod(dims)),))
        return dy.reshape(dy.softsign(flat), dims)
    if nonlin == "swish":
        return dy.cmult(expr, dy.logistic(expr))
    raise ValueError("Unknown non-linearity: {}".format(nonlin))


class DyNetModel:
    def __init__(self, args):
        self.args = args
        self.model = dy.ParameterCollection()
        self.hidden_dim = args.hidden
        self.use_words = bool(args.word_vectors)
        self.word_dim = 0
        self.id_to_token = []
        self.token_to_id = {}

        input_size = FEATURES
        if self.use_words:
            self.token_to_id, self.id_to_token, weights, _ = load_embeddings(args.word_vectors)
            self.word_dim = int(weights.shape[1])
            self.p_embedding = self.model.add_lookup_parameters((weights.shape[0], self.word_dim))
            self.p_embedding.init_from_array(weights)
            input_size += 4 * self.word_dim

        self.hidden = []
        self.bias = []
        self.hidden.append(self.model.add_parameters((args.hidden, input_size)))
        self.bias.append(self.model.add_parameters((args.hidden,)))
        for _ in range(args.layers - 1):
            self.hidden.append(self.model.add_parameters((args.hidden, args.hidden)))
            self.bias.append(self.model.add_parameters((args.hidden,)))
        self.final_sum = self.model.add_parameters((args.hidden, 1))

    def _zero_vector(self):
        return dy.inputTensor(np.zeros((self.word_dim,), dtype=np.float32))

    def _pool_token_ids(self, token_ids):
        if not token_ids:
            zero = self._zero_vector()
            return zero, zero
        vecs = [dy.lookup(self.p_embedding, token_id) for token_id in token_ids]
        return dy.emax(vecs), dy.average(vecs)

    def _pool_candidates(self, candidate_token_ids):
        max_cols = []
        mean_cols = []
        for token_ids in candidate_token_ids:
            pooled_max, pooled_mean = self._pool_token_ids(token_ids)
            max_cols.append(pooled_max)
            mean_cols.append(pooled_mean)
        return dy.concatenate_cols(max_cols), dy.concatenate_cols(mean_cols)

    def score_instance(self, query_token_ids, candidate_token_ids, feature_matrix, gold_indices=None, train=False):
        num_candidates = int(feature_matrix.shape[0])
        if num_candidates == 1:
            logits = dy.inputTensor(np.asarray([0.0], dtype=np.float32))
            return None, logits

        inputs = dy.inputTensor(np.asarray(feature_matrix.T, dtype=np.float32))
        if self.use_words:
            qvec_max, qvec_mean = self._pool_token_ids(query_token_ids)
            qvec_max_cols = dy.concatenate_cols([qvec_max for _ in range(num_candidates)])
            qvec_mean_cols = dy.concatenate_cols([qvec_mean for _ in range(num_candidates)])
            cand_max, cand_mean = self._pool_candidates(candidate_token_ids)
            inputs = dy.concatenate([inputs, qvec_max_cols, qvec_mean_cols, cand_max, cand_mean])

        if train and self.args.drop > 0:
            inputs = dy.dropout(inputs, self.args.drop)

        hidden = inputs
        for hidden_param, bias_param in zip(self.hidden, self.bias):
            weight = dy.parameter(hidden_param)
            bias = dy.parameter(bias_param)
            bias_cols = dy.concatenate_cols([bias for _ in range(num_candidates)])
            hidden = weight * hidden + bias_cols
            hidden = apply_nonlin(hidden, self.args.nonlin)

        logits = dy.sum_dim(hidden, [0])
        loss = None
        if gold_indices is not None:
            nll = -dy.log_softmax(logits)
            dense_gold = np.zeros((num_candidates,), dtype=np.float32)
            if gold_indices:
                value = 1.0 / float(len(gold_indices))
                for index in gold_indices:
                    dense_gold[index] = value
            answer = dy.inputTensor(dense_gold)
            loss = dy.transpose(answer) * nll
        return loss, logits
