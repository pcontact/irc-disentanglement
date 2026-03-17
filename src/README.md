# System

This folder contains code for reproducing our disentanglement experiments.

## Requirements

The only dependency is the [DyNet library](http://dynet.readthedocs.io), which can usually be installed with:

```
pip3 install dynet
```

## Running

To see all options, run:

```
python3 disentangle.py --help
```

### Train

To train, provide the `--train` argument followed by a series of filenames.

The example command below will train a model with the same parameters as used in the ACL paper.
The model is a feedforward neural network with 2 layers, 512 dimensional hidden vectors, and softsign non-linearities.

```
python3 disentangle.py \
  example-train \
  --train ../data/train/*annotation.txt \
  --dev ../data/dev/*annotation.txt \
  --hidden 512 \
  --layers 2 \
  --nonlin softsign \
  --word-vectors ../data/glove-ubuntu.txt \
  --epochs 20 \
  --dynet-autobatch \
  --drop 0 \
  --learning-rate 0.018804 \
  --learning-decay-rate 0.103 \
  --seed 10 \
  --clip 3.740 \
  --weight-decay 1e-07 \
  --opt sgd \
  > example-train.out 2>example-train.err
```

### Infer

This command will run the model trained above on the development set:

```
python3 disentangle.py \
  example-run.1 \
  --model example-train.dy.model \
  --test ../data/dev/*annotation* \
  --test-start 1000 \
  --test-end 2000 \
  --hidden 512 \
  --layers 2 \
  --nonlin softsign \
  --word-vectors ../data/glove-ubuntu.txt \
  > example-run.1.out 2>example-run.1.err
```

Note - the arguments defining the network (hidden, layers, nonlin), must match those given in training.

### Evaluate

This command will run the output produced by the command above through the evaluation script:

```
python3 ../tools/evaluation/graph-eval.py --gold ../data/dev/*annotation* --auto example-run.1.out
```

The output should be something like:

```
g/a/m: 2607 2500 1855
p/r/f: 74.2 71.2 72.6
```

The first row is a count of the gold links, auto links, and matching links.
The second line is the precision, recall, and F-score.

Note - the values in the paper are an average over 10 runs, so they will differ slightly from what you get here.

## PyTorch (GPU-Accelerated)

This repo also includes a PyTorch refactor that matches the DyNet model but uses precomputed features and batched GPU training.
The workflow is:

1. Precompute features and token IDs into `.pt` files
2. Train with `train_torch.py`
3. Predict with `predict_torch.py`

### Expected Input File Format

For each conversation `NAME`, the scripts expect the same file triplet as the DyNet code:

- `NAME.ascii.txt`: whitespace-tokenized messages, one message per line, matching DyNet’s parser.
- `NAME.tok.txt`: tokenized messages, one per line. If a line ends with `</s>`, it will be removed; if a line does not start with `<s>`, it will be inserted.
- `NAME.annotation.txt`: gold links for training/dev. Each line lists message indices for a thread; the max index is the query, the min index is the linked message.

You can pass any of these suffixes to `--train/--dev/--test`; the scripts strip the suffix to locate the other files.

### Precompute

```
python3 precompute.py \
  example-precompute \
  --train ../data/train/*annotation.txt \
  --dev ../data/dev/*annotation.txt \
  --test ../data/test/*annotation.txt \
  --word-vectors ../data/glove-ubuntu.txt \
  --max-dist 101 \
  --precomputed-dir ../data/precomputed
```

This writes per-conversation `.pt` files under `../data/precomputed/{train,dev,test}/` and a `manifest.jsonl` in `../data/precomputed/`.

### Train (PyTorch)

```
python3 train_torch.py \
  example-train.torch \
  --precomputed-dir ../data/precomputed \
  --word-vectors ../data/glove-ubuntu.txt \
  --hidden 512 \
  --layers 2 \
  --nonlin softsign \
  --epochs 20 \
  --learning-rate 0.018804 \
  --learning-decay-rate 0.103 \
  --seed 10 \
  --clip 3.740 \
  --weight-decay 1e-07 \
  --opt sgd \
  --batch-size 64 \
  --pin-memory \
  --num-workers 2 \
  --amp \
  > example-train.torch.out 2>example-train.torch.err
```

This saves a model checkpoint to `example-train.torch.pt` and logs to `example-train.torch.log`.

### Predict (PyTorch)

```
python3 predict_torch.py \
  example-run.torch \
  --model example-train.torch.pt \
  --precomputed-dir ../data/precomputed \
  --word-vectors ../data/glove-ubuntu.txt \
  --hidden 512 \
  --layers 2 \
  --nonlin softsign \
  --batch-size 128 \
  > example-run.torch.out 2>example-run.torch.err
```

The output format matches DyNet exactly:

```
NAME.annotation.txt:QUERY_INDEX LINK_INDEX -
```

where `LINK_INDEX` is the predicted absolute message index (e.g., `query - prediction` in the DyNet code).

### Running on a file

If you want to apply a model to a file, see this script for an example of how to do it: `example-running.sh`.
The script is set up so someone could call it like so (once the necessary placeholders in the script are set):

./disentangle-file.sh < sample.ascii.txt > sample.links.txt

## Ensemble

For the best results, we used a simple ensemble of multiple models.
We trained 10 models as described above, but with different random seeds (1 through to 10).
We combined their output using the `majority_vote.py` script in this directory.

The same script is used for all three ensemble methods, with slightly different input and arguments:

Union
```
./majority_vote.py example-run*graphs --method union > example-run.combined.union
```

Vote
```
./majority_vote.py example-run*graphs --method vote > example-run.combined.vote
```

Intersect
```
./majority_vote.py example-run*clusters --method intersect > example-run.combined.intersect
```

All of these assume the output files have been converted into our graph format.
Assuming you run `disentangle.py` above and save the output of each run as `example-run.1.out`, `example-run.2.out`, `example-run.3.out`, etc, then this command will use one of our tools to convert them to the graph format:
```
for name in example-run*out ; do ../tools/format-conversion/output-from-py-to-graph.py < $name > $name.graphs ; done
```

The intersect method also assumes they have been made into clusters, like this:
```
for name in example-run*out ; do ../tools/format-conversion/graph-to-cluster.py < $name.graphs > $name.clusters ; done
```

Note: An earlier version of the steps above didn't account for a change in the output of the main system. Apologies for the broken output this would have caused.

## C++ Model

As well as the main Python code, we also wrote a model in C++ that was used for DSTC 7 and the results in the 2018 arXiv version of the paper (the Python version was used for DSTC 8 and the 2019 ACL paper).
The python model has additional input features and a different text representation method.
The C++ model has support for a range of additional variations in both inference and modeling, which did not appear to improve performance.
For details on how to build and run the C++ code, see [this page](./old-cpp-version/).

[Go back](./../) to the main webpage.
