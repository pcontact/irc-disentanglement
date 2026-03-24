# System

This folder contains code for reproducing our disentanglement experiments.

## Requirements

The DyNet path now uses offline preprocessing plus NumPy-backed feature artifacts.
At minimum, install:

```
pip3 install numpy dynet
```

## Running

To see all options, run:

```
python3 disentangle.py --help
```

### Train

The DyNet workflow is now:

1. Precompute fixed-pair `.pkl` artifacts
2. Train from those artifacts with batched updates
3. Predict from the saved `.dy.model`

`disentangle.py` will automatically create compatible precomputed artifacts in `data/precomputed_dynet/`
when they are missing, but it is usually nicer to run preprocessing explicitly first.

### Precompute

```
python3 data/preprocess.py \
  example-precompute \
  --train ../data/train/*annotation.txt \
  --dev ../data/dev/*annotation.txt \
  --test ../data/test/*annotation.txt \
  --word-vectors ../data/glove-ubuntu.txt \
  --max-dist 101 \
  --precomputed-dir ../data/precomputed_dynet
```

This writes per-conversation `.pkl` files under `../data/precomputed_dynet/{train,dev,test}/`
plus a `manifest.jsonl` in `../data/precomputed_dynet/`.
Artifacts are tied to `--max-dist`, the test window, and the word-vector file hash.
If any of those change, regenerate the precomputed data.

### Train

To train, provide `--train` and `--dev` file lists. The example below keeps the original ACL
model shape, but uses the new precompute + batched DyNet training path.

The example command below will train a model with the same parameters as used in the ACL paper.
The model is a feedforward neural network with 2 layers, 512 dimensional hidden vectors, and softsign non-linearities.

```
python3 disentangle.py \
  example-train \
  --train ../data/train/*annotation.txt \
  --dev ../data/dev/*annotation.txt \
  --precomputed-dir ../data/precomputed_dynet \
  --hidden 512 \
  --layers 2 \
  --nonlin softsign \
  --word-vectors ../data/glove-ubuntu.txt \
  --epochs 20 \
  --batch-size 64 \
  --dynet-mem 4096 \
  --drop 0 \
  --learning-rate 0.018804 \
  --learning-decay-rate 0.103 \
  --seed 10 \
  --clip 3.740 \
  --weight-decay 1e-07 \
  --opt sgd \
  > example-train.out 2>example-train.err
```

Notes:

- DyNet autobatching is now enabled by default; use `--no-dynet-autobatch` to turn it off.
- GPU is used by default when DyNet supports it; use `--dynet-cpu` to force CPU mode.
- Mini dev evaluation runs during training, and full dev evaluation runs at the end of each epoch.
- `--speed-profile balanced` and `--speed-profile fast` provide smaller `max-dist` / model-size presets without changing explicit flags.
- If you use `--opt adam` without setting `--learning-rate`, the code defaults to `0.001`.

### Infer

This command will run the model trained above on the development set:

```
python3 disentangle.py \
  example-run.1 \
  --model example-train.dy.model \
  --precomputed-dir ../data/precomputed_dynet \
  --test ../data/dev/*annotation* \
  --test-start 1000 \
  --test-end 2000 \
  --hidden 512 \
  --layers 2 \
  --nonlin softsign \
  --word-vectors ../data/glove-ubuntu.txt \
  --batch-size 128 \
  > example-run.1.out 2>example-run.1.err
```

Note - the arguments defining the network (`hidden`, `layers`, `nonlin`, and `word-vectors`) must match training.
Prediction output is unchanged:

```
NAME.annotation.txt:QUERY_INDEX LINK_INDEX -
```

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
