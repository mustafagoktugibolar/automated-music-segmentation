# Label Classifier Experiments

This project evaluates semantic section labels with grouped splits. Always group by `raw_track_id` when available, otherwise `song_id`, so annotations or segments from the same track cannot appear in both train and test.

## Why Ablations Matter

The full feature set has 87 columns: acoustic descriptors, position/context, repetition, and local contrast. More features are not automatically better. Some features can improve labels such as Intro and Outro, while others can overfit a particular dataset or song structure. Use ablations to compare generalization, not only train score.

Available feature sets:

- `full`: all 87 features.
- `acoustic`: first 60 acoustic features.
- `acoustic_context`: acoustic plus 11 context features.
- `no_context`: remove context/position features.
- `no_repetition`: remove repetition features.

## Recommended Runs

```bash
python scripts/label_training/train_label_classifier.py   --merge-mode other   --feature-set acoustic_context   --regularization-preset balanced   --experiment-name other_acoustic_context_balanced   --extra-parquet data/label_training/harmonix_segments.parquet
```

```bash
python scripts/label_training/train_label_classifier.py   --merge-mode other   --feature-set no_repetition   --regularization-preset balanced   --experiment-name other_no_repetition_balanced   --extra-parquet data/label_training/harmonix_segments.parquet
```

```bash
python scripts/label_training/eval_label_classifier.py   --mode clean   --merge-mode other   --feature-set acoustic_context   --val-size 0.20   --test-size 0.20   --seed 42   --model-path models/experiments/other_acoustic_context_balanced.joblib   --extra-parquet data/label_training/harmonix_segments.parquet
```

## Experiment Runner

Run the default ablation matrix:

```bash
python scripts/label_training/run_label_experiments.py   --extra-parquet data/label_training/harmonix_segments.parquet
```

Preview commands without running training:

```bash
python scripts/label_training/run_label_experiments.py --dry-run
```

Build or refresh the comparison CSV only:

```bash
python scripts/label_training/run_label_experiments.py --summary-only
```

The summary is written to:

```text
models/evaluation/experiments/summary.csv
```

It sorts experiments by test Macro-F1 and then by smaller train-test gap.
