# On the Use of Bagging Ensembles for Configurable Software Performance Prediction

This repository contains the source code, the data used, and the raw
results for each experiment from the corresponding paper.

## Documents

### /data

This folder contains the data used for all experiments. There is one CSV
file for each subject system (eight in total). Each column represents a
configuration option, and the last column is the performance value to be
predicted.

### /results_original
This folder contains the results for every run of each experiment across all evaluation scenarios. Future executions will save results in a separate `/results` directory.

The format of the results is as follows:

One folder per dataset, and within each dataset folder, one folder per
sample size (five per dataset).

In each sample size subfolder, there are 4 files named
`{Model-Name}_single_metrics.json`. These files represent the results
of the single variants of the corresponding model type (4 model
types).

Each file has the following format:

A JSON file containing one list with one dict entry. The dict
represents the best variant of the corresponding model with its
corresponding hyperparameters (visible in the `"Params"` entry). The dict also contains the mean
values across 30 runs for each evaluation metric (`Mean_MAE`,
`Mean_MRE`, `Mean_MBRE`, `Mean_MIBRE`). Additionally, `"Metrics"` and
`"Runs"` contain the evaluation metric values of each run and the raw
predictions of each run for each element in the corresponding test sets.

Our experiments include 4 different combinations of ensemble variants
for each learning technique: 2 combination rules (Mean and
Median) and 2 different numbers of single learners (5 and 10).
Similar to the single variants, we have:

-   `{Model-Name}_bagging_N{k}_Fall_{combination-rule}.json`

with `k` representing the number of single learners and `combination-rule` representing the combination rule.

These files contain a single dict with the mean values across 30 runs
for each evaluation metric (`Mean_MAE`, `Mean_MRE`, `Mean_MBRE`,
`Mean_MIBRE`). `"Metrics"` contains the evaluation metric
values of each run.

Other files with other ensemble variants can be ignored as they are not in scope of the corresponding paper.

### /DEEPPERF and /HINNPERF

These directories contain the implementations of both state-of-the-art
approaches, using their publicly available resources:
https://github.com/DeepPerf/DeepPerf and https://drive.google.com/drive/folders/1qxYzd5Om0HE1rK0syYQsTPhTQEBjghLh

### ML Techniques

Each ML technique has its own class based on the `Base` class in
`base.py`. It contains the general logic described in the paper for
constructing the single variants.

### main.py

This file contains the general logic described in the paper for
evaluating the single variants and the ensemble variants.

## Prerequisites and Installation

Use `requirements.txt` and follow any runtime messages. The experiments were executed with Python 3.9.x.

## How to Run

Comment out the datasets and models you want to exclude. Otherwise, it
will run all experiments by default, as described in the paper.

To reproduce the experiments from the paper:

    python main.py --model baseline
    python main.py

Note: Results from the next execution will be saved in `/results` by default.
