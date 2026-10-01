# CCAR-NRS reproducibility package

Data and code for the study *Core-Centric Attribute Reduction for Neighborhood Rough Sets without Discernibility Matrix Construction*.

- `results_v7/`: fold-level results, summaries, sensitivity and scalability experiments, and run environment. See [`results_v7/README.md`](results_v7/README.md).
- `code/`: CCAR-NRS, comparison methods, experiment runners, tests, and figure/table scripts. See [`code/README.md`](code/README.md).

The dataset loaders in `code/experiments/datasets.py` document the public UCI data sources and preprocessing. Results can be inspected without rerunning the experiments. For the original software environment and reproduction commands, see `code/README.md` and `results_v7/RUN_INFO.txt`. The code is distributed under the MIT licence in `code/LICENSE`; data sources retain their original terms.

To rerun experiments, work from `code/` and fetch the datasets first:

```sh
python -m experiments.fetch_uci_data
python -m experiments.fetch_extended_uci
```

The commands in `code/README.md` then write results to the sibling `results_v7/` directory. Keep a copy of the supplied results before rerunning if you want to compare outputs.
