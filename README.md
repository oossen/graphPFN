## About

This repository contains the code for my Master Project, completed at the University of Freiburg in the winter semester 2025/26, and supervised by [Jake Robertson](https://jr2021.github.io/) and [Arik Reuter](https://arikreuter.github.io/).

See [my presentation of the project](presentation.pdf) for an overview.

## Reproducibility

Models were trained by running `pretrain.py`. By default, models are trained with cross-entropy loss/soft labels. To train with negative log-likelihood loss, add the flag `--nll`.
The type of graph information can be specified with the `--prob_adj_mode` flag, possible options being `"binary"`, `"beta"`, and `"uncertain"`.
To switch the model architecture, you need to switch out the configuration file in `pretrain.py`.
For example, to train a model with negative log-likelihood loss and "uncertain" graph information, run:
```
python pretrain.py --nll --prob_adj_mode "uncertain"
```

All evaluations used in the presentation are hardcoded in `visualization/scripts.py`. For example, the bar plot comparing model performance for different graph information types was created by running:
```
python -m visualization.scripts --swapped_input_mode
```
