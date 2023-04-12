# Director-pytorch

PyTorch implementation of the _Director_ agent, based on the paper [Deep Hierarchical Planning from Pixels](https://arxiv.org/abs/2206.04114).

Based on the [DreamerV2 PyTorch implementation](https://github.com/RajGhugare19/dreamerv2) by RajGhugare19.

Some example results can be seen at W&B [here](https://wandb.ai/jormeli/Director), e.g. [Sokoban](https://wandb.ai/jormeli/Director/runs/dwjzvtt4).

## Usage

Install Poetry:

`$ pip install poetry`

Install dependencies:

`$ poetry install`

Run an experiment:

`$ python pomdb.py --config-name breakout`
