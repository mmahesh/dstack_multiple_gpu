# Building a pipeline for training on multiple GPUs with dstack

This repo contains the code to run a training pipeline on multiple GPUs.

The `train.py` contains all the machine learning components.

In order to automate the workflow, I use `dstack`.

In order to run the workflow, all you need to do is 

```bash
dstack run train-mnist-multi-gpu
```