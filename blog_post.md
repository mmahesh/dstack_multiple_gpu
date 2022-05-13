# Building a pipeline for training on multiple GPUs with dstack

# Introduction

The objective of this blog is:
  - to show how to build a deep learning training pipeline that uses multiple GPUs via a AWS server on the cloud;
- to show how to use [dstack](https://dstack.ai), a framework to automate the entire workflow of training of a deep learning models on cloud, for building training pipelines.

The code pertaining to this blog can be found [here](https://github.com/mmahesh/dstack_test/).

The pre-requisites to understand this blog include:

    - familiarity with deep learning,
    - familiarity with python,
    - familiarity with pytorch and pytorch-lightning,
    - and familiarity with wandb (a cloud based tool for experiment tracking).

We use `Python 3` for the programming part. 



The contents of this blog mainly include 

    - creating a simple deep learning model to train on the popular MNIST dataset,
    - creating dstack workflows for training the model using multiple GPUs on a AWS server in the cloud,
    -  running the deep learning models using dstack workflows and monotoring the results via wandb.

# Requirements

Firstly, we need to install `dstack`,  `pytorch-lightning`, `torch`,  `torchvision` and `wandb` packages. For this, you need to run the following commands in your terminal:

```bash
pip install dstack
pip install pytorch-lightning
pip install torch
pip install torchvision
pip install wandb
```

# Directory Setup

We follow the directory structure below, where our main working directory is `dstack_test`.

```
dstack_test/
    .dstack/
        workflows.yaml
        variables.yaml
    train.py 
    requirements.txt 
```

The file `train.py` contains our deep learning model and the rest of the machine learning pipeline to train that model. 

The file `requirements.txt` contain all the packages required to train our deep learning model.

It has the following lines:

```
torch
torchvision
pytorch-lightning
wandb
```
There is no need to add `dstack` to the `requirements.txt` file.

The rest of the files will be detailed later in this blog.

# Simple Deep Learning Model

We rely on Pytorch Lightning, a deep learning framework to train our model.
For further information on Pytorch Lightning. please see [here](https://www.pytorchlightning.ai/).

For the purpose of visualization of the results, we used `wandb` package.  For more details regarding `wandb`, please see [here](https://wandb.ai/site).

The contents of the `train.py` file are given below and are mostly self explanatory. The training process is very similar to training a deep learning in pure Pytorch.

```python

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
import wandb


class LitAutoEncoder(pl.LightningModule):
    """
    Autoencoder model with Pytorch Lightning

    This class object contains all the required methods
    """

    # Simple Autoencoder
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28))

    # Forward pass of the encoder
    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    # Optimizer init
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    # Training step
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, sync_dist=True)
        return loss

    # Validation step
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)


def main():
    """
    Main function that handles all the dataset pre-processing, 
    instantiating the model  and training that model.
    """
    # download and pre-process the MNIST dataset
    dataset = MNIST('data', train=True, download=True,
                    transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    # check if cuda is available
    # and get number of gpus into the variable num_gpus
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 0

    # choose accelerator based on the number of gpus
    if num_gpus == 0:
        accelerator_name = 'cpu'
    else:
        accelerator_name = 'gpu'

    if num_gpus == 0:
        # setting number of processes for cpu
        num_devices = 2
        batch_size = 32
    else:
        # setting number of devices to be exactly the number of gpus
        num_devices = num_gpus
        # changing batch size accordingly
        batch_size = int(32*num_gpus)

    # Instantiate the dataloader on training dataset
    # and the validation dataset with appropriate batch_size
    train_loader = DataLoader(
                            mnist_train,
                            batch_size=batch_size,
                            num_workers=8,
                            pin_memory=True)
    val_loader = DataLoader(mnist_val,
                            batch_size=batch_size,
                            num_workers=8,
                            pin_memory=True)

    # Instantiate model instance
    model = LitAutoEncoder()

    # trainer instance with appropriate settings
    trainer = pl.Trainer(accelerator=accelerator_name,
                         limit_train_batches=0.5, max_epochs=10,
                         logger=wandb_logger,
                         devices=num_devices, strategy="ddp")

    # fit with trainer
    print('starting to fit')
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    # wandb login
    wandb.login(key='efd2c2663c2c06466197879239f29686dec4fbad')

    # wandb log results to a project 
    wandb_logger = WandbLogger(project="my-test-project")

    # running the deep learning model now
    main()
```
The `LitAutoEncoder` object contains our  Autoencoder model. The `main` function handles the `DataLoader` parts for our training and validation datasets, and also the  `pl.Trainer` object. 

With Pytorch Lightning, we use the `pl.Trainer` object to specify the training type. In particular, depending of the number of GPUs on the device (can be checked via `torch.cuda.is_available()`) we have to set the arguments `gpus`, `accelerator` appropriately. For CPU, we need to set `accelerator = 'cpu'` and for the rest of cases, we set `accelerator = 'gpu'`. 

We set the `strategy` of `pl.Trainer` object to `ddp` and `logger=wandb_logger` for  logging our results to wandb. There exist several other strategies for distributed training in Pytorch Lightning, regarding which the information can be found [here](https://pytorch-lightning.readthedocs.io/en/1.5.0/advanced/multi_gpu.html). [TODO: Change documentation link.] The authors of Pytorch Lightning recommend `ddp` strategy, as it is faster than the usual `dp` strategy. 

The `batch_size` parameter of the dataloaders can be tuned according to the number of devices under consideration.

Finally, in order to fit the model instance with the training data and obtain results also on the validation data, we have the call the `fit` method of `trainer` object as following:
`trainer.fit(model, train_loader, val_loader)`

For tracking the results, we used `wandb` for which we can login using the following (Please change the key accordingly):

`wandb.login(key='efd2c2663c2c06466197879239f29686dec4fbad')`

In order to keep the logged results under a wandb project, we use 

`wandb_logger = WandbLogger(project="my-test-project")`


# Our Dstack Workflow

Dstack is a comprehensive framework to automate the process of training deep learning models on the cloud. Typically, one is required is lauch a GPU/CPU instance on cloud using a vendor like AWS or Google Cloud or Azure and install the required packages. Then, download the git repository to train the deep learning models.

Dstack automates this entire process via a specification of the requirements in declarative configuration files. For more details, please see [here](https://docs.dstack.ai/).

Firstly, create a account at `dstack.ai` and configure the settings appropriately according to the CPU/GPU requirements. In our case, it looks like the following:
 ![AWS settings](/blog_figures/fig_1.png)
Since multiple GPUs  are required for our workflow, we may need to add `p3.8xlarge` GPU instance of AWS in the dstack settings. In order to do this, click on the settings tab on the left side of the [dstack.ai](https://dstack.ai) interface. In the settings frame, there is AWS tab, where we can see a button `Add a limit`. On clicking that button, you can select the  `p3.8xlarge` GPU instance of AWS.

This workflow needs to be be added in `.dstack/workflows.yaml` file. The contents of this file should be akin to the following:

```yaml
workflows:
  - name: WORKFLOW_1
```



Dstack extracts the requirements to run the deep learning model from `requirements.txt`, which we pass it as a value to requirements property in the workflow. We consider the workflow, where we train our deep learning model on multiple GPUs. We require four GPUs, we specify that under `resources` key.

The contents of the `.dstack/workflows.yaml` file  looks like below.

```yaml
workflows:
  - name: train-mnist-multi-gpu
    provider: python
    version: 3.9
    requirements: requirements.txt
    script: train.py
    artifacts:
      - data
    resources:
      gpu: ${{ gpu }}
```

The contents of the `.dstack/variables.yaml` file looks like below.

```yaml
variables:
  train-mnist-multi-gpu:
    gpu: 4
```

In order to run the workflow all we have to do is to execute the following command in the terminal

```bash
dstack run train-mnist-multi-gpu
```

Now, open [dstack.ai](https://dstack.ai) to see the workflows (after you perform the login). You will see contents of the `Runs` tab like below.

![Workflows](/blog_figures/fig_3.png)

You can check each workflow by clicking on the respective button. In the `Logs` tab, you will see the cloud server running the `train.py` after few minutes of starting the job. 

In the `Runners` tab on the left side, you will find information on the specific instances being used.

 
In the `Logs` of the run under consideration, after a few moments, towards the end of the `Logs`, you will information like below.

`2022-05-13 14:15 wandb: Synced prime-shape-23: https://wandb.ai/mmahesh/my-test-project/runs/1kwcoyo3`

Go to that particular url to see the plots of the results, as shown below.
![Wandb results 2](/blog_figures/fig_5.png)

You will also find several other related information, like system information and many others, as shown below.

![Wandb results 1](/blog_figures/fig_4.png)



# References

- [Dstack](https://docs.dstack.ai)
- [Pytorch Lightning](https://www.pytorchlightning.ai/)
- [Wandb](https://wandb.ai/site)
