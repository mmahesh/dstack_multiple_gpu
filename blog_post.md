# Introduction

In this blog, we are going to see how to run your deep learning models on cloud based compute servers via [dstack](https://dstack.ai). The documentation for dstack can be found [here](https://docs.dstack.ai).

Dstack is a framework to automate the entire workflow of managing the training of a deep learning model on a cloud based GPU/CPU server. 

The pre-requisites to understand this blog include:

    - familiarity with deep learning,
    - familiarity with python,
    - basic familiarity with pytorch (a popular deep learning framework).

We use `python 3` version for this blog. The contents of this blog include the following:

    - introduction to pytorch lightning,
    - create a simple deep learning model to train on MNIST dataset,
    - creating dstack workflow,
    - and run the deep learning on AWS GPU/CPU instances using dstack.

# Requirements

Firstly, we need to install `dstack`,  `pytorch-lightning`, `torch` and `torchvision` packages. For this, you need to run the following commands in your terminal:

```bash
pip install dstack==0.0.4rc8 
pip install pytorch-lightning==1.6.2
pip install torch
pip install torchvision
```

# Directory setup

We follow the directory structure below, where our main working directory is `dstack_test`.

```
dstack_test/
    .dstack/
        workflows.yaml
    train.py 
    run_exps.sh
    requirements.txt 
```

The file `train.py` contains our deep learning model and the rest of the machine learning pipeline to train that model. 

The file `requirements.txt` contain all the packages required to train our deep learning model.

It has the following lines:

```
torch
torchvision
pytorch-lightning==1.6.2
```
The rest of the files will be detailed later in this blog.

# Brief Primer on Pytorch Lightning

As mentioned earlier,  we use the Pytorch Lightning framework to create our deep learning models.

Pytorch Lightning is a deep learning framework built on pure PyTorch without having to write the boilerplate code. In essence, the training of complex deep learning models becomes straightforward. For example, with Pytorch Lightning the training of a deep learning model on a CPU, a GPU, and multiple GPUs can be done seamlessly without having to write large chunks of code for each case seperately.   

For further information, please see [here](https://www.pytorchlightning.ai/).

## Pytorch Lightning Imports
We now focus on the contents of the `train.py` file in our current working directory. 

Typically, we have to add the following import statements for a pytorch script.

```python
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
```

With Pytorch Lightning, we have to add one additional import statement as below.

```python
import pytorch_lightning as pl
```

# Simple Deep Learning Model
We continue editing the content of the `train.py` file in our current working directory.

We consider the standard deep learning example from the Pytorch Lightning website. We use a simple Autoencoder for training on the MNIST dataset. In the MNIST dataset, there are 60000 images of size 28 x 28. The Autoencoder is divided into two components, namely the encoder component and the decoder component. 

The encoder component typically takes the input signal and compresses the signal to a latent space of much lower dimension. The decoder component tries to reconstruct the signal using the information from the latent space. 

In Pytorch Lightning, we can capture all the components of the training of the deep learning model under a single python class, which we call 

```python
class LitAutoEncoder(pl.LightningModule):
```

Note that the class `LitAutoEncoder` inherits all the methods of `pl.LightningModule` object.

We use the  `__init__` method to create the model. We describe the components of the `__init__` method first.

Firstly, the encoder component involves a Linear layer with input size 28 x 28 and output size 64. Then, we apply a ReLU activation function and pass it to another linear layer with input size 64 and output size 3. The code for the encoder is the following:

```python
self.encoder = nn.Sequential(
                nn.Linear(28 * 28, 64),
                nn.ReLU(),
                nn.Linear(64, 3))
```

The decoder component involves the a Linear layer with input size 3  and output size 64. Then, we apply a ReLU activation function and pass it to another linear layer with input size 64 and output size 28 x 28. The code for the encoder is the following:

```python
self.decoder = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 28 * 28))
```

The full `__init__` method  looks like below.

```python
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
```


Apart from the  `__init__` method, the we have the following methods in `LitAutoEncoder` class, which we will describe below:

    - `forward` method,
    - `training_step` method,
    - `validation_step` method,
    - `configure_optimizers` method.

The output embedding of the encoder can be obtained by the following function:

```python
def forward(self, x):
    embedding = self.encoder(x)
    return embedding
```

The training part involves passing images of batch size 32 via a dataloader to the deep learning model which is a conjunction of encoder and the decoder. One the output is obtained, it is compared with the original images and then mean squared loss is computed. This entire process is captured by the following function:

```python
def training_step(self, train_batch, batch_idx):
    x, y = train_batch
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = F.mse_loss(x_hat, x)
    self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
    return loss
```

The same process for the validation dataset is captured by the following function:

```python
def validation_step(self, val_batch, batch_idx):
    x, y = val_batch
    x = x.view(x.size(0), -1)
    z = self.encoder(x)
    x_hat = self.decoder(z)
    loss = F.mse_loss(x_hat, x)
    self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
```



In order to minimize the loss, we use the Adam optimizer, which is a popular optimizer that is well known for training deep learning models efficiently. The optimizer is configured by the following function:

```python
def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer
```


Collecting all the above methods under the class `LitAutoEncoder` we finally have 

```python
class LitAutoEncoder(pl.LightningModule):
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
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    # Validation step
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
```

Now that our model is setup, we need to consider the other part of training the deep learning model which is handled by the `main()` function given below:

```python
def main():
	"""
	Main function that handles all the dataset pre-processing, 
	instantiating the model  and training that model.
	"""
	# download and pre-process the MNIST dataset
	dataset = MNIST('data', train=True, download=True, \
		transform=transforms.ToTensor())
	mnist_train, mnist_val = random_split(dataset, [55000, 5000])

	# Instantiate the dataloader on training dataset 
	# and the validation dataset with appropriate batch_size
	train_loader = DataLoader(mnist_train, batch_size=32, pin_memory=True)
	val_loader = DataLoader(mnist_val, batch_size=32,
							pin_memory=True)

	# Instantiate model instance 
	model = LitAutoEncoder()

	# check if cuda is available 
	# and get number of gpus into the variable num_gpus
	if torch.cuda.is_available():
		num_gpus = torch.cuda.device_count()
	else:
		num_gpus = 0

	# choose accelerator based on the number of gpus
	if num_gpus == 0:
		accelerator_name = 'cpu'
	elif num_gpus == 1:
		accelerator_name = 'gpu'
	elif num_gpus > 1:
		accelerator_name = 'gpu' # TODO: Later change this to dp
	else:
		raise 

	
	# trainer instance with appropriate settings
	trainer = pl.Trainer(gpus=num_gpus, accelerator=accelerator_name,
						 limit_train_batches=0.5, max_epochs=1)

	# fit with trainer 
	print('starting to fit')
	trainer.fit(model, train_loader, val_loader)
```

The contents of the main function are mostly self explanatory and training process is very similar to training a deep learning in pure Pytorch. Here, with Pytorch Lightning, we use the `pl.Trainer` object to specify the training type. In particular, depending of the number of GPU on the device we have to set the arguments `gpus`, `accelerator` appropriately. For CPU, we need to set `accelerator = 'cpu'` and for the rest of cases, we set `accelerator = 'gpu'`. 

# Our Dstack Workflow

Dstack is a comprehensive framework to automate the process of training deep learning models on the cloud. Typically, one is required is lauch a GPU/CPU instance on cloud using a vendor like AWS or Google Cloud or Azure and install the required packages. Then, download the git repository where one is required to do the experiments and then perform the training.

Dstack automates this entire process via a specification of the requirements in declarative configuration files. For more details, please see [here](https://docs.dstack.ai/).

Firstly, create a account at `dstack.ai` and configure the settings appropriately according to the CPU/GPU requirements. In our case, it looks like the following:
 ![AWS settings](/blog_figures/fig_1.png)


We consider three workflows, where we train our deep learning model on a CPU, a GPU and on multiple GPUs. These workflows need to be be added in `.dstack/workflows.yaml` file. The contents of this file should be akin to the following:

```yaml
workflows:
  - name: WORKFLOW_1
  - name: WORKFLOW_2
  - name: WORKFLOW_3
```

The workflow pertaining to training our deep learning model on a CPU is the following:

```yaml
name: train-mnist-no-gpu
provider: python
requirements: requirements.txt
python_script: train.py
artifacts:
  - data
resources:
  cpu: 1
```

As seen above, we named the workflow to reflect the fact that we do not use a GPU. The provider property is set to python, and under the hood dstack uses python 3.10 version. 

Dstack extracts the requirements to run the deep learning model from `requirements.txt`, which we pass it as a value to requirements property in the workflow. Since, we require one CPU instance, we specify that under `resources` key. 

The workflow pertaining to training our deep learning model on a GPU is the following:

```yaml
name: train-mnist-one-gpu
provider: python
requirements: requirements.txt
python_script: train.py
artifacts:
  - data
resources:
  gpu: 1
```

The workflow pertaining to training our deep learning model on multiple GPUs is the following:

```yaml
name: train-mnist-multi-gpu
provider: python
requirements: requirements.txt
python_script: train.py
artifacts:
  - data
resources:
  gpu: 4
```




The contents of the `.dstack/workflows.yaml` file finally looks like below.

```yaml
workflows:
  - name: train-mnist-no-gpu
    provider: python
    requirements: requirements.txt
    python_script: train.py
    artifacts:
      - data
    resources:
      cpu: 1

  - name: train-mnist-one-gpu
    provider: python
    requirements: requirements.txt
    python_script: train.py
    artifacts:
      - data
    resources:
      gpu: 1

  - name: train-mnist-multi-gpu
    provider: python
    requirements: requirements.txt
    python_script: train.py
    artifacts:
      - data
    resources:
      gpu: 4
```

# Automation

In order to run a workflow, say `train-mnist-no-gpu`, all we have to do is to execute the following command in the terminal

```bash 
dstack run train-mnist-no-gpu
```

dstack automatically provisions the AWS instance required on-demand.

In order to run all the workflows in an automated manner, we create a bash script `run_exps.sh` with the following contents:

```bash
dstack run train-mnist-no-gpu
wait 
dstack run train-mnist-one-gpu
wait 
dstack run train-mnist-multi-gpu
wait 
```

In order to make the above bash script an executable, you need to run  the following command in your terminal:

```bash
chmod +x run_exps.sh
```

Then, finally in order to run the experiments, run the following command in your terminal:

```bash
./run_exps.sh
```

Now, open [dstack.ai](https://dstack.ai) to see the workflows (after you perform the login). You will see contents of the `Runs` tab like below.

![Workflows](/blog_figures/fig_2.png)

You can check each workflow by clicking on the respective button. In the `Runners` tab, you will find information of the specific instances being used.


> For the workflow `train-mnist-multi-gpu`, since multiple GPUs are required you may need to add `p3.8xlarge` GPU instance of AWS in the dstack settings. In order to do this, click on the settings tab on the left side of the [dstack.ai](https://dstack.ai) interface. In the settings frame, there is AWS tab, where we can see a button `Add a limit`. On clicking that button, you can select the  `p3.8xlarge` GPU instance of AWS. In the end, you should see the following in the dstack website:
 ![AWS settings](/blog_figures/fig_1.png)

# Conclusion

In the above blog, we have seen how to 

- create a deep learning model using Pytorch Lightning,
- choose appropriate settings for the dstack workflows,
- and run the dstack workflows using AWS CPU/GPU instances.

# References

- [Dstack documentation](https://docs.dstack.ai)
- [Pytorch Lightning](https://www.pytorchlightning.ai/)
