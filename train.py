
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
		accelerator_name = 'dp' # TODO: Later change this to dp
	else:
		raise 

	
	# trainer instance with appropriate settings
	trainer = pl.Trainer(gpus=num_gpus, 							  accelerator=accelerator_name,
                      limit_train_batches=0.5, max_epochs=10, logger=wandb_logger)

	# fit with trainer 
	print('starting to fit')
	trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
	wandb_logger = WandbLogger(project="my-test-project")

	# running the deep learning model now
	main()
