#!/usr/bin/env python

from torch import optim, nn
import torch
import lightning as L


# define the LightningModule
class ClosurePhaseDecoder(L.LightningModule):
	def __init__(self, model, kappa=0.1, lr=1e-3):
		super().__init__()
		self.kappa = kappa
		self.lr = lr
		self.save_hyperparameters()
		self.model = model

	def antisymmetry_loss(self, outputs):
		# Should this be using views instead of slicing?
		loss = 0
		positive_x = outputs[:, outputs.size(1) // 2 + 1:]
		negative_x = torch.flip(outputs[:, :outputs.size(1) // 2], [1])
		# punish any phase that is not antisymmetric and does not pass through zero
		loss = torch.sum(torch.add(positive_x, negative_x)**2) + torch.sum(
			outputs[:, outputs.size(1) // 2]**2)
		return loss

	def training_step(self, batch, batch_idx):
		# training_step defines the train loop.
		# it is independent of forward
		x, y = batch
		x = x.view(-1, x.size(1)**2)
		y_hat = self.model(x)
		loss = (nn.functional.mse_loss(y_hat, y) + self.kappa *
				self.antisymmetry_loss(y_hat))
		# Logging to TensorBoard (if installed) by default
		self.log("train_loss", loss, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		# validation_step defines the validation loop.
		x, y = batch
		x = x.view(-1, x.size(1)**2)
		y_hat = self.model(x)
		# to make models comparable from check point, don't multiply hyperparameters
		# into the loss function, but compute the total loss with maximal hyperparameters for every model
		loss = nn.functional.mse_loss(y_hat, y) + self.antisymmetry_loss(y_hat)
		# Logging to TensorBoard (if installed) by default
		self.log("val_loss", loss, prog_bar=True)
		return loss

	def configure_optimizers(self):
		# optimizer = optim.AdamW(self.parameters(), lr=self.lr)
		optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
		return optimizer

	def predict_step(self, batch, batch_idx, dataloader_idx=0):
		x, y = batch
		x = x.view(-1, x.size(1)**2)
		y_hat = self.model(x)
		return y_hat, y