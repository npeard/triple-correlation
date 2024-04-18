#!/usr/bin/env python

from torch.utils.data import Dataset, DataLoader
import h5py
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from clodecoder import ClosurePhaseDecoder
import models
from torch import FloatTensor
import numpy as np
import matplotlib as plt
import lightning as L


# Define a custom Dataset class
class CustomDataset(Dataset):
	def __init__(self, h5_file):
		self.h5_file = h5_file
		self.file = h5py.File(self.h5_file, 'r')
		self.inputs = np.asarray(self.file["cosPhi_marginal"])
		self.targets = np.asarray(self.file["phase"])

	def __len__(self):
		return len(self.targets)

	def __getitem__(self, idx):
		return FloatTensor(self.inputs[idx]), FloatTensor(self.targets[idx])


class TrainingRunner:
	def __init__(self, training_h5, validation_h5, testing_h5):
		self.train_loader = self.get_custom_dataloader(training_h5,
													   batch_size=128)
		self.valid_loader = self.get_custom_dataloader(validation_h5,
													   batch_size=128,
													   shuffle=False)
		self.test_loader = self.get_custom_dataloader(testing_h5,
													  batch_size=128,
													  shuffle=False)

		# dimensions
		self.num_inputs = next(iter(self.train_loader))[0].size(-1)**2
		self.num_outputs = next(iter(self.train_loader))[1].size(-1)

		# directories
		self.checkpoint_dir = "checkpoints"

	def get_custom_dataloader(self, h5_file, batch_size=128, shuffle=True):
		# We can use DataLoader to get batches of data
		dataset = CustomDataset(h5_file)
		dataloader = DataLoader(dataset, batch_size=batch_size,
								shuffle=shuffle)  # , num_workers=31)#, persistent_workers=True)
		return dataloader

	def assign_color(number):
		colors = ["red", "green", "blue", "yellow", "purple", "cyan"]
		return colors[number % len(colors)]

	def train_singleLinear(self):
		# checkpoints
		# saves top-K checkpoints based on "val_loss" metric
		checkpoint_callback = ModelCheckpoint(
			save_top_k=5,
			monitor="val_loss",
			mode="min",
			dirpath=self.checkpoint_dir,
			filename="singleLinear-{epoch:02d}-{val_loss:.2f}",
		)

		# assign colors based on learning rate
		fig, (ax1, ax2) = plt.subplots(1, 2)
		for kappa in [1e-6]:  # , 1e-5, 1e-4]:
			for i, lr in enumerate([5e-2]):  # , 1e-2, 5e-3, 1e-3]):
				# early stopping
				early_stop_callback = EarlyStopping(monitor="val_loss",
													min_delta=0.00, patience=3,
													verbose=True, mode="min")

				# model
				sequential_model = ClosurePhaseDecoder(
					networks.Linear(self.num_inputs, self.num_outputs),
					kappa=kappa,
					lr=lr)

				# train model
				trainer = L.Trainer(max_epochs=100,
									callbacks=[early_stop_callback,
											   checkpoint_callback],
									check_val_every_n_epoch=10)
				trainer.fit(sequential_model, self.train_loader,
							self.valid_loader)
				final_val_loss = trainer.callback_metrics['val_loss']
				print(final_val_loss)

				# plot for trends
				ax1.scatter(kappa, final_val_loss, color=self.assign_color(i))
				ax1.set_xlabel("kappa")
				ax1.set_ylabel("val_loss")

				ax2.scatter(lr, final_val_loss, color=self.assign_color(i))
				ax2.set_xlabel("lr")
				ax2.set_ylabel("val_loss")

		plt.tight_layout()
		plt.show()

	def train_multiLinear(self):
		# checkpoints
		# saves top-K checkpoints based on "val_loss" metric
		checkpoint_callback = ModelCheckpoint(
			save_top_k=5,
			monitor="val_loss",
			mode="min",
			dirpath="/content/drive/MyDrive/Colab Notebooks/TriPhase ML/",
			filename="multilinear-{epoch:02d}-{val_loss:.2f}",
		)

		# assign colors based on model size
		fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
		for kappa in [1e-6, 1e-5, 1e-4]:
			for lr in [5e-2, 1e-2, 5e-3, 1e-3]:
				for i, hidden_size in enumerate(
						[2 * self.num_inputs, self.num_inputs,
						 5 * self.num_outputs,
						 self.num_outputs]):
					# early stopping
					early_stop_callback = EarlyStopping(monitor="val_loss",
														min_delta=0.00,
														patience=3,
														verbose=True,
														mode="min")

					# model
					multilinear_model = ClosurePhaseDecoder(
						networks.MultiLinear(self.num_inputs, hidden_size,
										   self.num_outputs),
						kappa=kappa, lr=lr)

					# train model
					trainer = L.Trainer(max_epochs=100,
										callbacks=[early_stop_callback,
												   checkpoint_callback],
										check_val_every_n_epoch=10)
					trainer.fit(multilinear_model, self.train_loader,
								self.valid_loader)
					final_val_loss = trainer.callback_metrics['val_loss']
					print(final_val_loss)

					# plot for trends
					ax1.scatter(kappa, final_val_loss,
								color=self.assign_color(i))
					ax1.set_xlabel("kappa")
					ax1.set_ylabel("val_loss")

					ax2.scatter(lr, final_val_loss, color=self.assign_color(i))
					ax2.set_xlabel("lr")
					ax2.set_ylabel("val_loss")

					ax3.scatter(hidden_size, final_val_loss,
								color=self.assign_color(i))
					ax3.set_xlabel("hidden_size")
					ax3.set_ylabel("val_loss")

		plt.tight_layout()
		plt.show()

	def train_sequential(self):
		# checkpoints
		# saves top-K checkpoints based on "val_loss" metric
		checkpoint_callback = ModelCheckpoint(
			save_top_k=5,
			monitor="val_loss",
			mode="min",
			dirpath="/content/drive/MyDrive/Colab Notebooks/TriPhase ML/",
			filename="sequential-{epoch:02d}-{val_loss:.2f}",
		)

		# assign colors based on model size
		fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
		for kappa in [1e-6]:  # , 1e-5, 1e-4]:
			for lr in [5e-2]:  # , 1e-2, 5e-3, 1e-3]:
				for i, hidden_size in enumerate([
													2 * self.num_inputs]):  # ,
					# num_inputs, 5*num_outputs, num_outputs]):
					# early stopping
					early_stop_callback = EarlyStopping(monitor="val_loss",
														min_delta=0.00,
														patience=3,
														verbose=True,
														mode="min")

					# model
					sequential_model = ClosurePhaseDecoder(
						networks.SequentialNN(self.num_inputs, hidden_size,
										   self.num_outputs),
						kappa=kappa, lr=lr)

					# train model
					trainer = L.Trainer(accelerator="gpu", devices=1,
										max_epochs=100,
										callbacks=[early_stop_callback],
										check_val_every_n_epoch=10)
					trainer.fit(sequential_model, self.train_loader,
								self.valid_loader)
					final_val_loss = trainer.callback_metrics['val_loss']
					print(final_val_loss)

					# plot for trends
					ax1.scatter(kappa, final_val_loss,
								color=self.assign_color(i))
					ax1.set_xlabel("kappa")
					ax1.set_ylabel("val_loss")

					ax2.scatter(lr, final_val_loss, color=self.assign_color(i))
					ax2.set_xlabel("lr")
					ax2.set_ylabel("val_loss")

					ax3.scatter(hidden_size, final_val_loss,
								color=self.assign_color(i))
					ax3.set_xlabel("hidden_size")
					ax3.set_ylabel("val_loss")

		plt.tight_layout()
		plt.show()

	def load_checkpoint(self):
		model = ClosurePhaseDecoder.load_from_checkpoint(
			"/content/drive/MyDrive/Colab Notebooks/TriPhase ML/sequential-epoch=39-val_loss=3.17.ckpt")

		print(model.kappa)
		print(model.lr)

		# disable randomness, dropout, etc...
		model.eval()

		# predict with the model
		y_hat = model(x)