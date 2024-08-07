#!/usr/bin/env python

import os, sys, glob
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from decoder import ClosurePhaseDecoder
from torch import FloatTensor, arccos
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from itertools import product


# Define a custom Dataset class
class PhiDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(self.h5_file, 'r') as f:
            self.length = len(f['phase'])

    def open_hdf5(self):
        # solves issue where hdf5 file opened in __init__ prevents multiple
        # workers: https://github.com/pytorch/pytorch/issues/11929
        self.file = h5py.File(self.h5_file, 'r')
        self.inputs = self.file["Phi_marginal"]
        self.targets = self.file["phase"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not hasattr(self, self.h5_file):
            self.open_hdf5()
        return FloatTensor(self.inputs[idx]), FloatTensor(self.targets[idx])


class cosPhiDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(self.h5_file, 'r') as f:
            self.length = len(f['phase'])

    def open_hdf5(self):
        # solves issue where hdf5 file opened in __init__ prevents multiple
        # workers: https://github.com/pytorch/pytorch/issues/11929
        self.file = h5py.File(self.h5_file, 'r')
        self.inputs = self.file["cosPhi_marginal"]
        self.targets = self.file["phase"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not hasattr(self, self.h5_file):
            self.open_hdf5()
        return FloatTensor(self.inputs[idx]), FloatTensor(self.targets[idx])


class TrainingRunner:
    def __init__(self, training_h5, validation_h5, testing_h5,
                 linear_only=False):
        self.training_h5 = training_h5
        self.validation_h5 = validation_h5
        self.testing_h5 = testing_h5
        self.linear_only = linear_only

        # get dataloaders
        self.set_dataloaders()

        # dimensions
        self.input_size = next(iter(self.train_loader))[0].size(-1) ** 2
        self.output_size = next(iter(self.train_loader))[1].size(-1)

        # directories
        self.checkpoint_dir = "./checkpoints"

    def get_custom_dataloader(self, h5_file, batch_size=128, shuffle=True,
                              linear_only=False):
        if linear_only:
            dataset = PhiDataset(h5_file)
        else:
            dataset = cosPhiDataset(h5_file)

        # We can use DataLoader to get batches of data
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=16, persistent_workers=True,
                                pin_memory=True)

        return dataloader

    def set_dataloaders(self, batch_size=128):
        self.batch_size = batch_size
        self.train_loader = self.get_custom_dataloader(self.training_h5, linear_only=self.linear_only, batch_size=self.batch_size)
        self.valid_loader = self.get_custom_dataloader(self.validation_h5, linear_only=self.linear_only, batch_size=self.batch_size, shuffle=False)
        self.test_loader = self.get_custom_dataloader(self.testing_h5, linear_only=self.linear_only, batch_size=self.batch_size, shuffle=False)

    def train_model(self, model_name, save_name=None, **kwargs):
        """Train model.

        Args:
            model_name: Name of the model you want to run. Is used to look up the class in "model_dict"
            save_name (optional): If specified, this name will be used for creating the checkpoint and logging directory.
        """
        if save_name is None:
            save_name = model_name

        # logger
        logger = WandbLogger(project='triple_correlation',
                             group=model_name, log_model=True,
                             save_dir=os.path.join(self.checkpoint_dir, save_name))

        # callbacks
        # early stopping
        early_stop_callback = EarlyStopping(monitor="val_loss",
                                            min_delta=0.00,
                                            patience=5,
                                            verbose=True,
                                            mode="min")
        checkpoint_callback = ModelCheckpoint(save_weights_only=True,
                                              mode="min", monitor="train_loss")
        # Save the best checkpoint based on the maximum val_acc recorded.
        # Saves only weights and not optimizer

        # Create a PyTorch Lightning trainer with the generation callback
        trainer = L.Trainer(
            default_root_dir=os.path.join(self.checkpoint_dir, save_name),
            accelerator="gpu",
            devices=[0],
            max_epochs=180,
            callbacks=[early_stop_callback, checkpoint_callback],
            check_val_every_n_epoch=10,
            logger=logger
        )

        # L.seed_everything(42)  # To be reproducible
        model = ClosurePhaseDecoder(model_name=model_name, **kwargs)
        trainer.fit(model, self.train_loader, self.valid_loader)

        # Load best checkpoint after training
        model = ClosurePhaseDecoder.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)

        # Test best model on validation and test set
        val_result = trainer.test(model, dataloaders=self.valid_loader,
                                  verbose=False)
        test_result = trainer.test(model, dataloaders=self.test_loader,
                                   verbose=False)
        result = {"test": test_result[0]["test_acc"],
                  "val": val_result[0]["test_acc"]}

        logger.experiment.finish()

        return model, result

    def train_linear_model(self, model_name, save_name=None, **kwargs):
        """Train model.

        Args:
            model_name: Name of the model you want to run. Is used to look up the class in "model_dict"
            save_name (optional): If specified, this name will be used for creating the checkpoint and logging directory.
        """
        if save_name is None:
            save_name = model_name

        # logger
        logger = WandbLogger(project='triple_correlation',
                             group=model_name, log_model=True,
                             save_dir=os.path.join(self.checkpoint_dir,
                                                   save_name))

        # callbacks
        # early stopping
        early_stop_callback = EarlyStopping(monitor="val_loss",
                                            min_delta=0.00,
                                            patience=5,
                                            verbose=True,
                                            mode="min")
        checkpoint_callback = ModelCheckpoint(save_weights_only=True,
                                              mode="min", monitor="train_loss")
        # Save the best checkpoint based on the maximum val_acc recorded.
        # Saves only weights and not optimizer

        # Create a PyTorch Lightning trainer with the generation callback
        trainer = L.Trainer(
            default_root_dir=os.path.join(self.checkpoint_dir, save_name),
            accelerator="gpu",
            devices=[0],
            max_epochs=180,
            callbacks=[early_stop_callback, checkpoint_callback],
            check_val_every_n_epoch=10,
            logger=logger
        )

        # L.seed_everything(42)  # To be reproducible
        model = ClosurePhaseDecoder(model_name=model_name, **kwargs)
        trainer.fit(model, self.train_loader, self.valid_loader)

        # Load best checkpoint after training
        model = ClosurePhaseDecoder.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)

        # Test best model on validation and test set
        val_result = trainer.test(model, dataloaders=self.valid_loader,
                                  verbose=False)
        test_result = trainer.test(model, dataloaders=self.test_loader,
                                   verbose=False)
        result = {"test": test_result[0]["test_acc"],
                  "val": val_result[0]["test_acc"]}

        logger.experiment.finish()

        return model, result

    def scan_hyperparams(self):
        for lr, num_layers, num_conv_layers, kernel_size, activation in product([1e-2, 3e-2],
                                                                [1],
                                                                [5],
                                                                [7],
                                                                ["LeakyReLU"]):

            model_config = {"num_layers": num_layers,
                            "num_conv_layers": num_conv_layers,
                            "activation": activation,
                            "norm": False,
                            "input_size": self.input_size,
                            "output_size": self.output_size,
                            "kernel_size": kernel_size,}
            optimizer_config = {"lr": lr,
                                "momentum": 0.9,}
            misc_config = {"batch_size": self.batch_size}

            self.train_model(model_name="WideCNN",
                             model_hparams=model_config,
                             optimizer_name="SGD",
                             optimizer_hparams=optimizer_config,
                             misc_hparams=misc_config)

    def scan_linear_hyperparams(self):
        for lr, num_layers, hidden_size, Phi_sign in product([1e-2, 3e-2],
                                                    [2, 3],
                                                    [self.input_size, 2*self.input_size, 3*self.input_size],
                                                    [True, False]):

            model_config = {"num_layers": num_layers,
                            "norm": False,
                            "input_size": self.input_size,
                            "output_size": self.output_size,
                            "hidden_size": hidden_size,
                            "Phi_sign": Phi_sign,}
            optimizer_config = {"lr": lr,
                                "momentum": 0.9,}
            misc_config = {"batch_size": self.batch_size}

            self.train_linear_model(model_name="LinearNet",
                             model_hparams=model_config,
                             optimizer_name="SGD",
                             optimizer_hparams=optimizer_config,
                             misc_hparams=misc_config)

    def load_model(self):
        # Check whether pretrained model exists. If yes, load it and skip training
        print(self.checkpoint_dir)
        pretrained_filename = os.path.join(self.checkpoint_dir,"WideCNN", "triple_correlation", "nws62xci",
                                           "checkpoints", "*" + ".ckpt")
        print(pretrained_filename)
        if os.path.isfile(glob.glob(pretrained_filename)[0]):
            pretrained_filename = glob.glob(pretrained_filename)[0]
            print(
                f"Found pretrained model at {pretrained_filename}, loading...")
            # Automatically loads the model with the saved hyperparameters
            model = ClosurePhaseDecoder.load_from_checkpoint(pretrained_filename)

            return model

    def plot_predictions(self):

        model = self.load_model()
        trainer = L.Trainer(
            accelerator="gpu",
            devices=[0]
        )
        y = trainer.predict(model, dataloaders=self.train_loader)

        for i in range(len(y[0][0].numpy()[:,0])):
            plt.plot(y[0][0].numpy()[i,:], label="Predictions")
            plt.plot(y[0][1].numpy()[i,:], label="Targets")
            plt.legend()
            plt.show()
