#!/usr/bin/env python

import os, sys, glob
from torch import nn
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
        self.opened_flag = False

    def open_hdf5(self):
        # solves issue where hdf5 file opened in __init__ prevents multiple
        # workers: https://github.com/pytorch/pytorch/issues/11929
        self.file = h5py.File(self.h5_file, 'r')
        self.inputs = self.file["Phi_marginal"]
        self.targets = self.file["phase"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not self.opened_flag:  # not hasattr(self, 'h5_file'):
            self.open_hdf5()
            self.opened_flag = True
            # print("open_hdf5 finished")
        return FloatTensor(self.inputs[idx]), FloatTensor(self.targets[idx])


class cosPhiDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(self.h5_file, 'r') as f:
            self.length = len(f['phase'])
        self.opened_flag = False

    def open_hdf5(self):
        # solves issue where hdf5 file opened in __init__ prevents multiple
        # workers: https://github.com/pytorch/pytorch/issues/11929
        self.file = h5py.File(self.h5_file, 'r')
        self.inputs = self.file["cosPhi_marginal"]
        self.targets = self.file["phase"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not self.opened_flag:  # not hasattr(self, 'h5_file'):
            self.open_hdf5()
            self.opened_flag = True
            # print("open_hdf5 finished")
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
                                              mode="min", monitor="train_loss",
                                              save_top_k=3)
        # Save the best checkpoint based on the maximum val_acc recorded.
        # Saves only weights and not optimizer

        # Create a PyTorch Lightning trainer with the generation callback
        trainer = L.Trainer(
            default_root_dir=os.path.join(self.checkpoint_dir, save_name),
            accelerator="gpu",
            devices=[0],
            max_epochs=200,
            callbacks= [checkpoint_callback],
            check_val_every_n_epoch=5,
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
        result = {"test": test_result[0]["test_loss"],
                  "val": val_result[0]["test_loss"]}

        logger.experiment.finish()

        return model, result

    def scan_hyperparams(self):
        for num_layers, num_conv_layers, kernel_size, dropout_rate, momentum, lr, batch_size, zeta in product(
                                                 [2,3,6],
                                                 [2,3,5],
                                                 [3,5,7],
                                                 [0.0, 0.1, 0.2],
                                                 [0.5, 0.8],
                                                 [1e-2, 1e-3],
                                                 [512, 128],
                                                [0.1, 1]):
            optimizer = "SGD"

            # model_config = {"num_layers": num_layers,
            #                 "activation": activation,
            #                 "norm": False,
            #                 "input_size": self.input_size,
            #                 "hidden_size": self.input_size,
            #                 "output_size": self.output_size}
            model_config = {"num_layers": num_layers,
                            "num_conv_layers": num_conv_layers,
                            "kernel_size": kernel_size,
                            "dropout_rate": dropout_rate,
                            "activation": "LeakyReLU",
                            "norm": True,
                            "input_size": self.input_size,
                            "hidden_size": self.output_size,
                            "output_size": self.output_size}
            optimizer_config = {"lr": lr,
                                "momentum": momentum,}
            loss_config = {"loss_name": "mse",
                            "zeta": zeta}
            if optimizer == "Adam":
                optimizer_config = {"lr": lr}
            misc_config = {"batch_size": batch_size}
            self.set_dataloaders(batch_size=batch_size)

            self.train_model(model_name="WideCNN",
                             model_hparams=model_config,
                             optimizer_name=optimizer,
                             optimizer_hparams=optimizer_config,
                             misc_hparams=misc_config,
                             loss_hparams=loss_config)

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

    def scan_linear_hyperparams(self):
        for optimizer, num_layers, hidden_size, Phi_sign in product(["SGD", "Adam"],
                                                    [2, 3],
                                                    [self.input_size, 2*self.input_size, 3*self.input_size],
                                                    [True, False]):

            model_config = {"num_layers": num_layers,
                            "norm": False,
                            "input_size": self.input_size,
                            "output_size": self.output_size,
                            "hidden_size": hidden_size,
                            "Phi_sign": Phi_sign,}
            optimizer_config = {"lr": 1e-2,
                                "momentum": 0.9,}
            if optimizer == "Adam":
                optimizer_config= {"lr": 1e-2}
            misc_config = {"batch_size": self.batch_size}

            self.train_linear_model(model_name="LinearNet",
                             model_hparams=model_config,
                             optimizer_name=optimizer,
                             optimizer_hparams=optimizer_config,
                             misc_hparams=misc_config)

    def load_model(self, model_name="WideCNN", model_id="5nozki8z"):
        # Check whether pretrained model exists. If yes, load it and skip training
        print(self.checkpoint_dir)
        pretrained_filename = os.path.join(self.checkpoint_dir, model_name, "triple_correlation", model_id,
                                           "checkpoints", "*" + ".ckpt")
        print(pretrained_filename)
        if os.path.isfile(glob.glob(pretrained_filename)[2]):
            pretrained_filename = glob.glob(pretrained_filename)[2]
            print(
                f"Found pretrained model at {pretrained_filename}, loading...")
            # Automatically loads the model with the saved hyperparameters
            model = ClosurePhaseDecoder.load_from_checkpoint(pretrained_filename)

            return model

    def plot_predictions(self, model_name="WideCNN", model_id="5nozki8z"):

        model = self.load_model(model_name=model_name, model_id=model_id)
        trainer = L.Trainer(
            accelerator="gpu",
            devices=[0]
        )
        y = trainer.predict(model, dataloaders=self.test_loader)

        print(y[0][0].numpy().shape)
        print(y[0][2].numpy().shape)
        # y[batch_idx][return_idx], return_idx 0...3: 0: Predictions, 1: Targets, 2: inputs, 3: encoded
        print("MSE Loss: ", np.mean((y[0][0].numpy() - y[0][1].numpy())**2))

        for i in range(len(y[0][0].numpy()[:,0])):
            fig = plt.figure(figsize=(15, 5))
            ax1, ax2, ax3 = fig.subplots(1, 3)

            ax1.imshow(y[0][2].numpy()[i,:,:], origin="lower", vmin=-1, vmax=1)
            ax1.set_title("Inputs")

            ax2.plot(y[0][0].numpy()[i,:], label="Predictions")
            ax2.plot(y[0][1].numpy()[i,:], label="Targets")
            ax2.set_title("MSE Loss: " + str(nn.MSELoss(reduction='sum')(y[0][0][i,:], y[0][1][i,:]).item()))
            ax2.legend()

            ax3.imshow(y[0][3].numpy()[i,:,:], origin="lower", vmin=-1, vmax=1)
            ax3.set_title("Encoded Prediction, MSE Loss: " + str(nn.MSELoss(reduction='sum')(y[0][3][i,:], y[0][2][i,:]).item()))

            plt.tight_layout()
            plt.show()
