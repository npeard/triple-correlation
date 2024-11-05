#!/usr/bin/env python

import os
import glob
from torch import nn
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from decoder import BaseDecoder, SignClassifier, PhaseRegressor, HybridClassifier
import numpy as np
import matplotlib.pyplot as plt
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from itertools import product
from datasets import get_custom_dataloader


class Trainer:
    def __init__(self, training_h5, validation_h5, testing_h5,
                 absPhi=False, signPhi=False, multiTask=False):
        self.training_h5 = training_h5
        self.validation_h5 = validation_h5
        self.testing_h5 = testing_h5
        self.absPhi = absPhi
        self.signPhi = signPhi
        self.multiTask = multiTask

        # get dataloaders
        self.set_dataloaders()

        # dimensions
        self.input_size = next(iter(self.train_loader))[0].size(-1) ** 2
        self.output_size = next(iter(self.train_loader))[1].size(-1)
        if signPhi:
            self.output_size = self.output_size**2

        # directories
        self.checkpoint_dir = "./checkpoints"

    def set_dataloaders(self, batch_size=128):
        self.batch_size = batch_size
        self.train_loader = get_custom_dataloader(
            self.training_h5, batch_size=self.batch_size,
            absPhi=self.absPhi,
            signPhi=self.signPhi,
            multiTask=self.multiTask,
            shuffle=True)
        verify_loader = get_custom_dataloader(
            self.training_h5, batch_size=self.batch_size,
            absPhi=self.absPhi,
            signPhi=self.signPhi,
            multiTask=self.multiTask,
            shuffle=False)
        self.valid_loader = get_custom_dataloader(
            self.validation_h5,
            batch_size=self.batch_size,
            absPhi=self.absPhi,
            signPhi=self.signPhi,
            multiTask=self.multiTask,
            shuffle=False)
        self.test_loader = get_custom_dataloader(
            self.testing_h5,
            batch_size=self.batch_size,
            absPhi=self.absPhi,
            signPhi=self.signPhi,
            multiTask=self.multiTask,
            shuffle=False)

        verify_batch = next(iter(verify_loader))
        valid_batch = next(iter(self.valid_loader))
        test_batch = next(iter(self.test_loader))
        # Check for independence of the data sets by checking for equality of the first batch (unshuffled)
        assert (verify_batch[:] == valid_batch[:]).all(), "Training and validation sets are not independent"
        assert (verify_batch[:] == test_batch[:]).all(), "Training and test sets are not independent"
        assert (valid_batch[:] == test_batch[:]).all(), "Validation and test sets are not independent"

    def train_model(self, model_name, task_name, save_name=None, **kwargs):
        """Train model.

        Args:
            model_name: Name of the model you want to run. Is used to look up
            the class in "model_dict"
            save_name (optional): If specified, this name will be used for
            creating the checkpoint and logging directory.
        """
        if save_name is None:
            save_name = model_name

        # logger
        logger = None
        # logger = WandbLogger(
        #     project='triple_correlation',
        #     group=model_name,
        #     log_model=True,
        #     save_dir=os.path.join(
        #         self.checkpoint_dir,
        #         save_name))

        # callbacks
        # early stopping
        early_stop_callback = EarlyStopping(monitor="val_loss",
                                            min_delta=0.005,
                                            patience=7,
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
            max_epochs=1000,
            callbacks=[checkpoint_callback, early_stop_callback],
            #callbacks=[checkpoint_callback],
            check_val_every_n_epoch=10,
            logger=logger
        )
        task_dict = {
            "sign_classification": SignClassifier,
            "phase_regression": PhaseRegressor,
            "hybrid_classification": HybridClassifier
        }
        # L.seed_everything(42)  # To be reproducible
        model = task_dict[task_name](model_name=model_name, **kwargs)
        trainer.fit(model, self.train_loader, self.valid_loader)

        # Load best checkpoint after training
        model = task_dict[task_name].load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)

        # Run test set through best model and log metrics
        trainer.test(model, dataloaders=self.test_loader,
                                   verbose=False)

        logger.experiment.finish()

        return model

    def scan_hyperparams(self):
        for (num_layers, num_conv_layers, kernel_size, dropout_rate, momentum,
             lr, batch_size, zeta, norm, hidden_size) in product(
                [3], [None], [None], [0.0], [0.9], [1e-3], [128],
                [0], [True], [2*self.input_size]):
            optimizer = "Adam"

            model_config = {"num_layers": num_layers,
                            "activation": "LeakyReLU",
                            "norm": norm,
                            "input_size": self.input_size,
                            "hidden_size": hidden_size,
                            "output_size": self.output_size
                            }
            optimizer_config = {"lr": lr,
                                "momentum": momentum, }
            loss_config = {"loss_name": "mse",
                           "zeta": zeta,}
                           #"alpha": np.log(2)/np.pi}
            if optimizer == "Adam":
                optimizer_config = {"lr": lr}
            misc_config = {"batch_size": batch_size}
            self.set_dataloaders(batch_size=batch_size)

            self.train_model(model_name="MLP",
                             task_name="sign_classification",
                             model_hparams=model_config,
                             optimizer_name=optimizer,
                             optimizer_hparams=optimizer_config,
                             misc_hparams=misc_config,
                             loss_hparams=loss_config)

    def load_model(self, model_name="BottleCNN", model_id="5nozki8z"):
        # Check whether pretrained model exists. If yes, load it and skip
        # training
        print(self.checkpoint_dir)
        pretrained_filename = os.path.join(
            self.checkpoint_dir,
            model_name,
            "triple_correlation",
            model_id,
            "checkpoints",
            "*" + ".ckpt")
        pretrained_filename = glob.glob(pretrained_filename)[0]
        if os.path.isfile(pretrained_filename):
            print(f"Found pretrained model at {
                  pretrained_filename}, loading...")
            # Automatically loads the model with the saved hyperparameters
            # TODO: add conditional loading based on task name here
            model = PhaseRegressor.load_from_checkpoint(
                pretrained_filename)

            return model

    def plot_phase_predictions(self, model_name="BottleCNN",
                              model_id="i52c3rlz"):

        model = self.load_model(model_name=model_name, model_id=model_id)
        model.task_name = "phase_regression"
        trainer = L.Trainer(
            accelerator="cpu",
            # devices=[0]
        )
        y = trainer.predict(model, dataloaders=self.test_loader)

        print(y[0][0].numpy().shape)
        print(y[0][2].numpy().shape)
        # y[batch_idx][return_idx], return_idx 0...3: 0: Predictions, 1:
        # Targets, 2: inputs, 3: encoded
        print("MSE Loss: ", np.mean((y[0][0].numpy() - y[0][1].numpy())**2))

        for i in range(len(y[0][0].numpy()[:, 0])):
            fig = plt.figure(figsize=(15, 5))
            ax1, ax2, ax3 = fig.subplots(1, 3)

            im1 = ax1.imshow(y[0][2].numpy()[i, :, :], origin="lower")
            ax1.set_title("Inputs")
            plt.colorbar(im1, ax=ax1)

            ax2.plot(y[0][1].numpy()[i, :], label="Targets")
            ax2.plot(y[0][0].numpy()[i, :], label="Predictions")
            ax2.set_title("MSE Loss: " + str(nn.MSELoss(reduction='sum')
                          (y[0][0][i, :], y[0][1][i, :]).item()))
            ax2.legend()

            im3 = ax3.imshow(y[0][3].numpy()[i, :, :], origin="lower")
            ax3.set_title("Encoded Prediction, MSE Loss: " +
                          str(nn.MSELoss(reduction='sum')(y[0][3][i, :],
                                                          y[0][2][i, :]).item())
                          )
            plt.colorbar(im3, ax=ax3)

            plt.tight_layout()
            plt.show()

    def plot_sign_predictions(self, model_name="MLP",
                              model_id="i52c3rlz"):

        model = self.load_model(model_name=model_name, model_id=model_id)
        model.task_name = "sign_classification"
        trainer = L.Trainer(
            accelerator="cpu",
            # devices=[0]
        )
        y = trainer.predict(model, dataloaders=self.test_loader)

        print(y[0][0].numpy().shape)
        print(y[0][2].numpy().shape)
        # y[batch_idx][return_idx], return_idx 0...2: 0: Predictions, 1:
        # Targets, 2: inputs
        print("MSE Loss: ", np.mean((y[0][0].numpy() - y[0][1].numpy())**2))

        for i in range(len(y[0][0].numpy()[:, 0])):
            fig = plt.figure(figsize=(10, 5))
            ax1, ax2, ax3 = fig.subplots(1, 3)

            im1 = ax1.imshow(y[0][2].numpy()[i, :, :], origin="lower")
            ax1.set_title("Inputs")
            plt.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(2*y[0][1].numpy()[i, :, :]-1, origin="lower", vmin=-1,
                             vmax=1, cmap="coolwarm")
            ax2.set_title("Targets [-1, 1]")
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_ticks([-1, 0, 1])

            im3 = ax3.imshow(2*y[0][0].numpy()[i, :, :]-1, origin="lower", vmin=-1,
                             vmax=1, cmap="coolwarm")
            ax3.set_title("Predictions [-1, 1]")
            cbar3 = plt.colorbar(im3, ax=ax3)
            cbar3.set_ticks([-1, 0, 1])

            plt.tight_layout()
            plt.show()
