#!/usr/bin/env python

from torch import optim, nn
import torch
import lightning as L
from models import SequentialNN, LinearNet, WideCNN

model_dict = {"SequentialNN": SequentialNN, "LinearNet": LinearNet, "WideCNN": WideCNN}


class ClosurePhaseDecoder(L.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name,
                 optimizer_hparams, misc_hparams):
        """Decoder for the closure phase

        Args:
            model_name: Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams: Hyperparameters for the model, as dictionary.
            optimizer_name: Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = self.create_model(model_name, model_hparams)
        # Create loss module
        self.loss_function = nn.MSELoss()

        torch.set_float32_matmul_precision('medium')

    def create_model(self, model_name, model_hparams):
        if model_name in model_dict:
            self.model_name = model_name
            return model_dict[model_name](**model_hparams)
        else:
            assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(),
                                    **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(),
                                  **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[100, 150],
                                                   gamma=0.1)
        return [optimizer], [scheduler]

    def get_loss_function(self, loss_hparams):
        if loss_hparams["loss_name"] == "mse":
            self.loss_function = nn.MSELoss()
        else:
            assert False, f'Unknown loss: "{loss_hparams["loss_name"]}"'

        # if loss_hparams["kappa"] > 0:
        #     self.loss_function += loss_hparams["kappa"] * self.antisymmetry_loss()

    def antisymmetry_loss(self, outputs):
        # Punish phases that are not antisymmetric about the origin

        positive_x = outputs[:, outputs.size(1) // 2 + 1:]
        negative_x = torch.flip(outputs[:, :outputs.size(1) // 2], [1])

        loss = torch.sum(torch.add(positive_x, negative_x)**2) + torch.sum(
            outputs[:, outputs.size(1) // 2]**2)
        return loss

    def encoding_loss(self, outputs, inputs):
        # Punish phases that cannot be used to reconstruct the input cosPhi

        phase = outputs[:, outputs.size(1) // 2:]

        # Assuming phase is a batch input of phases as a PyTorch tensor
        # Shape of phase: (batch_size, phase_length)
        batch_size, phase_length = phase.shape

        # Calculate the phase difference array for each batch element
        Phi = torch.zeros((batch_size, phase_length, phase_length)).type_as(outputs)
        for n in range(phase_length):
            Phi[:, n, :] = torch.abs(
                torch.roll(phase[:,:], -n, dims=-1) - phase - phase[:, n].unsqueeze(-1))

        Phi = Phi[:, :phase_length // 2 + 1, :phase_length // 2 + 1]
        Phi = Phi.reshape(batch_size, Phi.size(1)**2)

        if self.linear_only:
            return nn.functional.mse_loss(Phi, inputs)
        else:
            return nn.functional.mse_loss(torch.cos(Phi), inputs)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        if self.model_name == "WideCNN":
            x = x.view(-1, 1, x.size(1), x.size(2))
        else:
            x = x.view(-1, x.size(1)**2)
        preds = self.model(x)
        loss = self.loss_function(preds, y)
        acc = (preds == y).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        x, y = batch
        if self.model_name == "WideCNN":
            x = x.view(-1, 1, x.size(1), x.size(2))
        else:
            x = x.view(-1, x.size(1) ** 2)
        preds = self.model(x)
        loss = self.loss_function(preds, y)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    # def configure_optimizers(self):
    #     # optimizer = optim.AdamW(self.parameters(), lr=self.lr)
    #     optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
    #     return optimizer
    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.model_name == "WideCNN":
            x = x.view(-1, 1, x.size(1), x.size(2))
        else:
            x = x.view(-1, x.size(1) ** 2)
        preds = self.model(x)
        loss = self.loss_function(preds, y)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.log("test_loss", loss, prog_bar=True)
        return y, preds, loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        if self.model_name == "WideCNN":
            x = x.view(-1, 1, x.size(1), x.size(2))
        else:
            x = x.view(-1, x.size(1) ** 2)
        y_hat = self.model(x)
        return y_hat, y