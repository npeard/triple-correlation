#!/usr/bin/env python

from torch import optim, nn
import torch
import lightning as L
from models import MLP, LinearNet, BottleCNN

model_dict = {"MLP": MLP, "LinearNet": LinearNet, "BottleCNN": BottleCNN}


class ClosurePhaseDecoder(L.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name,
                 optimizer_hparams, misc_hparams, loss_hparams=None):
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
        if loss_hparams is None:
            self.loss_function = nn.MSELoss(reduction='sum')
        else:
            self.loss_function = self.get_loss_function(loss_hparams)

        torch.set_float32_matmul_precision('high')

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

        # We will reduce the learning rate by factor gamma at each milestone (epoch number)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[50, 100],
                                                   gamma=0.1)
        return [optimizer], [scheduler]

    def get_loss_function(self, loss_hparams):
        # Choose the loss function
        if loss_hparams["loss_name"] == "mse":
            self.loss_function = nn.MSELoss(reduction='mean')
        else:
            assert False, f'Unknown loss: "{loss_hparams["loss_name"]}"'

        if loss_hparams["zeta"] > 0:
            self.zeta = loss_hparams["zeta"]
        else:
            self.zeta = None

        return self.loss_function

    def antisymmetry_loss(self, outputs):
        # Punish phases that are not antisymmetric about the origin

        positive_x = outputs[:, outputs.size(1) // 2 + 1:]
        negative_x = torch.flip(outputs[:, :outputs.size(1) // 2], [1])

        loss = torch.sum(torch.add(positive_x, negative_x)**2) + torch.sum(
            outputs[:, outputs.size(1) // 2]**2)
        return loss

    def encoding_loss(self, outputs, inputs):
        # Punish phases that cannot be used to reconstruct the input cosPhi

        encoded = self.encode(outputs)

        return self.loss_function(encoded, inputs)

    def encode(self, outputs):
        # compute the encoded version of the outputs

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
        # Phi = Phi.reshape(batch_size, Phi.size(1)**2)

        # if self.linear_only:
        #     return Phi
        # else:
        return torch.cos(Phi)


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        if self.model_name == "BottleCNN":
            # batch dim, channel dim, height dim, width dim
            x_view = x.view(-1, 1, x.size(1), x.size(2))
        else:
            x_view = x.view(-1, x.size(1)**2)
        preds = self.model(x_view)

        # compute encoding loss
        if self.zeta is not None:
            encoded = self.encode(preds)
            # x = torch.squeeze(x)
            loss = self.loss_function(preds, y) + self.zeta * self.loss_function(encoded, x)
        else:
            loss = self.loss_function(preds, y)

        # acc = (preds == y).float().mean()
        # self.log("train_acc", acc, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        x, y = batch
        if self.model_name == "BottleCNN":
            # batch dim, channel dim, height dim, width dim
            x_view = x.view(-1, 1, x.size(1), x.size(2))
        else:
            x_view = x.view(-1, x.size(1) ** 2)
        preds = self.model(x_view)

        # compute encoding loss
        if self.zeta is not None:
            encoded = self.encode(preds)
            # x = torch.squeeze(x)
            loss = self.loss_function(preds, y) + self.zeta * self.loss_function(encoded, x)
        else:
            loss = self.loss_function(preds, y)

        # acc = (preds == y).float().mean()
        # self.log("val_acc", acc, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.model_name == "BottleCNN":
            # batch dim, channel dim, height dim, width dim
            x_view = x.view(-1, 1, x.size(1), x.size(2))
        else:
            x_view = x.view(-1, x.size(1) ** 2)
        preds = self.model(x_view)

        # compute encoding loss
        if self.zeta is not None:
            encoded = self.encode(preds)
            # x = torch.squeeze(x)
            loss = self.loss_function(preds, y) + self.zeta * self.loss_function(encoded, x)
        else:
            loss = self.loss_function(preds, y)

        # acc = (preds == y).float().mean()
        # self.log("test_acc", acc, on_epoch=True)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        return y, preds, loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        if self.model_name == "BottleCNN":
            x_view = x.view(-1, 1, x.size(1), x.size(2))
        else:
            x_view = x.view(-1, x.size(1) ** 2)

        y_hat = self.model(x_view)
        encoded = self.encode(y_hat)
        return y_hat, y, x, encoded