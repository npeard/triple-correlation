#!/usr/bin/env python

from torch import optim, nn
import torch
import lightning as L

# define the LightningModule
class ClosurePhaseDecoder(L.LightningModule):
    def __init__(self, model, kappa=0., zeta=0., learning_rate=1e-3, linear_only=False):
        super().__init__()
        self.kappa = kappa
        self.zeta = zeta
        self.learning_rate = learning_rate
        self.model = model
        self.linear_only = linear_only
        self.save_hyperparameters()
        torch.set_float32_matmul_precision('medium')

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
        x = x.view(-1, x.size(1)**2)
        y_hat = self.model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        if self.kappa > 0:
            loss += self.kappa * self.antisymmetry_loss(y_hat)
        if self.zeta > 0:
            loss = self.zeta * self.encoding_loss(y_hat, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        x, y = batch
        x = x.view(-1, x.size(1)**2)
        y_hat = self.model(x)
        # to make models comparable from check point, don't multiply hyperparameters
        # into the loss function, but compute the total loss with maximal
        # hyperparameters for every model
        loss = nn.functional.mse_loss(y_hat, y)
        if self.kappa > 0:
            loss += self.kappa * self.antisymmetry_loss(y_hat)
        if self.zeta > 0:
            loss = self.zeta * self.encoding_loss(y_hat, x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x = x.view(-1, x.size(1)**2)
        y_hat = self.model(x)
        return y_hat, y
