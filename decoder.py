#!/usr/bin/env python

from torch import optim, nn
import torch
import lightning as L
from models import PhaseMLP, LinearNet, BottleCNN, MLP
from tqdm import tqdm
import utils
import numpy as np
from speckle1d import Fluorescence1D


class BaseDecoder(L.LightningModule):
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
        # Exports the hyperparameters to a YAML file, and create "self.hparams"
        # namespace
        self.save_hyperparameters()
        # Create model
        self.model_name = model_name
        self.model = self.create_model(model_name, model_hparams)
        self.loss_hparams = loss_hparams

        torch.set_float32_matmul_precision('high')

    @staticmethod
    def create_model(model_name, model_hparams):
        model_dict = {"PhaseMLP": PhaseMLP, "LinearNet": LinearNet,
                      "BottleCNN": BottleCNN, "SignMLP": MLP}
        
        if model_name in model_dict:
            return model_dict[model_name](**model_hparams)
        else:
            assert False, f'Unknown model name "{
                model_name}". Available models are: {str(model_dict.keys())}'

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay
            # (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(),
                                    **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(),
                                  **self.hparams.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        # We will reduce the learning rate by factor gamma at each milestone
        # (epoch number). Setting gamma to 1.0 has no effect on learning rate.
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[150, 500],
                                                   gamma=1)
        return [optimizer], [scheduler]

    def loss_function(self, y_hat, y, x):
        # Choose the loss function
        #if self.loss_hparams["loss_name"] == "mse":
        loss = nn.MSELoss(y_hat, y, reduction='mean')
        
        return loss

    @staticmethod
    def antisymmetry_loss(outputs):
        # Punish phases that are not antisymmetric about the origin

        positive_x = outputs[:, outputs.size(1) // 2 + 1:]
        negative_x = torch.flip(outputs[:, :outputs.size(1) // 2], [1])

        loss = torch.sum(torch.add(positive_x, negative_x)**2) + torch.sum(
            outputs[:, outputs.size(1) // 2]**2)
        return loss

    def encoding_loss(self, outputs, inputs):
        # Punish phases that cannot be used to reconstruct the input cosPhi

        # encoded = torch.cos(self.encode(outputs))
        encoded = self.encode(outputs)

        return self.loss_function(encoded, inputs)

    @staticmethod
    def encode(outputs):
        # compute the encoded version of the outputs, the signed version of Phi

        phase_positive = outputs[:, outputs.size(1) // 2:]
        phase_negative = torch.flip(outputs[:, :outputs.size(1) // 2 + 1], [1])
        phase = (phase_positive - phase_negative) / 2

        # Assuming phase is a batch input of phases as a PyTorch tensor
        # Shape of phase: (batch_size, phase_length)
        batch_size, phase_length = phase.shape

        # Calculate the phase difference array for each batch element
        Phi = torch.zeros(
            (batch_size, phase_length, phase_length)).type_as(outputs)
        for n in range(phase_length):
            Phi[:, n, :] = (torch.roll(
                phase[:, :], -n, dims=-1) - phase - phase[:, n].unsqueeze(-1))

        Phi = Phi[:, :phase_length // 2 + 1, :phase_length // 2 + 1]

        return Phi

    @staticmethod
    def generate_pretraining_data(num_pix, num_samples, file_path):
        # Generate pretraining data
        # num_pix: number of pixels in each sample

        for _ in tqdm(range(num_samples)):
            phase = np.random.uniform(-np.pi, np.pi, num_pix // 2)
            phase = np.concatenate((-phase, np.zeros(1), np.flip(phase)))
            
            # These lines work in PyTorch
            # phase = torch.FloatTensor(phase)
            # Phi = ClosurePhaseDecoder.encode(phase.unsqueeze(0))
            
            # Use this Numpy module for JIT compilation speeds and compatibility
            # with h5py concatenation
            Phi = Fluorescence1D.compute_Phi_from_phase(phase[num_pix // 2:])

            utils.append_to_h5file(np.cos(Phi), Phi, phase, file_path)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        y_hat = self.model(x)
        
        loss = self.loss_function(y_hat, y, x)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # validation_step defines the validation loop.
        x, y = batch
        y_hat = self.model(x)
        
        loss = self.loss_function(y_hat, y, x)
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        
        loss = self.loss_function(y_hat, y, x)
        
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        return y_hat, y, loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        
        # TODO: change output shapes
        if self.model_name == "SignMLP":
            print(y_hat.shape)
            y_hat = nn.Sigmoid()(y_hat)
            return y_hat.view(-1, x.size(1), x.size(2)), y.view(-1, x.size(1), x.size(2)), x
        else:
            encoded = self.encode(y_hat)
            return y_hat, y, x, encoded
        
class SignClassifier(BaseDecoder):
    def __init__(self, model_name, model_hparams, optimizer_name,
                 optimizer_hparams, misc_hparams, loss_hparams=None):
        super().__init__(model_name, model_hparams, optimizer_name,
                         optimizer_hparams, misc_hparams, loss_hparams)
    
    def loss_function(self, y_hat, y, x):
        # Choose the loss function
        #if self.loss_hparams["loss_name"] == "bce_logits":
        loss = nn.BCEWithLogitsLoss(y_hat, y)
        
        return loss
    
class PhaseRegressor(BaseDecoder):
    def __init__(self, model_name,model_hparams, optimizer_name,
                 optimizer_hparams, misc_hparams, loss_hparams=None):
        super().__init__(model_name, model_hparams, optimizer_name,
                         optimizer_hparams, misc_hparams, loss_hparams)
    
    def loss_function(self, y_hat, y, x):
        # Choose the loss function
        #if self.loss_hparams["loss_name"] == "mse":
        loss = nn.MSELoss(y_hat, y, reduction='mean')
            
        if self.loss_hparams['zeta'] > 0:
            zeta = self.loss_hparams['zeta']
            loss += zeta * self.encoding_loss(y_hat, x)
        
        return loss
        
    def forward(self, x):
        phase = self.model(x)
        y_hat = torch.atan2(torch.sin(phase), torch.cos(phase))
        return y_hat
    
    
class HybridClassifier(L.LightningModule):
    # This is a hybrid model that combines the sign classifier and a pretrained
    # linear phase regressor in order to train the sign classifier on the
    # phase regression results
    def __init__(self, model_name, model_hparams, optimizer_name,
                 optimizer_hparams, misc_hparams, loss_hparams=None):
        super().__init__()
        self.sign_classifier = SignClassifier(model_name, model_hparams, optimizer_name, optimizer_hparams, misc_hparams, loss_hparams)
        self.phase_regressor = PhaseRegressor(model_name, model_hparams, optimizer_name, optimizer_hparams, misc_hparams, loss_hparams)
        
        # Exports the hyperparameters to a YAML file, and create "self.hparams"
        # namespace
        self.save_hyperparameters()
        # Create model
        self.model_name = model_name
        self.loss_hparams = loss_hparams
        
        # Freeze the regressor parameters
        for param in self.phase_regressor.parameters():
            param.requires_grad = False
        
        torch.set_float32_matmul_precision('high')
        
    def loss_function(self, y_hat, y, x):
        bce_loss = nn.BCEWithLogitsLoss(y_hat, y)
    
    def forward(self, x):
        # forward defines the prediction/inference actions, not the training loop
        # x should be abs(Phi), but don't put abs in forward(x)
        pre_logit = self.sign_classifier(x)
        sign_prob = nn.Sigmoid()(pre_logit)
        return sign_prob
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        pre_logit = self.sign_classifier(x)
        
        loss = self.loss_function(y_hat, y, x)
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    
class MultiTaskRegressor(L.LightningModule):
    # This is a multi-task model that combines the sign classifier and an
    # untrained non-linear regressor to determine if end-to-end training gives
    # better results than sign classification plus exact linear solver in hybrid
    # model alone. Might deal with noise better?
    def __init__(self, model_name, model_hparams, optimizer_name,
                 optimizer_hparams, misc_hparams, loss_hparams=None):
        super().__init__()
        self.sign_classifier = SignClassifier(model_name, model_hparams, optimizer_name, optimizer_hparams, misc_hparams, loss_hparams)
        
    def forward(self, x):
        # x should be abs(Phi), but don't put abs in forward(x)
        sign_prob = self.sign_classifier(x)
        sign = (sign_prob > 0.5).float() * 2 - 1  # Convert to {-1, 1}
        Phi = torch.dot(sign, x) # signed version of Phi
        phase = self.phase_regressor(Phi)
        y_hat = torch.atan2(torch.sin(phase), torch.cos(phase))
        return y_hat, sign_prob
