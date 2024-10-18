#!/usr/bin/env python

import os, glob
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
                      "BottleCNN": BottleCNN, "MLP": MLP}
        
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
        loss = nn.MSELoss()(y_hat, y)
        
        return loss

    @staticmethod
    def antisymmetry_loss(outputs):
        # Punish phases that are not antisymmetric about the origin

        positive_x = outputs[:, outputs.size(1) // 2 + 1:]
        negative_x = torch.flip(outputs[:, :outputs.size(1) // 2], [1])

        loss = torch.sum(torch.add(positive_x, negative_x)**2) + torch.sum(
            outputs[:, outputs.size(1) // 2]**2)
        return loss

    def encoding_loss(self, phase, absPhi):
        # Punish phases that cannot be used to reconstruct the input abs(Phi)

        encoded = torch.abs(self.encode(phase))

        return nn.MSELoss()(encoded, torch.abs(absPhi))

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

            # TODO: don't save cos(Phi)
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
        # TODO: Do we need to return y_hat, y, loss?
        # return y_hat, y, loss

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
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        
        return loss
    
class PhaseRegressor(BaseDecoder):
    def __init__(self, model_name,model_hparams, optimizer_name,
                 optimizer_hparams, misc_hparams, loss_hparams=None):
        super().__init__(model_name, model_hparams, optimizer_name,
                         optimizer_hparams, misc_hparams, loss_hparams)
    
    def loss_function(self, y_hat, y, x):
        # Choose the loss function
        #if self.loss_hparams["loss_name"] == "mse":
        loss = nn.MSELoss()(y_hat, y)
            
        if self.loss_hparams['zeta'] > 0:
            zeta = self.loss_hparams['zeta']
            # Want x to be abs(Phi) here, so that all sign prediction is handled
            # by the sign classifier, and the PhaseRegressor is only concerned
            # with the magnitude of the phase
            loss += zeta * self.encoding_loss(y_hat, torch.abs(x))
        
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
        self.sign_classifier = SignClassifier(model_name, model_hparams,
                                              optimizer_name,
                                              optimizer_hparams, misc_hparams,
                                              loss_hparams)
        self.phase_regressor = self.load_regressor(model_name=model_name,
                                                   model_id="5nozki8z")
        
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
        
    @staticmethod
    def load_regressor(model_name, model_id):
        # Check whether pretrained model exists. If yes, load it and skip
        # training
        checkpoint_dir = "./checkpoints"
        pretrained_filename = os.path.join(
            checkpoint_dir,
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
            model = PhaseRegressor.load_from_checkpoint(
                pretrained_filename)

            return model
        
    def loss_function(self, phase, phase_hat, Phi_binary, Phi_binary_prelogit,
                      abs_Phi):
        bce_loss = nn.BCEWithLogitsLoss()(Phi_binary_prelogit, Phi_binary)
        mse_loss = nn.MSELoss()(phase_hat, phase)
        encoding_loss = self.encoding_loss(phase_hat, abs_Phi)
        
        loss = bce_loss
        
        if self.loss_hparams['zeta'] > 0:
            zeta = self.loss_hparams['zeta']
            loss += zeta * encoding_loss
            
        if self.loss_hparams['alpha'] > 0:
            alpha = self.loss_hparams['alpha']
            loss += alpha * mse_loss
    
        return loss, bce_loss, mse_loss, encoding_loss
    
    def forward(self, x):
        pre_logit = self.sign_classifier(x)
        sign_prob = nn.Sigmoid()(pre_logit)
        return sign_prob
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward, which is only for inference
        abs_Phi, Phi_binary, phase = batch
        prelogit = self.sign_classifier(abs_Phi)
        sign_prob = nn.Sigmoid()(prelogit)
        sign = (sign_prob > 0.5).float() * 2 - 1  # Convert to {-1, 1}
        
        Phi = sign * abs_Phi
        phase_hat = self.phase_regressor(Phi)
        
        loss, bce_loss, mse_loss, encoding_loss = self.loss_function(phase,
                                                                     phase_hat,
                                                                     Phi_binary,
                                                                     prelogit,
                                                                     abs_Phi)
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_bce_loss", bce_loss, prog_bar=False,
                 on_epoch=True)
        self.log("train_mse_loss", mse_loss, prog_bar=False,
                 on_epoch=True)
        self.log("train_encoding_loss", encoding_loss, prog_bar=False,
                 on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        abs_Phi, Phi_binary, phase = batch
        prelogit = self.sign_classifier(abs_Phi)
        sign_prob = nn.Sigmoid()(prelogit)
        sign = (sign_prob > 0.5).float() * 2 - 1  # Convert to {-1, 1}
        
        Phi = sign * abs_Phi
        phase_hat = self.phase_regressor(Phi)
        
        loss, bce_loss, mse_loss, encoding_loss = self.loss_function(phase,
                                                                     phase_hat,
                                                                     Phi_binary,
                                                                     prelogit,
                                                                     abs_Phi)
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_bce_loss", bce_loss, prog_bar=False, on_epoch=True)
        self.log("val_mse_loss", mse_loss, prog_bar=False, on_epoch=True)
        self.log("val_encoding_loss", encoding_loss, prog_bar=False,
                 on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        abs_Phi, Phi_binary, phase = batch
        prelogit = self.sign_classifier(abs_Phi)
        sign_prob = nn.Sigmoid()(prelogit)
        sign = (sign_prob > 0.5).float() * 2 - 1  # Convert to {-1, 1}
        
        Phi = sign * abs_Phi
        phase_hat = self.phase_regressor(Phi)
        
        loss, bce_loss, mse_loss, encoding_loss = self.loss_function(phase,
                                                                     phase_hat,
                                                                     Phi_binary,
                                                                     prelogit,
                                                                     abs_Phi)
        
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_bce_loss", bce_loss, prog_bar=False, on_epoch=True)
        self.log("test_mse_loss", mse_loss, prog_bar=False, on_epoch=True)
        self.log("test_encoding_loss", encoding_loss, prog_bar=False,
                 on_epoch=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass
    
    
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
