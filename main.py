#!/usr/bin/env python

import numpy as np
import sys
import torch

import models
import plot1d
#import plot2d
#import PaperFigures
import timeit

import speckle1d
#import speckle2d
import triphase1d
#import triphase2d
import training
import decoder
import models
import datasets

if __name__ == '__main__':
    np.random.seed()
    if len(sys.argv) == 1:
        """Run functions in this scratch area. 
        """
        num_pix = 11
        train_samples = int(1e3)
        valid_samples = int(1e2)
        train_file = f"./data/pretrain_numpix{num_pix}_{train_samples:.0e}_samples.h5"
        valid_file = f"./data/prevalid_numpix{num_pix}_{valid_samples:.0e}_samples.h5"
        test_file = f"./data/pretest_numpix{num_pix}_{valid_samples:.0e}_samples.h5"
        # datasets.generate_pretraining_data(
        #     num_pix=num_pix, num_samples=int(train_samples),
        #     file_path=train_file)
        #
        # datasets.generate_pretraining_data(
        #     num_pix=num_pix, num_samples=int(valid_samples),
        #     file_path=valid_file)
        #
        # datasets.generate_pretraining_data(
        #     num_pix=num_pix, num_samples=int(valid_samples),
        #     file_path=test_file)

        runner = training.Trainer(train_file, valid_file, test_file,
                                  absPhi=True, signPhi=True, multiTask=True)
        runner.scan_hyperparams()
        
        # runner.plot_phase_predictions(model_name="ImplicitMultiMLP",
        #                             model_id="9uzn9wi7")
        
        # Best hybrid classifier so far
        # runner.plot_phase_predictions(model_name="MLP",
        #                              model_id="eojqvv3v")
        #runner.plot_sign_predictions(model_name="MLP", model_id="eojqvv3v")


    else:
        raise NotImplementedError("Unsupported number of command-line "
                                  "arguments")