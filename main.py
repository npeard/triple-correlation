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

if __name__ == '__main__':
    np.random.seed(0x5EED+7)
    if len(sys.argv) == 1:
        """Run functions in this scratch area. 
        """

        # decoder.ClosurePhaseDecoder.generate_pretraining_data(
        #     num_pix=9, num_samples=1000,
        #     file_path="./data/pretrain_numpix9_1e3_samples.h5")
        
        train_file = './data/pretrain_numpix9_1e3_samples.h5'
        test_file = './data/pretest_numpix9_1e2_samples.h5'
        runner = training.Trainer(train_file, test_file,test_file,
                                  absPhi=False, signPhi=False)
        runner.scan_hyperparams()
        #runner.plot_predictions(model_name="LinearNet", model_id="6nficxdw")
        #runner.plot_sign_predictions(model_name="SignMLP", model_id="hxu0on8b")


    else:
        raise NotImplementedError("Unsupported number of command-line "
                                  "arguments")