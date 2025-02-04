#!/usr/bin/env python

import numpy as np
import sys
import training
import datasets

if __name__ == '__main__':
    np.random.seed()
    if len(sys.argv) == 1:
        """Run functions in this scratch area. 
        """
        num_pix = 11
        train_samples = int(1e5)
        valid_samples = int(1e4)
        train_file = f"./data/pretrain_numpix{num_pix}_{train_samples:.0e}_samples.h5"
        valid_file = f"./data/prevalid_numpix{num_pix}_{valid_samples:.0e}_samples.h5"
        test_file = f"./data/pretest_numpix{num_pix}_{valid_samples:.0e}_samples.h5"

        # datasets.generate_pretraining_data(
        #     num_pix=num_pix, num_samples=int(train_samples),
        #     file_path=train_file)
        
        # datasets.generate_pretraining_data(
        #     num_pix=num_pix, num_samples=int(valid_samples),
        #     file_path=valid_file)
        
        # datasets.generate_pretraining_data(
        #     num_pix=num_pix, num_samples=int(valid_samples),
        #     file_path=test_file)

        runner = training.Trainer(train_file, valid_file, test_file,
                                  absPhi=True, signPhi=False, multiTask=False, log=True)
        runner.scan_hyperparams(num_samples=16)
        
        # runner.plot_phase_predictions(model_name="GPT", model_id="apvlrfv3")
        
        # Best hybrid classifier so far
        # runner.plot_phase_predictions(model_name="MLP",
        #                              model_id="eojqvv3v")
        #runner.plot_sign_predictions(model_name="MLP", model_id="eojqvv3v")


    else:
        raise NotImplementedError("Unsupported number of command-line "
                                  "arguments")