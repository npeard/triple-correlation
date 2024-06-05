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
import train
import decoder
import models

if __name__ == '__main__':
    np.random.seed(0x5EED+3)
    if len(sys.argv) == 1:
        """Run functions in this scratch area. 
        """

        # triphase1d.generate_training_set_from_reverse(num_data=int(1e3),
        #                                    file='C:\\Users\\npeard\\PycharmProjects\\triple-correlation\\data\\Test1D-kmax3-pix51-ideal-1e3.h5')

        valid_file = 'C:\\Users\\npeard\\PycharmProjects\\triple-correlation\\data\\Valid1D-kmax3-pix51-ideal-2e4.h5'
        train_file = 'C:\\Users\\npeard\\PycharmProjects\\triple-correlation\\data\\Train1D-kmax3-pix51-ideal-1e6.h5'
        test_file = 'C:\\Users\\npeard\\PycharmProjects\\triple-correlation\\data\\Test1D-kmax3-pix51-ideal-1e3.h5'
        runner = train.TrainingRunner(train_file, valid_file, test_file)
        runner.scan_hyperparams()
        runner = train.TrainingRunner(train_file, valid_file, test_file, linear_only=True)
        runner.scan_linear_hyperparams()
        #runner.load_model()

        # test_dataloader = runner.get_custom_dataloader(test_file, batch_size=1024)
        #
        # num_inputs = next(iter(test_dataloader))[0].size(-1) ** 2
        # num_outputs = next(iter(test_dataloader))[1].size(-1)
        #
        # closure = []
        # phases = []
        #
        # dec_obj = decoder.ClosurePhaseDecoder(model=models.SequentialNN(0,0,0))
        #
        # for i, (inputs, labels) in enumerate(test_dataloader):
        #     if i == 1:  # We only need one batch
        #         break
        #     #inputs = inputs.view(-1, num_inputs)  # Reshape the input data
        #     closure.extend(inputs.numpy())
        #     phases.extend(labels.numpy())
        #     start = timeit.default_timer()
        #     loss = dec_obj.encoding_loss2(outputs=labels, inputs=inputs)
        #     print("The difference of time is :",
        #           timeit.default_timer() - start)
        #
        # print(loss)
        # closure_ex = np.asarray(closure)[:3, :, :]
        # phases_ex = np.asarray(phases)[:3, num_outputs // 2:]
        #
        # print(closure_ex.shape, phases_ex.shape)

        # def test_get_g2():
        #     # Test case 1: g2 is already computed
        #     obj = Speckle_1D.Fluorescence_1D()
        #     obj.g2 = np.ones((10,))
        #     assert np.array_equal(obj.get_g2(), np.ones((10,)))
        #
        #     # Test case 2: g2 is not computed
        #     obj = Speckle_1D.Fluorescence_1D()
        #     obj.get_incoh_intens = lambda: np.ones((10,))
        #     obj.num_pix = 10
        #     assert np.array_equal(obj.get_g2(), np.ones((10,10)))
        #
        #     # Test case 3: num_shots = 1000
        #     obj = Speckle_1D.Fluorescence_1D()
        #     obj.get_incoh_intens = lambda: np.ones((10,))
        #     obj.num_pix = 10
        #     result = obj.get_g2(num_shots=1000)
        #     assert result.shape == (10,10)
        #     assert np.allclose(result, np.ones((10,10)))
        #     #assert np.allclose(result, np.zeros((10,10)))
        #


    else:
        print("Error: Unsupported number of command-line arguments")