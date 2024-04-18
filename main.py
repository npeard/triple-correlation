#!/usr/bin/env python

import numpy as np
import sys
import plot1d
#import plot2d
#import PaperFigures
#import timeit

import speckle1d
#import speckle2d
import triphase1d
#import triphase2d
import train

if __name__ == '__main__':
    # np.random.seed(0x5EED)
    if len(sys.argv) == 1:
        """Run functions in this scratch area. 
        """

        # triphase1d.generate_training_set_from_reverse(num_data=1000,
        #                                    file='C:\\Users\\npeard\\PycharmProjects\\triple-correlation\\data\\Valid1D-kmax3-pix51-ideal-1k.h5')

        valid_file = 'C:\\Users\\npeard\\PycharmProjects\\triple-correlation\\data\\Valid1D-kmax3-pix51-ideal-1k.h5'
        train_file = 'C:\\Users\\npeard\\PycharmProjects\\triple-correlation\\data\\Train1D-kmax3-pix51-ideal-20k.h5'
        test_file = 'C:\\Users\\npeard\\PycharmProjects\\triple-correlation\\data\\Test1D-kmax3-pix51-ideal-1k.h5'
        runner = train.TrainingRunner(train_file, valid_file, test_file, linearOnly=True)
        runner.train_singleLinear()

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