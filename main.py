#!/usr/bin/env python

import numpy as np
import sys
import plot1d
import plot2d
import PaperFigures
import timeit

import speckle1d
import speckle2d
import triphase1d
import triphase2d

if __name__ == '__main__':
    np.random.seed(0x5EED)
    if len(sys.argv) == 1:
        """Run functions in this scratch area. 
        """
        #plot = Plot_1D.Plot_1D(num_atoms=5, num_pix=101, kmax=10)
        #plot.plot_Object()
        #for n in range(3):
        #    plot.plot_Shot()
        #plot.plot_g2(num_shots=1000)
        #plot.plot_Intensities(num_shots=10000)
        #plot.plot_g3(num_shots=1000)
        #plot.plot_Intensity_Error()
        #plot.plot_cosPhi(num_shots=1000)
        #plot.plot_Closure(num_shots=1000)
        #plot.plot_ClosurePhase(num_shots=1000)
        #plot.plot_PhiSolver(num_shots=10000)
        #plot.plot_simple_PhiSolve(num_shots=1000)
        #plot.learnStructure(num_shots=10000)

        #plot = Plot_2D.Plot_2D(num_pix=11, num_atoms=7, kmax=2)
        #plot.plot_Object()
        #for n in range(3):
        #     plot.plot_Shot()
        #plot.plot_Intensities(num_shots=1000)
        #plot.plot_Closure(num_shots=1000)
        #plot.plot_ClosurePhase(num_shots=1000)
        #plot.plot_cosPhi(num_shots=1000)
        #plot.plot_PhiSolver(num_shots=20000)
        #plot.plot_PhiSolver_manualSelect(num_shots=1000)
        #plot.learnStructure(num_shots=10000)

        TriPhase_1D.generate_training_set_from_reverse(num_data=1000,
                                           file="/Users/nolanpeard/Desktop/Test1D-kmax3-ideal.h5")

        #TriPhase_2D.generate_training_data(num_data=100,
        #
    #                                   file="/Users/nolanpeard/Desktop/Test2D-kmax2.h5")

        #PaperFigures.Figure_S1()

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