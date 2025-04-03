#!/usr/bin/env python

import numpy as np
import pylab as P
from fluo import Fluorescence1D


class Plot1D:
    def __init__(self, num_atoms=5, num_pix=201, kmax=10):
        """Instantiate a Fluorescence_1D object with the given parameters. Use
        this object to run simulations and plot results.

        Keyword arguments:
            kmax (float) - maximum coordinate in reciprocal space

            num_pix (int) - number of pixels in reciprocal space,
        must be an odd number

            num_atoms (int) - number of atoms in the random array,
        smaller numbers of atoms lead to less HBT phase noise
        """
        self.num_atoms = num_atoms
        self.num_pix = num_pix
        self.kmax = kmax
        self.fluo = Fluorescence1D(kmax=self.kmax,
                                  num_pix=self.num_pix,
                                  num_atoms=self.num_atoms)

    def plot_Object(self):
        """Plot the atomic array, the array retrieved by taking the inverse
        Fourier transform of the coherent diffraction intensity, and the
        array retrieved from inverting the diffraction intensity plus the phase.
        """
        P.plot(self.fluo.x_pix, self.fluo.object, label="Object")
        print("Coordinates", self.fluo.coords)
        obj_NoPhase = np.fft.fftshift(self.fluo.coh_ft)
        obj_NoPhase = np.abs(obj_NoPhase)
        obj_Phase = np.fft.fftshift(self.fluo.coh_ft)
        phase = np.fft.fftshift(self.fluo.coh_phase)
        obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)

        obj_NoPhase = np.fft.ifft(obj_NoPhase)
        obj_Phase = np.fft.ifft(obj_Phase)

        obj_NoPhase = np.fft.fftshift(obj_NoPhase)
        obj_Phase = np.fft.fftshift(obj_Phase)

        scaled_x = np.fft.fftshift(np.fft.fftfreq(
            self.fluo.num_pix, d=2 * self.fluo.kmax / self.fluo.num_pix))
        P.plot(scaled_x, np.abs(obj_NoPhase),
               label="Object from Intensity")
        P.plot(scaled_x, np.abs(obj_Phase), '--',
               label="Object from Intensity + Phase")
        P.xlim([-1, 1])
        P.legend()
        P.tight_layout()
        P.show()

    def plot_Shot(self):
        """Plot a single detector shot.
        """
        P.plot(self.fluo.k_pix, np.abs(self.fluo.coh_ft)**2)
        P.plot(self.fluo.k_pix, self.fluo.get_incoh_intens())
        P.title("Single Shot Intensity at Detector")
        P.legend(("Coherent Intensity", "Incoherent (Fluorescence) Intensity"))
        P.tight_layout()
        P.show()

    def plot_Intensities(self, num_shots=10000):
        """Plot the coherent diffraction intensity and compare it to the
        intensity pattern computed from the double correlation. Compare both
        to the sum of speckle patterns (white noise).

        Keyword arguments:
            num_shots (int) - number of shots to compute the correlation
        """
        P.plot(self.fluo.k_pix, np.abs(self.fluo.coh_ft)**2,
               label='True Intensity')

        g2 = self.fluo.marginalize_g2(num_shots=num_shots)
        measured = (g2 - 1 + 1. / self.fluo.num_atoms)
        P.plot(self.fluo.q_pix, measured, 'o--',
               label=r'Intensity Computed via $g^2$')

        incoh_sum = np.zeros_like(self.fluo.k_pix)
        for n in range(num_shots):
            incoh_sum += self.fluo.get_incoh_intens()
        P.plot(self.fluo.k_pix, incoh_sum / num_shots
               * self.fluo.num_atoms, label='Summed Speckle Pattern')

        P.title("Field Intensity at Detector")
        P.legend()
        P.tight_layout()
        P.show()

    def plot_g2(self, num_shots=10000):
        """Plot the full second-order correlation function.

        Keyword arguments:
            num_shots (int) - number of shots to compute the correlation
        """
        g2 = self.fluo.get_g2(num_shots=num_shots)
        P.imshow(g2, cmap='gray', origin='lower')
        P.title("g2")
        P.tight_layout()
        P.show()

    def plot_g3(self, num_shots=10000):
        """Plot the marginalized third-order correlation function.

        Keyword arguments:
            num_shots (int) - number of shots to compute the correlation
        """
        fig = P.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(111)
        g3 = self.fluo.marginalize_g3(num_shots=num_shots)
        im = ax1.imshow(g3)
        P.colorbar(im, ax=ax1)

        P.tight_layout()
        P.show()

    def plot_Closure(self, num_shots=10000):
        """Plot the closure computed from the structure and from the
        simulated data and compare. The two closures have different domains
        in k-space, however.

        Keyword arguments:
            num_shots (int) - number of shots to compute the correlation
        """
        fig = P.figure(figsize=(15, 5))
        cdata = self.fluo.closure_from_data(num_shots=num_shots)
        s = fig.add_subplot(131)
        im = s.imshow(cdata, origin='lower')
        s.set_title("Closure from Data")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(132)
        cstruct = self.fluo.closure_from_structure()
        im = s.imshow(self.fluo.closure_from_structure(), origin='lower')
        s.set_title("Closure from Structure")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(133)
        im = s.imshow(cdata - cstruct, origin='lower')
        s.set_title("Difference")
        P.colorbar(im, ax=s)
        P.tight_layout()
        P.show()

    def plot_ClosurePhase(self, num_shots=10000):
        """Plot the closure phase from the structure and simulated data.

        Keyword arguments:
            num_shots (int) - number of shots to compute the correlation
        """
        fig = P.figure(figsize=(10, 5))
        s = fig.add_subplot(121)
        im = s.imshow(np.arccos(self.fluo.cosPhi_from_data(
            num_shots=num_shots)), origin='lower')
        s.set_title("Phase from Data")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(122)
        im = s.imshow(np.arccos(self.fluo.cosPhi_from_structure()),
                      origin='lower')
        s.set_title("Phase from Structure")
        P.colorbar(im, ax=s)
        P.tight_layout()
        P.show()

    def plot_cosPhi(self, num_shots=10000):
        """Plot the cosine of the closure phase from the structure and
        simulated data.

        Keyword arguments:
            num_shots (int) - number of shots to compute the correlation
        """
        cosPhi_from_data = self.fluo.cosPhi_from_data(num_shots=num_shots)

        # Same information content is in both sides of
        # diagonal axis, but the cosPhi is not symmetric as I've
        # assumed. Does the antisymmetry with a transpose convey extra
        # information?

        cosPhi_from_data = cosPhi_from_data[
            self.fluo.num_pix - 1:3 * self.fluo.num_pix // 2,
            self.fluo.num_pix - 1:3 * self.fluo.num_pix // 2]
        cosPhi_from_data_symmetrized = (
            cosPhi_from_data + cosPhi_from_data.T) / 2

        # cosPhi_from_structurePhase = self.fluo.cosPhi_from_structure()[
        #     self.fluo.num_pix-1:, self.fluo.num_pix-1:]

        cosPhi_from_phase = self.fluo.cosPhi_from_phase()
        cosPhi_from_phase_symmetrized = (cosPhi_from_phase +
                                         cosPhi_from_phase.T) / 2
        print(cosPhi_from_phase.shape)
        print(cosPhi_from_data.shape)
        assert cosPhi_from_data.shape == cosPhi_from_phase.shape, \
            "Shapes do not match"

        fig = P.figure(figsize=(7, 7))
        s = fig.add_subplot(221)
        im = s.imshow(cosPhi_from_data, origin="lower")
        s.set_title("cosPhi from Data")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(222)
        im = s.imshow(cosPhi_from_phase, origin="lower")
        s.set_title("cosPhi from Phase")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(223)
        im = s.imshow(cosPhi_from_data - cosPhi_from_data_symmetrized,
                      origin="lower")
        s.set_title("AntiSymmetry from Data")
        P.colorbar(im, ax=s)

        s = fig.add_subplot(224)
        im = s.imshow(cosPhi_from_phase - cosPhi_from_phase_symmetrized,
                      origin="lower")
        s.set_title("AntiSymmetry from Phase")
        P.colorbar(im, ax=s)

        P.tight_layout()
        P.show()

    def plot_simple_PhiSolve(self, num_shots=1000):
        """Plot the phase retrieved without sign information from all
        constraints of Phi.

        Keyword arguments:
            num_shots (int) - number of shots to compute the correlation
        """
        cosPhi = self.fluo.cosPhi_from_data(num_shots=num_shots)
        initial_phase = self.fluo.coh_phase_double[
            self.num_pix - 1:3 * self.num_pix // 2][1]
        solved = TriPhase_1D.simple_PhiSolver(cosPhi,
                                              initial_phase=initial_phase)
        from skimage.restoration import unwrap_phase
        solved = unwrap_phase(solved)
        real_phase = self.fluo.coh_phase[self.fluo.num_pix // 2:]
        real_phase = unwrap_phase(real_phase)

        fig = P.figure(figsize=(7, 7))
        # Plot the solved phase branch
        s = fig.add_subplot(111)
        P.plot(np.linspace(0, len(real_phase), len(real_phase)),
               real_phase, label='Exact')
        P.plot(np.linspace(0, len(solved), len(solved)), solved,
               'o--', label=r'$\Phi = |\Phi|$')
        s.set_ylabel(r'$\phi$')
        s.set_xlabel("Pixel Index")
        P.legend()
        P.tight_layout()
        P.show()

    def plot_PhiSolver(self, num_shots=10000):
        """Plot the phase retrieved using sign information from all
        constraints of Phi.

        Keyword arguments:
            num_shots (int) - number of shots to compute the correlation
        """
        cosPhi = self.fluo.cosPhi_from_data(num_shots=num_shots)
        initial_phase = self.fluo.coh_phase_double[
            self.num_pix - 1:3 * self.num_pix // 2][1]
        solved, error = TriPhase_1D.PhiSolver(cosPhi,
                                              initial_phase=initial_phase)
        real_phase = self.fluo.coh_phase_double[self.fluo.num_pix - 1:]

        # Unwrap the phase
        from skimage.restoration import unwrap_phase
        plot_solved = unwrap_phase(solved[:])
        plot_real_phase = unwrap_phase(real_phase)

        fig = P.figure(figsize=(5, 5))
        # Plot the solved phase branch
        s = fig.add_subplot(111)
        P.plot(np.linspace(0, len(real_phase), len(real_phase)),
               plot_real_phase, 'o--', label='Exact')
        P.plot(np.linspace(0, len(real_phase), len(real_phase)),
               plot_solved, label='Solved')
        P.plot(np.linspace(0, len(real_phase), len(real_phase)),
               np.cos(plot_real_phase - plot_solved), label='cos(Diff)')
        P.legend()
        P.tight_layout()
        P.show()

if __name__ == '__main__':
    plotter = Plot1D(num_atoms=5, num_pix=21, kmax=10)
    plotter.plot_cosPhi()
