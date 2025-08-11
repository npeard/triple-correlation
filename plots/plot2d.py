#!/usr/bin/env python

import logging
import matplotlib.pyplot as P
import matplotlib.ticker
import numpy as np
from matplotlib import gridspec
from mpl_point_clicker import clicker
from mpl_toolkits.axes_grid1 import make_axes_locatable

logger = logging.getLogger(__name__)

from biphase.solver.iterative import IterativeSolver
from fluo import Fluorescence2D


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(
        self,
        order: int = 0,
        fformat: str = '%1.1f',
        offset: bool = True,
        mathText: bool = True,
    ):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(
            self, useOffset=offset, useMathText=mathText
        )

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin=None, vmax=None):  # noqa: ARG002, ANN001
        self.format = self.fformat
        if self._useMathText:
            self.format = rf'$\mathdefault{self.format}$'


class Plot2D:
    def __init__(self, num_atoms: int = 3, num_pix: int = 201, kmax: float = 25):
        """Instantiate a Fluorescence_1D object with the given parameters. Use
        this object to run some common simulations and plot results.

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
        self.fluo = Fluorescence2D(kmax=kmax, num_pix=num_pix, num_atoms=num_atoms)

    def plot_Object(self):
        """Plot the atomic array, the array retrieved by taking the inverse
        Fourier transform of the coherent diffraction intensity, and the
        array retrieved from inverting the diffraction intensity plus the phase.
        """
        box_extent = np.max(self.fluo.x_pix[0])
        fig = P.figure(figsize=(15, 10))
        ax1 = fig.add_subplot(231)
        ax1.set_title('Object')
        ax1.imshow(
            self.fluo.object,
            extent=[-box_extent, box_extent, -box_extent, box_extent],
            origin='lower',
        )

        obj_NoPhase = np.fft.fftshift(self.fluo.coh_ft)
        obj_NoPhase = np.abs(obj_NoPhase)
        obj_Phase = np.fft.fftshift(self.fluo.coh_ft)
        phase = np.fft.fftshift(self.fluo.coh_phase)
        obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)

        obj_NoPhase = np.fft.ifftn(obj_NoPhase)
        obj_Phase = np.fft.ifftn(obj_Phase)

        obj_NoPhase = np.fft.fftshift(obj_NoPhase)
        obj_Phase = np.fft.fftshift(obj_Phase)

        box_extent = np.max(
            np.fft.fftshift(
                np.fft.fftfreq(
                    self.fluo.num_pix, d=2 * self.fluo.kmax / self.fluo.num_pix
                )
            )
        )
        ax2 = fig.add_subplot(232)
        ax2.set_title('Object from Intensity + Phase')
        # Not sure why this needs to be transposed below, but it does give the
        # correct image
        ax2.imshow(
            np.abs(obj_Phase),
            extent=[-box_extent, box_extent, -box_extent, box_extent],
            origin='lower',
        )
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])
        ax3 = fig.add_subplot(233)
        ax3.set_title('Object from Intensity')
        ax3.imshow(
            np.abs(obj_NoPhase),
            extent=[-box_extent, box_extent, -box_extent, box_extent],
            origin='lower',
        )
        ax3.set_xlim([-1, 1])
        ax3.set_ylim([-1, 1])

        # Double resolution
        box_extent = np.max(self.fluo.x_pix[0])
        ax4 = fig.add_subplot(234)
        ax4.set_title('Object')
        ax4.imshow(
            self.fluo.object_double,
            extent=[-box_extent, box_extent, -box_extent, box_extent],
            origin='lower',
        )

        obj_NoPhase = np.fft.fftshift(self.fluo.coh_ft_double)
        obj_NoPhase = np.abs(obj_NoPhase)
        obj_Phase = np.fft.fftshift(self.fluo.coh_ft_double)
        phase = np.fft.fftshift(self.fluo.coh_phase_double)
        obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)

        obj_NoPhase = np.fft.ifftn(obj_NoPhase)
        obj_Phase = np.fft.ifftn(obj_Phase)

        obj_NoPhase = np.fft.fftshift(obj_NoPhase)
        obj_Phase = np.fft.fftshift(obj_Phase)

        box_extent = np.max(
            np.fft.fftshift(
                np.fft.fftfreq(
                    self.fluo.num_pix, d=2 * self.fluo.kmax / self.fluo.num_pix
                )
            )
        )
        ax5 = fig.add_subplot(235)
        ax5.set_title('Object from Intensity + Phase')
        # Not sure why this needs to be transposed below, but it does give the
        # correct image
        ax5.imshow(
            np.abs(obj_Phase),
            extent=[-box_extent, box_extent, -box_extent, box_extent],
            origin='lower',
        )
        ax5.set_xlim([-1, 1])
        ax5.set_ylim([-1, 1])
        ax6 = fig.add_subplot(236)
        ax6.set_title('Object from Intensity')
        ax6.imshow(
            np.abs(obj_NoPhase),
            extent=[-box_extent, box_extent, -box_extent, box_extent],
            origin='lower',
        )
        ax6.set_xlim([-1, 1])
        ax6.set_ylim([-1, 1])

        P.tight_layout()
        P.show()

    def plot_Shot(self):
        """Plot a single detector shot."""
        fig = P.figure(figsize=(10, 5))
        s = fig.add_subplot(121)
        s.set_title('Coherent Intensity')
        im = s.imshow(np.abs(self.fluo.coh_ft) ** 2)
        P.colorbar(im, ax=s)

        s = fig.add_subplot(122)
        s.set_title('Incoherent (Fluorescence) Intensity')
        im = s.imshow(self.fluo.get_incoh_intens())
        P.colorbar(im, ax=s)

        fig.suptitle('Single Shot Field Intensity at Detector')
        P.tight_layout()
        P.show()

    def plot_Intensities(self, num_shots: int = 10000):
        """Plot the coherent diffraction intensity and compare it to the
        intensity pattern computed from the double correlation. Compare both
        to the sum of speckle patterns (white noise).

        Keyword arguments:
            num_shots (int) - number of shots to compute the correlation
        """
        colormap = 'viridis'

        fig = P.figure(figsize=(10, 10))
        s = fig.add_subplot(221)
        s.set_title('True Intensity')
        true = np.abs(self.fluo.coh_ft) ** 2
        im = s.imshow(true, cmap=colormap, origin='lower')
        P.colorbar(im, ax=s)

        s = fig.add_subplot(222)
        s.set_title('Intensity Measured via g2')
        measured = (
            self.fluo.marginalize_g2(num_shots=num_shots)
            - 1
            + 1.0 / self.fluo.num_atoms
        )
        im = s.imshow(measured, cmap=colormap, origin='lower')
        P.colorbar(im, ax=s)

        s = fig.add_subplot(223)
        s.set_title('Sum of Speckle Intensities')
        incoh_sum = np.zeros(2 * (self.fluo.num_pix,))
        for _n in range(num_shots):
            incoh_sum += self.fluo.get_incoh_intens()
        im = s.imshow(
            incoh_sum / num_shots, cmap=colormap, vmax=1, vmin=0, origin='lower'
        )
        P.colorbar(im, ax=s)

        s = fig.add_subplot(224)
        s.set_title('Phase Noise (True-Measured)')
        # You can't compare to coh_ft_double here, the double resolution object
        # is not the same object!
        measured = measured[
            self.fluo.num_pix // 2 : 3 * self.fluo.num_pix // 2,
            self.fluo.num_pix // 2 : 3 * self.fluo.num_pix // 2,
        ]
        im = s.imshow(
            true - measured, cmap=colormap, origin='lower'
        )  # , vmax=1, vmin=0)
        P.colorbar(im, ax=s)

        fig.suptitle('Field Intensity at Detector')
        P.tight_layout()
        P.show()

    def plot_Closure(self, num_shots: int = 10000):
        """Plot the closure computed from the structure and from the
        simulated data and compare. The two closures have different domains
        in k-space, however.

        Keyword arguments:
            num_shots (int) - number of shots to compute the correlation
        """
        dim = (2 * self.fluo.num_pix - 1) // 2
        cstruct = self.fluo.closure_from_structure()[dim, :, dim, :]
        cdata = self.fluo.closure_from_data(num_shots=num_shots)[dim, :, dim, :]

        # Plot
        fig = P.figure(figsize=(15, 5))
        s = fig.add_subplot(131)
        im = s.imshow(cdata)
        s.set_title('Closure from Data')
        P.colorbar(im, ax=s)

        s = fig.add_subplot(132)
        im = s.imshow(cstruct)
        s.set_title('Closure from Structure')
        P.colorbar(im, ax=s)

        s = fig.add_subplot(133)
        im = s.imshow(cdata - cstruct)
        s.set_title('Difference')
        P.colorbar(im, ax=s)
        P.tight_layout()
        P.show()

    def plot_ClosurePhase(self, num_shots: int = 10000):
        """Plot a slice of the closure phase from the structure and simulated
        data.

        Keyword arguments:
            num_shots (int) - number of shots to compute the correlation
        """
        dim = (2 * self.fluo.num_pix - 1) // 2
        cPhaseStruct = np.arccos(self.fluo.cosPhi_from_structure()[dim, :, dim, :])
        cPhaseData = np.arccos(
            self.fluo.cosPhi_from_data(num_shots=num_shots)[dim, :, dim, :]
        )

        # Plot
        fig = P.figure(figsize=(15, 5))
        s = fig.add_subplot(131)
        im = s.imshow(cPhaseData)
        s.set_title('Phase from Data')
        P.colorbar(im, ax=s)

        s = fig.add_subplot(132)
        im = s.imshow(np.abs(cPhaseStruct))
        s.set_title('Phase from Structure')
        P.colorbar(im, ax=s)

        s = fig.add_subplot(133)
        im = s.imshow(np.arccos(np.cos(cPhaseData) - np.cos(cPhaseStruct)))
        s.set_title('arccos(cos(Phi_data) - cos(Phi_Struct))')
        P.colorbar(im, ax=s)
        P.tight_layout()
        P.show()

    def plot_cosPhi(self, num_shots: int = 10000):
        """Plot a slice of the cosine of the closure phase from the structure and
        simulated data.

        Keyword arguments:
            num_shots (int) - number of shots to compute the correlation
        """
        cosPhi_from_phase = self.fluo.cosPhi_from_phase()

        cosPhi_from_data = self.fluo.cosPhi_from_data(num_shots=num_shots)
        cosPhi_from_data = cosPhi_from_data[
            self.fluo.num_pix - 1 : 3 * self.fluo.num_pix // 2,
            self.fluo.num_pix - 1 : 3 * self.fluo.num_pix // 2,
            self.fluo.num_pix - 1 : 3 * self.fluo.num_pix // 2,
            self.fluo.num_pix - 1 : 3 * self.fluo.num_pix // 2,
        ]

        assert cosPhi_from_data.shape == cosPhi_from_phase.shape, 'Shapes do not match'

        # Choose a random 2D slice for plotting
        cosPhi_from_data = cosPhi_from_data[1, :, 1, :]
        cosPhi_from_phase = cosPhi_from_phase[1, :, 1, :]

        assert cosPhi_from_data.shape == cosPhi_from_phase.shape, 'Shapes do not match'

        fig = P.figure(figsize=(10, 5))
        s = fig.add_subplot(121)
        im = s.imshow(cosPhi_from_data, origin='lower')
        s.set_title('cos(Phi) from Data')
        P.colorbar(im, ax=s)

        s = fig.add_subplot(122)
        im = s.imshow(cosPhi_from_phase, origin='lower')
        s.set_title('cos(Phi) from Phase')
        P.colorbar(im, ax=s)
        P.tight_layout()
        P.show()

    def plot_PhiSolver_manualSelect(  # noqa: PLR0915, PLR0912, C901
        self, num_shots: int = 1000, altLabel: bool = False
    ):
        """Plot the phase retrieved using sign information from all
        constraints of Phi and perform re-solving using user input.

        Keyword arguments:
            num_shots (int) - number of shots to compute the correlation
        """
        quad1_real_phase = self.fluo.coh_phase_double[
            self.num_pix - 1 :, self.num_pix - 1 :
        ]
        quad2_real_phase = self.fluo.coh_phase_double[
            : self.num_pix, self.num_pix - 1 :
        ][::-1, :]

        cosPhi_from_dataPhase = self.fluo.cosPhi_from_data(num_shots=num_shots)
        quad1_cosPhi_from_dataPhase = (
            cosPhi_from_dataPhase[
                self.num_pix - 1 : 2 * self.num_pix,
                self.num_pix - 1 : 2 * self.num_pix,
                self.num_pix - 1 : 2 * self.num_pix,
                self.num_pix - 1 : 2 * self.num_pix,
            ]
            + cosPhi_from_dataPhase[
                0 : self.num_pix, 0 : self.num_pix, 0 : self.num_pix, 0 : self.num_pix
            ][::-1, ::-1, ::-1, ::-1]
        ) / 2
        quad2_cosPhi_from_dataPhase = (
            cosPhi_from_dataPhase[
                0 : self.num_pix,
                self.num_pix - 1 : 2 * self.num_pix,
                0 : self.num_pix,
                self.num_pix - 1 : 2 * self.num_pix,
            ]
            + cosPhi_from_dataPhase[
                self.num_pix - 1 : 2 * self.num_pix,
                0 : self.num_pix,
                self.num_pix - 1 : 2 * self.num_pix,
                0 : self.num_pix,
            ][::-1, ::-1, ::-1, ::-1]
        ) / 2

        quad1_cosPhi = quad1_cosPhi_from_dataPhase
        quad2_cosPhi = quad2_cosPhi_from_dataPhase[::-1, :, ::-1, :]

        quad1_solved = None
        quad2_solved = None
        quad1_error = None
        quad2_error = None

        quad1_alternates = np.zeros(2 * (self.num_pix,))
        quad2_alternates = np.zeros(2 * (self.num_pix,))

        # Manually solve quadrant 1
        xAlt = 0  # X position of user-labeled alternates
        yAlt = 0  # Y position of user-labeled alternates
        while (xAlt is not None) & (yAlt is not None):
            X0 = [quad1_real_phase[0, 1], quad1_real_phase[1, 0]]
            solver = IterativeSolver(quad1_cosPhi)
            quad1_solved, quad1_error = solver.manual_solve(
                initial_phase=X0, alt_pixels=quad1_alternates
            )

            fig = P.figure(figsize=(7, 7))
            P.rcParams.update({'font.size': 22})
            # Plot the error
            ax1 = fig.add_subplot(221)
            im = ax1.imshow(quad1_error, cmap='coolwarm', origin='lower')
            P.colorbar(im, ax=ax1)
            klicker = clicker(ax1, ['event'], markers=['x'])
            ax1.set_title('Error ' + str(int(np.sum(quad1_error))))
            # Plot the alternates
            ax2 = fig.add_subplot(222)
            im = ax2.imshow(quad1_alternates, cmap='gray', origin='lower')
            P.colorbar(im, ax=ax2)
            ax2.set_title('Alternate')
            # Plot solved phase
            ax3 = fig.add_subplot(223)
            im = ax3.imshow(quad1_solved, cmap='viridis', origin='lower')
            P.colorbar(im, ax=ax3)
            ax3.set_title('Solved')
            # Plot real phase
            ax4 = fig.add_subplot(224)
            im = ax4.imshow(quad1_real_phase, cmap='viridis', origin='lower')
            P.colorbar(im, ax=ax4)
            ax4.set_title('Quad 1')
            P.show()

            if klicker.get_positions()['event'].size != 0:
                xAlt = np.round(klicker.get_positions()['event'][:, 1]).astype(int)
                yAlt = np.round(klicker.get_positions()['event'][:, 0]).astype(int)

                logger.debug('xAlt: %s, yAlt: %s', xAlt, yAlt)
                if (quad1_alternates[xAlt, yAlt] == 1).any():
                    quad1_alternates[xAlt, yAlt] = 0
                else:
                    quad1_alternates[xAlt, yAlt] = 1
            else:
                xAlt = None
                yAlt = None

        # Manually solve quadrant 2
        xAlt = 0  # X position of user-labeled alternates
        yAlt = 0  # Y position of user-labeled alternates
        while (xAlt is not None) & (yAlt is not None):
            X0 = [quad2_real_phase[0, 1], quad2_real_phase[1, 0]]
            solver = IterativeSolver(quad2_cosPhi)
            quad2_solved, quad2_error = solver.manual_solve(
                initial_phase=X0, alt_pixels=quad2_alternates
            )

            fig = P.figure(figsize=(7, 7))
            P.rcParams.update({'font.size': 22})
            # Plot the error
            ax1 = fig.add_subplot(221)
            im = ax1.imshow(quad2_error, cmap='coolwarm', origin='lower')
            P.colorbar(im, ax=ax1)
            # klicker = clicker(ax1, ['event'], markers=['x'])  # clicker not available
            ax1.set_title('Error ' + str(int(np.sum(quad2_error))))
            # Plot the alternates
            ax2 = fig.add_subplot(222)
            im = ax2.imshow(quad2_alternates, cmap='gray', origin='lower')
            P.colorbar(im, ax=ax2)
            ax2.set_title('Alternate')
            # Plot solved phase
            ax3 = fig.add_subplot(223)
            im = ax3.imshow(quad2_solved, cmap='viridis', origin='lower')
            P.colorbar(im, ax=ax3)
            ax3.set_title('Solved')
            # Plot real phase
            ax4 = fig.add_subplot(224)
            im = ax4.imshow(quad2_real_phase, cmap='viridis', origin='lower')
            P.colorbar(im, ax=ax4)
            ax4.set_title('Quad 2')
            P.show()

            if klicker.get_positions()['event'].size != 0:
                xAlt = np.round(klicker.get_positions()['event'][:, 1]).astype(int)
                yAlt = np.round(klicker.get_positions()['event'][:, 0]).astype(int)

                logger.debug('xAlt: %s, yAlt: %s', xAlt, yAlt)
                if (quad2_alternates[xAlt, yAlt] == 1).any():
                    quad2_alternates[xAlt, yAlt] = 0
                else:
                    quad2_alternates[xAlt, yAlt] = 1
            else:
                xAlt = None
                yAlt = None

        # Show final complete result
        error = np.zeros_like(self.fluo.coh_phase_double)
        solved = np.zeros_like(self.fluo.coh_phase_double)
        true = self.fluo.coh_phase_double

        solved[self.num_pix - 1 :, self.num_pix - 1 :] = quad1_solved
        solved[: self.num_pix, self.num_pix - 1 :] = quad2_solved[::-1, :]
        solved[:, : self.num_pix] = -solved[:, self.num_pix - 1 :][::-1, ::-1]

        error[self.num_pix - 1 :, self.num_pix - 1 :] = quad1_error
        error[: self.num_pix, self.num_pix - 1 :] = quad2_error[::-1, :]
        error[:, : self.num_pix] = error[:, self.num_pix - 1 :][::-1, ::-1]

        box_extent = np.max(
            np.fft.fftshift(
                np.fft.fftfreq(
                    2 * self.fluo.num_pix, d=2 * self.fluo.kmax / self.fluo.num_pix
                )
            )
        )
        measured_amplitude = (
            self.fluo.marginalize_g2(num_shots=num_shots)
            - 1
            + 1.0 / self.fluo.num_atoms
        )
        measured_amplitude[measured_amplitude < 0] = 0

        obj_solved = np.abs(
            np.fft.fftshift(
                np.fft.ifftn(np.fft.fftshift(measured_amplitude * np.exp(1j * solved)))
            )
        )
        # obj_true = np.abs(
        #     np.fft.fftshift(
        #         np.fft.ifftn(np.fft.fftshift(measured_amplitude * np.exp(1j * true)))
        #     )
        # )

        outer = gridspec.GridSpec(1, 2, width_ratios=[3.7, 1], wspace=0.4)
        gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.4)
        gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])
        fig = P.figure(figsize=(16, 4))
        P.rcParams.update({'font.size': 16})
        # Plot full solved
        ax1 = fig.add_subplot(gs1[0])  # 141)
        im = ax1.imshow(
            solved,
            cmap='twilight_shifted',
            origin='lower',
            extent=[-2 * self.kmax, 2 * self.kmax, -2 * self.kmax, 2 * self.kmax],
            vmin=-np.pi,
            vmax=np.pi,
        )
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes('right', size='5%', pad=0.05)
        cbar = P.colorbar(im, cax=cax1)
        ax1.set_aspect('equal')
        ax1.get_xaxis().set_visible(False)
        # ax1.get_yaxis().set_visible(False)
        cbar.set_ticks([-np.pi, 0, np.pi])
        cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
        if not altLabel:
            ax1.text(
                0.05,
                0.95,
                'A',
                transform=ax1.transAxes,
                fontsize=22,
                fontweight='bold',
                va='top',
                c='white',
            )
        if altLabel:
            ax1.text(
                0.05,
                0.95,
                'E',
                transform=ax1.transAxes,
                fontsize=22,
                fontweight='bold',
                va='top',
                c='white',
            )
            ax1.get_xaxis().set_visible(True)
        ax1.set_xlabel(r'$k_x$ $[\mathrm{Length}]^{-1}$')
        ax1.set_ylabel(r'$k_y$ $[\mathrm{Length}]^{-1}$')
        if not altLabel:
            ax1.set_title(r'$\phi_{\mathrm{Solved}}$')
        ax1.set_xticks([-4, 0, 4])
        ax1.set_yticks([-4, 0, 4])
        # Plot full true
        # ax2 = fig.add_subplot(145)
        # im = ax2.imshow(true, cmap='twilight_shifted', origin="lower", extent=[-2*self.kmax,2*self.kmax,-2*self.kmax,2*self.kmax], vmin=-np.pi, vmax=np.pi)
        # divider = make_axes_locatable(ax2)
        # cax2 = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = P.colorbar(im, cax=cax2)
        # ax2.set_aspect('equal')
        # #ax2.get_xaxis().set_visible(False)
        # #ax2.get_yaxis().set_visible(False)
        # cbar.set_ticks([-np.pi, 0, np.pi])
        # cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
        # ax2.text(0.05, 0.95, 'D', transform=ax2.transAxes,
        #          fontsize=22, fontweight='bold', va='top', c='white')
        # ax2.set_xlabel(r"$k_x$ $[\mathrm{Length}]^{-1}$")
        # ax2.set_ylabel(r"$k_y$ $[\mathrm{Length}]^{-1}$")
        # ax2.set_title(r"$\phi_{\mathrm{True}}$")
        # Plot full error
        ax3 = fig.add_subplot(gs1[1])  # 142)
        im = ax3.imshow(
            error,
            cmap='coolwarm',
            origin='lower',
            extent=[-2 * self.kmax, 2 * self.kmax, -2 * self.kmax, 2 * self.kmax],
        )
        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes('right', size='5%', pad=0.05)
        cbar = P.colorbar(im, cax=cax3)
        ax3.set_aspect('equal')
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        # cbar.set_ticks([-np.pi, 0, np.pi])
        # cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
        if not altLabel:
            ax3.text(
                0.05,
                0.95,
                'B',
                transform=ax3.transAxes,
                fontsize=22,
                fontweight='bold',
                va='top',
                c='white',
            )
        if altLabel:
            ax3.text(
                0.05,
                0.95,
                'F',
                transform=ax3.transAxes,
                fontsize=22,
                fontweight='bold',
                va='top',
                c='white',
            )
            ax3.get_xaxis().set_visible(True)
        ax3.set_xlabel(r'$k_x$ $[\mathrm{Length}]^{-1}$')
        ax3.set_ylabel(r'$k_y$ $[\mathrm{Length}]^{-1}$')
        if not altLabel:
            ax3.set_title(r'$\log \left[E(\phi_{\mathrm{Solved}}) \right]$')
        ax3.set_xticks([-4, 0, 4])
        ax3.set_yticks([-4, 0, 4])
        # Plot difference
        ax4 = fig.add_subplot(gs1[2])  # 143)
        im = ax4.imshow(
            true - solved,
            cmap='PRGn',
            origin='lower',
            extent=[-2 * self.kmax, 2 * self.kmax, -2 * self.kmax, 2 * self.kmax],
            vmin=-2 * np.pi,
            vmax=2 * np.pi,
        )
        divider = make_axes_locatable(ax4)
        cax4 = divider.append_axes('right', size='5%', pad=0.05)
        cbar = P.colorbar(im, cax=cax4)
        ax4.set_aspect('equal')
        ax4.get_xaxis().set_visible(False)
        ax4.get_yaxis().set_visible(False)
        cbar.set_ticks([-2 * np.pi, 0, 2 * np.pi])
        cbar.set_ticklabels([r'$-2\pi$', '0', r'$2\pi$'])
        if not altLabel:
            ax4.text(
                0.05,
                0.95,
                'C',
                transform=ax4.transAxes,
                fontsize=22,
                fontweight='bold',
                va='top',
                c='black',
            )
        if altLabel:
            ax4.text(
                0.05,
                0.95,
                'G',
                transform=ax4.transAxes,
                fontsize=22,
                fontweight='bold',
                va='top',
                c='black',
            )
            ax4.get_xaxis().set_visible(True)
        ax4.set_xlabel(r'$k_x$ $[\mathrm{Length}]^{-1}$')
        ax4.set_ylabel(r'$k_y$ $[\mathrm{Length}]^{-1}$')
        if not altLabel:
            ax4.set_title(r'$\phi_{\mathrm{True}} - \phi_{\mathrm{Solved}}$')
        ax4.set_xticks([-4, 0, 4])
        ax4.set_yticks([-4, 0, 4])
        # Plot object from solved
        ax5 = fig.add_subplot(gs2[0])  # 144)
        # I'm not sure why the axis reversals and transposes are needed here to
        # make the output match the scatter plot
        im = ax5.imshow(
            (obj_solved / np.max(obj_solved))[:, ::-1].T[::-1, :],
            cmap='hot',
            origin='lower',
            extent=[-box_extent, box_extent, -box_extent, box_extent],
            vmin=0,
            vmax=1,
        )
        ax5.scatter(
            self.fluo.coords[0, :],
            self.fluo.coords[1, :],
            facecolors='none',
            edgecolors='cyan',
        )
        divider = make_axes_locatable(ax5)
        cax5 = divider.append_axes('right', size='5%', pad=0.05)
        cbar = P.colorbar(im, cax=cax5)
        ax5.set_aspect('equal')
        ax5.get_xaxis().set_visible(False)
        # ax5.get_yaxis().set_visible(False)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels([r'0', '1'])
        if not altLabel:
            ax5.text(
                0.05,
                0.95,
                'D',
                transform=ax5.transAxes,
                fontsize=22,
                fontweight='bold',
                va='top',
                c='white',
            )
        if altLabel:
            ax5.text(
                0.05,
                0.95,
                'H',
                transform=ax5.transAxes,
                fontsize=22,
                fontweight='bold',
                va='top',
                c='white',
            )
            ax5.get_xaxis().set_visible(True)
        ax5.set_xlabel(r'$X$ $[\mathrm{Length}]$')
        ax5.set_ylabel(r'$Y$ $[\mathrm{Length}]$')
        if not altLabel:
            ax5.set_title(
                r'$ \tilde{\mathcal{F}} \left\{ \left|g^{(1)} \right| '
                r'e^{i \phi_{\mathrm{Solved}}} \right\}$'
            )
        ax5.set_xticks([-1, 0, 1])
        ax5.set_yticks([-1, 0, 1])
        # Plot object from true
        # ax6 = fig.add_subplot(246)
        # # I'm not sure why the axis reversals and transposes are needed here to make the output match the scatter plot
        # im = ax6.imshow( (obj_true/np.max(obj_true))[:,::-1].T[::-1,:], cmap='hot', origin="lower", extent=[-box_extent,box_extent,-box_extent,box_extent])
        # ax6.scatter(self.fluo.coords[0, :], self.fluo.coords[1, :])
        # divider = make_axes_locatable(ax6)
        # cax6 = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = P.colorbar(im, cax=cax6)
        # ax6.set_aspect('equal')
        # #ax6.get_xaxis().set_visible(False)
        # #ax6.get_yaxis().set_visible(False)
        # #cbar.set_ticks([-np.pi, 0, np.pi])
        # #cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
        # ax6.text(0.05, 0.95, 'F', transform=ax6.transAxes,
        #          fontsize=22, fontweight='bold', va='top', c='white')
        # ax6.set_xlabel(r"$X$ $[\mathrm{Length}]$")
        # ax6.set_ylabel(r"$Y$ $[\mathrm{Length}]$")
        # ax6.set_title(r"$ \tilde{\mathcal{F}} \left\{ \left|g^{(1)} \right| e^{i \phi_{\mathrm{True}}} \right\}$")
        # ax6.set_xticks([-1, 0, 1])
        # ax6.set_yticks([-1, 0, 1])
        # P.tight_layout(pad = 0)
        P.show()

    def plot_PhiSolver(self, num_shots: int = 1000):
        """Plot the phase retrieved using sign information from all
        constraints of Phi.

        Keyword arguments:
            num_shots (int) - number of shots to compute the correlation
        """
        real_phase = self.fluo.coh_phase_double[
            self.fluo.num_pix - 1 : 2 * self.fluo.num_pix - 1,
            self.fluo.num_pix - 1 : 2 * self.fluo.num_pix - 1,
        ]
        # real_phase = unwrap_phase(real_phase)
        # real_phase = self.coh_phase_double[
        #              self.num_pix - 1:3 * self.num_pix // 2,
        #              self.num_pix - 1:3 * self.num_pix // 2]
        initial_phase = [real_phase[0, 1], real_phase[1, 0]]
        cosPhi = self.fluo.cosPhi_from_data(num_shots=num_shots)
        solver = IterativeSolver(cosPhi)
        solved, error = solver.solve(initial_phase=initial_phase[0], error_reject=-10)

        fig = P.figure(figsize=(10, 10))
        P.rcParams.update({'font.size': 22})
        # Plot the solved phase branch
        ax1 = fig.add_subplot(221)
        im = ax1.imshow(solved, cmap='viridis', origin='lower')
        ax1.set_title(r'$\phi$ Solved')
        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes('right', size='5%', pad=0.05)
        P.colorbar(im, cax=cax1)
        ax1.set_aspect('equal')

        ax2 = fig.add_subplot(222)
        im = ax2.imshow(real_phase, cmap='viridis', origin='lower')
        ax2.set_title(r'$\phi$ Truth')
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes('right', size='5%', pad=0.05)
        P.colorbar(im, cax=cax2)
        ax2.set_aspect('equal')

        ax3 = fig.add_subplot(223)
        im = ax3.imshow(error, cmap='coolwarm', origin='lower')
        ax3.set_title(r'$\log \left[E(\phi) \right]$')
        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes('right', size='5%', pad=0.05)
        P.colorbar(im, cax=cax3)
        ax3.set_aspect('equal')

        ax4 = fig.add_subplot(224)
        im = ax4.imshow(real_phase - solved, cmap='coolwarm', origin='lower')
        ax4.set_title('Truth - Solved')
        divider = make_axes_locatable(ax4)
        cax4 = divider.append_axes('right', size='5%', pad=0.05)
        P.colorbar(im, format=OOMFormatter(-4, mathText=True), cax=cax4)
        ax4.set_aspect('equal')
        P.tight_layout()
        P.subplots_adjust(wspace=0.4, hspace=0.4)
        P.show()


if __name__ == '__main__':
    plotter = Plot2D(num_pix=11, num_atoms=4, kmax=3)
    plotter.plot_PhiSolver_manualSelect(num_shots=1000)
