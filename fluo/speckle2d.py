#!/usr/bin/env python

import numpy as np
from numba import jit


@jit(nopython=True)
def roll2d(arr, shift_x, shift_y):
    """Numba-compatible implementation of np.roll for 2D arrays.

    Parameters:
        arr (ndarray): 2D input array
        shift_x (int): Number of places to shift along first axis
        shift_y (int): Number of places to shift along second axis

    Returns:
        ndarray: Shifted array
    """
    nx, ny = arr.shape
    out = np.empty_like(arr)

    for i in range(nx):
        for j in range(ny):
            # Calculate rolled indices with proper wrapping
            idx_x = (i - shift_x) % nx
            idx_y = (j - shift_y) % ny
            out[i, j] = arr[idx_x, idx_y]

    return out


class Fluorescence2D:
    def __init__(self, kmax=5, num_pix=51, num_atoms=3, x=None):
        """Simulate fluorescence speckle from a 2D array of atoms and compute
        various correlation functions.

        Args:
            kmax (float): maximum coordinate in reciprocal space
            num_pix (int): number of pixels in reciprocal space, must be an odd
            number
            num_atoms (int): number of atoms in the random array, smaller
            numbers of atoms lead to less HBT phase noise
            x (float): array of user-supplied coordinates to define custom atom
            array
        """
        self.kmax = kmax
        self.num_pix = num_pix
        self.x = x  # User-supplied coordinates
        self.num_atoms = num_atoms
        self.init_system()

    def init_system(self):
        """Initialize arrays and variables, generate atomic array. Some arrays
        are not initialized on startup to save resources.
        """
        # print("Initializing system...")
        self.k_pix = np.mgrid[
            -self.kmax : self.kmax : 1j * self.num_pix,
            -self.kmax : self.kmax : 1j * self.num_pix,
        ]
        self.x_pix = np.mgrid[-1 : 1 : 1j * self.num_pix, -1 : 1 : 1j * self.num_pix]
        self.x_double_pix = np.mgrid[
            -1 : 1 : 1j * (2 * self.num_pix - 1), -1 : 1 : 1j * (2 * self.num_pix - 1)
        ]
        self.weights_2d = np.correlate(
            np.ones(self.num_pix), np.ones(self.num_pix), mode='full'
        )
        self.weights_2d = np.multiply.outer(self.weights_2d, self.weights_2d)
        self.q_pix = np.mgrid[
            -2 * self.kmax : 2 * self.kmax : (2 * self.num_pix - 1) * 1j,
            -2 * self.kmax : 2 * self.kmax : (2 * self.num_pix - 1) * 1j,
        ]
        self.g2 = None
        self.g3 = None
        self.g2_2d = None
        self.g3_4d = None
        self.weights_4d = None

        if self.x is None:
            self.randomize_coords()
        self.digitize_coords()

    def init_weights_4d(self):
        """Initialize the 4D weights used to marginalize the 6D triple
        correlation function.
        """
        self.weights_4d = self.compute_weights_4d(num_pix=self.num_pix)

    @staticmethod
    @jit(nopython=True, parallel=False)
    def compute_weights_4d(num_pix=1):
        """Calculate the 4D weights using explicit for-loops.

        Args:
            num_pix (int): the number of pixels, self.num_pix, for the
            simulation

        Returns:
            weights_4d (float): array of weights
        """
        weights_4d = np.zeros(
            (2 * num_pix - 1, 2 * num_pix - 1, 2 * num_pix - 1, 2 * num_pix - 1)
        )

        for k1x in range(num_pix):
            for k2x in range(num_pix):
                for k3x in range(num_pix):
                    for k1y in range(num_pix):
                        for k2y in range(num_pix):
                            for k3y in range(num_pix):
                                q1x = k1x - k2x + num_pix - 1
                                q1y = k1y - k2y + num_pix - 1
                                q2x = k2x - k3x + num_pix - 1
                                q2y = k2y - k3y + num_pix - 1
                                weights_4d[q1x, q1y, q2x, q2y] += 1
        return weights_4d

    def digitize_coords(self) -> None:
        """Digitize the atomic coordinates into a real space object for plotting.
        """
        if self.x is not None:
            self.coords = self.x

        self.object = np.zeros_like(self.x_pix[0, :, :])
        self.object[
            np.digitize(self.coords[0, :], self.x_pix[0, :, 0]),
            np.digitize(self.coords[1, :], self.x_pix[1, 0, :]),
        ] = 1 / self.num_atoms

        self.object_double = np.zeros_like(self.x_double_pix[0, :, :])
        self.object_double[
            np.digitize(self.coords[0, :], self.x_double_pix[0, :, 0]),
            np.digitize(self.coords[1, :], self.x_double_pix[1, 0, :]),
        ] = 1 / self.num_atoms

    def randomize_coords(self):
        """Randomize or load atomic coordinates and compute coherent
        diffraction quantities.
        """
        self.coords = np.random.random((2, self.num_atoms)) * 2 - 1

        self.kr_product_x = np.multiply.outer(self.k_pix[0, :, :], self.coords[0, :])
        self.kr_product_y = np.multiply.outer(self.k_pix[1, :, :], self.coords[1, :])
        self.qr_product_x = np.multiply.outer(self.q_pix[0, :, :], self.coords[0, :])
        self.qr_product_y = np.multiply.outer(self.q_pix[1, :, :], self.coords[1, :])

        self.coh_ft = np.exp(
            -1j * (self.kr_product_x + self.kr_product_y + 0) * 2 * np.pi
        ).mean(2)
        self.coh_phase = np.angle(self.coh_ft)
        self.coh_ft_double = np.exp(
            -1j * (self.qr_product_x + self.qr_product_y + 0) * 2 * np.pi
        ).mean(2)
        self.coh_phase_double = np.angle(self.coh_ft_double)

    def get_incoh_intens(self):
        """Get the fluorescence intensity in a single shot.

        Returns:
            (float) - The fluorescence intensity (2d array) across the
            detector
        """
        random_nums = np.random.random(self.num_atoms)
        exponential = np.exp(
            -1j * ((self.kr_product_x + self.kr_product_y + random_nums) * 2.0 * np.pi)
        )
        incoh = np.abs(exponential.mean(2)) ** 2

        return incoh

    def get_g2(self, num_shots=1000):
        """Get the second-order correlation function computed from the
        specified number of incoherent shots.

        Args:
            num_shots (int): the number of shots to use when computing the
            ensemble double correlation function.

        Returns:
            self.g2 (float): the 4d array of the computed double correlations
        """
        if self.g2 is not None:
            return self.g2

        print('Performing second-order intensity correlation using outer product...')
        ave_intens = np.zeros(2 * (self.num_pix,))
        self.g2 = np.zeros(4 * (self.num_pix,))

        for i in range(num_shots):
            print('Correlating frame ', i)
            incoh = self.get_incoh_intens()
            self.g2 += np.multiply.outer(incoh, incoh)
            ave_intens += incoh
        self.g2 *= num_shots / np.multiply.outer(ave_intens, ave_intens)
        print('Finished correlation...')
        return self.g2

    def marginalize_g2(self, num_shots=1000):
        """Reduce the dimensionality of the double correlation by writing it
        as a function of q instead of k in reciprocal space.

        Args:
            num_shots (int): number of shots to compute the correlation

        Returns:
            self.g2_2d (float): the dimension reduced version of the double
            correlation
        """
        if self.g2_2d is not None:
            return self.g2_2d

        if self.g2 is None:
            self.g2 = self.get_g2(num_shots=num_shots)

        self.g2_2d = self.compute_marginalized_g2(self.g2, num_pix=self.num_pix)
        self.g2_2d[self.weights_2d > 0] /= self.weights_2d[self.weights_2d > 0]

        return self.g2_2d

    @staticmethod
    @jit(nopython=True, parallel=False)
    def compute_marginalized_g2(g2, num_pix=1):
        """Compute the marginalized double correlation function by summing the 4D
        array of double correlations.

        Args:
            g2 (ndarray): The 4D array of computed double correlations.
            num_pix (int): The number of pixels for the simulation.

        Returns:
            g2_2d (ndarray): The dimension-reduced version of the double
            correlation function.
        """
        g2_2d = np.zeros((2 * num_pix - 1, 2 * num_pix - 1))

        for k1x in range(num_pix):
            for k2x in range(num_pix):
                for k1y in range(num_pix):
                    for k2y in range(num_pix):
                        g2_2d[num_pix - 1 + k1x - k2x, num_pix - 1 + k1y - k2y] += g2[
                            k1x, k1y, k2x, k2y
                        ]

        return g2_2d

    def get_g3(self, num_shots=1000):
        """Compute the third-order correlation function.

        Args:
            num_shots (int): number of shots to compute the correlation

        Returns:
            self.g3 (float): 6d array of the computed triple correlations
        """
        if self.g3 is not None:
            return self.g3

        print('Performing third-order correlation using outer product...')
        ave_intens = np.zeros(2 * (self.num_pix,))
        self.g3 = np.zeros(6 * (self.num_pix,))
        for i in range(num_shots):
            incoh = self.get_incoh_intens()
            self.g3 += np.multiply.outer(np.multiply.outer(incoh, incoh), incoh)
            ave_intens += incoh
        self.g3 *= num_shots**2 / np.multiply.outer(
            np.multiply.outer(ave_intens, ave_intens), ave_intens
        )
        print('Finished correlation...')
        return self.g3

    def marginalize_g3(self, num_shots=1000):
        """Reduce the dimensionality of the triple correlation by writing it
        as a function of q instead of k in reciprocal space.

        Keyword arguments:
            num_shots (int) - number of shots to compute the correlation

        Returns:
            self.g3_4d (float) - the dimension reduced version of the triple
            correlations
        """
        if self.g3 is None:
            self.g3 = self.get_g3(num_shots=num_shots)

        self.g3_4d = self.compute_marginalized_g3(self.g3, num_pix=self.num_pix)

        if self.weights_4d is None:
            self.init_weights_4d()
        self.g3_4d[self.weights_4d > 0] /= self.weights_4d[self.weights_4d > 0]

        return self.g3_4d

    @staticmethod
    @jit(nopython=True, parallel=False)
    def compute_marginalized_g3(g3, num_pix=1):
        """Compute the marginalized triple correlation function by summing the
        6D array of triple correlations.

        Args:
            g3 (ndarray): The 6D array of computed triple correlations.
            num_pix (int): The number of pixels for the simulation.

        Returns:
            g3_4d (ndarray): The dimension-reduced version of the triple
            correlation function.
        """
        g3_4d = np.zeros(
            (2 * num_pix - 1, 2 * num_pix - 1, 2 * num_pix - 1, 2 * num_pix - 1)
        )

        for k1x in range(num_pix):
            for k2x in range(num_pix):
                for k3x in range(num_pix):
                    for k1y in range(num_pix):
                        for k2y in range(num_pix):
                            for k3y in range(num_pix):
                                q1x = k1x - k2x + num_pix - 1
                                q1y = k1y - k2y + num_pix - 1
                                q2x = k2x - k3x + num_pix - 1
                                q2y = k2y - k3y + num_pix - 1
                                g3_4d[q1x, q1y, q2x, q2y] += g3[
                                    k1x, k1y, k2x, k2y, k3x, k3y
                                ]

        return g3_4d

    def closure_from_structure(self, return_phase=False):
        """Compute the closure from the structure coherent diffraction.

        Args:
            return_phase (bool): If True, return the closure phase instead
                of the closure magnitude

        Returns:
            numpy.ndarray: 4d array of the closure or the closure phase computed
                from the structure
        """
        pseudo_coh_ft_double = np.exp(
            -1j * (self.qr_product_x + self.qr_product_y)
        ).sum(2)
        coh_12 = np.multiply.outer(pseudo_coh_ft_double, pseudo_coh_ft_double)
        sum_q_x = -np.add.outer(self.q_pix[0, :, :], self.q_pix[0, :, :])
        sum_q_y = -np.add.outer(self.q_pix[1, :, :], self.q_pix[1, :, :])
        sumqr_product_x = np.multiply.outer(sum_q_x, self.coords[0, :])
        sumqr_product_y = np.multiply.outer(sum_q_y, self.coords[1, :])
        coh_1plus2 = np.exp(-1j * (sumqr_product_x + sumqr_product_y)).sum(4)

        if return_phase:
            return np.angle(coh_12 * coh_1plus2)
        else:
            if self.weights_4d is None:
                self.init_weights_4d()
            c = 2.0 * np.real(coh_12 * coh_1plus2)
            c = c / self.num_atoms**3 * (self.weights_4d > 0)
            return c

    def closure_from_data(self, num_shots=1000):
        """Compute the closure from correlations of incoherent fluorescence
        data.

        Args:
            num_shots (int): number of shots to compute the correlation

        Returns:
            (float): the 4d array of the closure computed from the
            correlations of the fluorescence data
        """
        if self.g3_4d is None:
            self.marginalize_g3(num_shots=num_shots)
        if self.g2_2d is None:
            self.marginalize_g2(num_shots=num_shots)

        g1sq = self.g2_2d - 1 + 1.0 / self.num_atoms
        dim = 2 * self.num_pix - 1
        qx, qy = np.indices(2 * (dim,))
        q12x = np.add.outer(qx, qx)
        q12x -= dim // 2
        q12x[(q12x < 0) | (q12x >= dim)] = 0
        q12y = np.add.outer(qy, qy)
        q12y -= dim // 2
        q12y[(q12y < 0) | (q12y >= dim)] = 0
        n = self.num_atoms

        weights = self.weights_4d

        c = (
            self.g3_4d
            - (1 - 3 / n + 4 / n**2)
            - (1 - 2 / n) * (np.add.outer(g1sq, g1sq) + g1sq[q12x, q12y])
        ) * (weights > 0)
        return c

    def cosPhi_from_structure(self) -> np.ndarray:
        """Get the cosine of the closure phase from the structure coherent
        diffraction.

        Returns:
            (float): the 4d array of the cosine of the closure phase computed
            from the structure
        """
        return np.cos(self.closure_from_structure(return_phase=True))

    def cosPhi_from_phase(self) -> np.ndarray:
        """Get the cosine of the closure phase from the true phase. Reverse model.

        Returns:
            numpy.ndarray: The cosine of the closure phase computed from the unknown
            phase.
        """
        true_phase = self.coh_phase_double[self.num_pix - 1 :, self.num_pix - 1 :]
        Phi = self.compute_Phi_from_phase(true_phase)
        cosPhi = np.cos(Phi)

        return cosPhi

    @staticmethod
    @jit(nopython=True, parallel=False)
    def compute_Phi_from_phase(phase):
        """Computes the phase difference array, Phi, for 2D phase data.

        Parameters:
            phase (ndarray): The 2D phase array, one quadrant of origin

        Returns:
            ndarray: The signed phase difference array in 4D
        """
        nx, ny = phase.shape
        assert nx == ny, 'Phase must be square'
        Phi = np.zeros((nx, ny, nx, ny))

        for nx_shift in range(nx):
            for ny_shift in range(ny):
                shifted_phase = roll2d(phase, -nx_shift, -ny_shift)
                Phi[nx_shift, ny_shift, :, :] = (
                    shifted_phase - phase - phase[nx_shift, ny_shift]
                )

        # Trim to match the expected output dimensions
        half_nx = nx // 2 + 1
        half_ny = ny // 2 + 1
        Phi = Phi[:half_nx, :half_ny, :half_nx, :half_ny]

        return Phi

    def cosPhi_from_data(self, num_shots=1000) -> np.ndarray:
        """Compute the cosine of the closure phase from correlations of
        incoherent fluorescence data.

        Args:
            num_shots (int): number of shots to compute the correlation

        Returns:
            (float): the 4d array of the cosine of the closure phase computed
            from the correlations of simulated fluorescence
        """
        clos = self.closure_from_data(num_shots=num_shots)
        clos = clos / 2

        # Remove magnitude of the g1 product
        g1sq = self.g2_2d - 1 + 1.0 / self.num_atoms
        g1sq[g1sq < 0] = 0.00000000001
        g1 = np.sqrt(g1sq)
        dim = 2 * self.num_pix - 1
        qx, qy = np.indices(2 * (dim,))
        q12x = np.add.outer(qx, qx)
        q12x -= dim // 2
        q12x[(q12x < 0) | (q12x >= dim)] = 0
        q12y = np.add.outer(qy, qy)
        q12y -= dim // 2
        q12y[(q12y < 0) | (q12y >= dim)] = 0

        clos = clos / (np.multiply.outer(g1, g1) * g1[q12x, q12y])
        clos[np.abs(clos) > 1] = np.sign(clos[np.abs(clos) > 1])

        cosPhi = clos

        return cosPhi
