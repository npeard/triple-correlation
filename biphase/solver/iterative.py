#!/usr/bin/env python

from itertools import permutations

import numpy as np
from scipy import optimize


class IterativeSolver:
    """Iterative phase retrieval solver for 1D and 2D fluorescence correlation data.

    Automatically determines dimensionality from input cosPhi array:
    - 2D array: 1D phase retrieval
    - 4D array: 2D phase retrieval
    """

    def __init__(self, cosPhi: np.ndarray):
        """Initialize solver with cosPhi correlation array.

        Args:
            cosPhi: Correlation array
                   - For 1D: 2D array of shape (2*num_pix-1, 2*num_pix-1)
                   - For 2D: 4D array of shape (2*num_pix-1, 2*num_pix-1, 2*num_pix-1, 2*num_pix-1)
        """
        self.cosPhi = cosPhi
        self.ndim = len(cosPhi.shape)

        if self.ndim == 2:
            self.is_1d = True
            self.num_pix = int((cosPhi.shape[0] + 1) / 2)
        elif self.ndim == 4:
            self.is_1d = False
            self.num_pix = int((cosPhi.shape[0] + 1) / 2)
        else:
            raise ValueError(
                f'cosPhi must be 2D (for 1D solve) or 4D (for 2D solve), got {self.ndim}D'
            )

    def naive_solve(self, initial_phase: float = 0) -> np.ndarray:
        """Simple phase solver without error correction.

        Args:
            initial_phase: Initial phase estimate
                          - For 1D: float (default 0)
                          - For 2D: list of 2 floats (default [0, 0])

        Returns:
            Solved phase array
        """
        if self.is_1d:
            return self._naive_solve_1d(initial_phase)
        else:
            raise NotImplementedError('2D naive solve not implemented in original code')

    def solve(
        self, initial_phase: float = 0, error_reject: float = -10
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full iterative phase solver with error correction.

        Args:
            initial_phase: Initial phase estimate
                          - For 1D: float (default 0)
                          - For 2D: list of 2 floats (default [0, 0])
            error_reject: Error threshold for rejecting solutions

        Returns:
            Tuple of (solved_phases, errors)
        """
        if self.is_1d:
            return self._solve_1d(initial_phase)
        else:
            return self._solve_2d([initial_phase, initial_phase], error_reject)

    def _symmetrize(self) -> np.ndarray:
        """Symmetrize and trim cosPhi array for processing.

        Returns:
            Symmetrized cosPhi array ready for arccos conversion
        """
        if self.is_1d:
            # 1D symmetrization - combine symmetric regions and average
            return (
                self.cosPhi[
                    self.num_pix - 1 : 3 * self.num_pix // 2,
                    self.num_pix - 1 : 3 * self.num_pix // 2,
                ]
                + self.cosPhi[
                    self.num_pix // 2 : self.num_pix, self.num_pix // 2 : self.num_pix
                ][::-1, ::-1]
            ) / 2
        else:
            # 2D symmetrization - combine symmetric 4D regions and average
            return (
                self.cosPhi[
                    self.num_pix - 1 : 2 * self.num_pix,
                    self.num_pix - 1 : 2 * self.num_pix,
                    self.num_pix - 1 : 2 * self.num_pix,
                    self.num_pix - 1 : 2 * self.num_pix,
                ]
                + self.cosPhi[
                    0 : self.num_pix,
                    0 : self.num_pix,
                    0 : self.num_pix,
                    0 : self.num_pix,
                ][::-1, ::-1, ::-1, ::-1]
            ) / 2

    def _naive_solve_1d(self, initial_phase: float) -> np.ndarray:
        """1D naive solver (equivalent to simple_PhiSolver)."""
        cosPhi_sym = self._symmetrize()
        Phi = np.arccos(cosPhi_sym)

        solved = np.zeros(self.num_pix // 2)
        solved[1] = initial_phase

        for p in range(2, self.num_pix // 2):
            sum_val = 0
            for k in range(1, p):
                sum_val += Phi[k, 1]
            solved[p] = sum_val + p * solved[1]

        return solved

    def _solve_1d(self, initial_phase: float) -> tuple[np.ndarray, np.ndarray]:
        """1D iterative solver with error correction."""
        cosPhi_sym = self._symmetrize()
        Phi = np.arccos(cosPhi_sym)

        # Initialize
        solved = np.zeros(self.num_pix)
        error = np.zeros_like(solved)
        solved[1] = initial_phase

        # Find phi out to KMAX
        error_threshold = 10
        n = 1
        useAlt = False

        while n < self.num_pix // 2:
            branches = np.zeros((int((n + 3) / 2) - 1, 2))
            for m in range(1, int((n + 3) / 2), 1):
                plus = Phi[n - m + 1, m] + solved[n - m + 1] + solved[m]
                minus = -Phi[n - m + 1, m] + solved[n - m + 1] + solved[m]
                branches[m - 1, 0] = plus
                branches[m - 1, 1] = minus

            theta1 = np.append(branches[:, 0], branches[:, 1])
            theta2 = np.append(branches[:, 1], branches[:, 0])
            xdata = np.cos(theta1)
            ydata = np.sin(theta2)

            next_phi, error_val = self._find_next_phi(
                xdata=xdata, ydata=ydata, AltReturn=useAlt
            )
            solved[n + 1] = next_phi
            error[n + 1] = error_val

            if error[n + 1] - error[n] > error_threshold:
                n -= 1
                useAlt = True
            else:
                useAlt = False
                n += 1

        # Find phi out to QMAX
        n = 0
        while n < self.num_pix // 2:
            branches = np.zeros((int((self.num_pix // 2 - n + 3) / 2) - 1, 2))
            for m in range(1, int((self.num_pix // 2 - n + 3) / 2), 1):
                plus = (
                    Phi[self.num_pix // 2 - m + 1, m + n]
                    + solved[self.num_pix // 2 - m + 1]
                    + solved[m + n]
                )
                minus = (
                    -Phi[self.num_pix // 2 - m + 1, m + n]
                    + solved[self.num_pix // 2 - m + 1]
                    + solved[m + n]
                )
                branches[m - 1, 0] = plus
                branches[m - 1, 1] = minus

            theta1 = np.append(branches[:, 0], branches[:, 1])
            theta2 = np.append(branches[:, 1], branches[:, 0])
            xdata = np.cos(theta1)
            ydata = np.sin(theta2)

            next_phi, error_val = self._find_next_phi(
                xdata=xdata, ydata=ydata, AltReturn=useAlt
            )
            solved[n + self.num_pix // 2 + 1] = next_phi
            error[n + self.num_pix // 2 + 1] = error_val

            if (
                error[n + self.num_pix // 2 + 1] - error[n + self.num_pix // 2]
                > error_threshold
            ):
                n -= 1
                useAlt = True
            else:
                useAlt = False
                n += 1

        return solved, error

    def _solve_2d(
        self, initial_phase: list, error_reject: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """2D iterative solver with error correction."""
        cosPhi_sym = self._symmetrize()
        Phi = np.arccos(cosPhi_sym)
        solved = np.zeros(2 * (self.num_pix,))
        solved[0, 1] = initial_phase[0]
        solved[1, 0] = initial_phase[1]

        error = np.zeros_like(solved)

        n = 3
        diagonal_flag = 0
        suspect_num = -1
        num_pixels = 1
        perm_num = -1
        perm = np.zeros(self.num_pix)

        while n < len(Phi[0, 0, 0, :]) + 1:
            # Generate list of points across the diagonal to be solved this round
            to_solve_1 = np.arange(n)
            to_solve_2 = to_solve_1[::-1]
            to_solve = np.asarray([to_solve_1, to_solve_2])

            for m in range(len(to_solve[0, :])):
                current_pair = to_solve[:, m]
                # Generate matrix of indices which fill the box defined by the
                # origin and our current point
                A = np.indices((current_pair[0] + 1, current_pair[1] + 1))
                B = np.indices((current_pair[0] + 1, current_pair[1] + 1))
                B[0, :, :] = current_pair[0] - B[0, :, :]
                B[1, :, :] = current_pair[1] - B[1, :, :]

                # Flatten and remove trivial pairs
                A = A.reshape((2, -1))
                B = B.reshape((2, -1))
                A = A[:, 1:-1]
                B = B[:, 1:-1]

                plus = np.empty(len(A[0, :]))
                minus = np.empty(len(A[0, :]))
                for i in range(len(A[0, :])):
                    plus[i] = (
                        Phi[A[0, i], A[1, i], B[0, i], B[1, i]]
                        + solved[A[0, i], A[1, i]]
                        + solved[B[0, i], B[1, i]]
                    )
                    minus[i] = (
                        -Phi[A[0, i], A[1, i], B[0, i], B[1, i]]
                        + solved[A[0, i], A[1, i]]
                        + solved[B[0, i], B[1, i]]
                    )

                theta1 = np.append(plus, minus)
                theta2 = np.append(minus, plus)
                xdata = np.cos(theta1)
                ydata = np.sin(theta2)

                # Use alternate value if flagged
                if diagonal_flag == n + 1 and perm[m] == 1:
                    next_phi, error_val = self._find_next_phi(
                        xdata=xdata, ydata=ydata, AltReturn=True
                    )
                else:
                    next_phi, error_val = self._find_next_phi(xdata=xdata, ydata=ydata)

                solved[current_pair[0], current_pair[1]] = next_phi
                error[current_pair[0], current_pair[1]] = error_val

            # Error rejection logic
            if np.any(error[to_solve[0, :], to_solve[1, :]] > error_reject):
                diagonal_flag = n
                err_idx = np.argmax(error[to_solve[0, :], to_solve[1, :]])
                suspects = np.zeros((4, diagonal_flag - 0))
                suspects[0, err_idx] = 1
                suspects[1, err_idx - 1] = 1
                suspects[2, err_idx - 1 : err_idx + 1] = 1
                suspect_num += 1
                perm = suspects[suspect_num, :]

                if suspect_num > 2:
                    suspect_num = 2
                    elements = np.zeros(diagonal_flag - 1)
                    elements[:num_pixels] = 1
                    perms = np.asarray(list(set(permutations(elements))))
                    perm_num += 1
                    if perm_num >= len(perms[:, 0]):
                        num_pixels += 1
                        elements[:num_pixels] = 1
                        perms = np.asarray(list(set(permutations(elements))))
                        perm_num = 0
                    if num_pixels > len(elements):
                        raise RuntimeError(
                            'All permutations tested but error threshold still exceeded'
                        )
                    perm = perms[perm_num, :]
                n -= 2
            elif diagonal_flag == n:
                diagonal_flag = 0
                suspect_num = -1
                perm_num = -1
                perm = np.zeros(self.num_pix)

            n += 1

        # Solve out to q_max
        for n in range(1, len(Phi[0, 0, 0, :])):
            to_solve_1 = np.arange(len(Phi[0, 0, 0, :]) - n) + n
            to_solve_2 = to_solve_1[::-1]
            to_solve = np.asarray([to_solve_1, to_solve_2])

            for m in range(len(to_solve[0, :])):
                current_pair = to_solve[:, m]
                A = np.mgrid[0 : current_pair[0] + 1, 0 : current_pair[1] + 1]
                B = np.mgrid[0 : current_pair[0] + 1, 0 : current_pair[1] + 1]
                B[0, :, :] = current_pair[0] - B[0, :, :]
                B[1, :, :] = current_pair[1] - B[1, :, :]

                A = A.reshape((2, -1))
                B = B.reshape((2, -1))
                A = A[:, 1:-1]
                B = B[:, 1:-1]

                plus = np.empty(len(A[0, :]))
                minus = np.empty(len(A[0, :]))
                for i in range(len(A[0, :])):
                    plus[i] = (
                        Phi[A[0, i], A[1, i], B[0, i], B[1, i]]
                        + solved[A[0, i], A[1, i]]
                        + solved[B[0, i], B[1, i]]
                    )
                    minus[i] = (
                        -Phi[A[0, i], A[1, i], B[0, i], B[1, i]]
                        + solved[A[0, i], A[1, i]]
                        + solved[B[0, i], B[1, i]]
                    )

                theta1 = np.append(plus, minus)
                theta2 = np.append(minus, plus)
                xdata = np.cos(theta1)
                ydata = np.sin(theta2)

                next_phi, error_val = self._find_next_phi(xdata=xdata, ydata=ydata)
                solved[current_pair[0], current_pair[1]] = next_phi
                error[current_pair[0], current_pair[1]] = error_val

        return solved, error

    def _find_next_phi(
        self, xdata: np.ndarray, ydata: np.ndarray, AltReturn: bool = False
    ) -> tuple[float, float]:
        """Find the next phase value by optimizing theta candidates.

        Args:
            xdata: Cosine of candidate theta values
            ydata: Sine of candidate theta values
            AltReturn: If True, return alternate (less optimal) solution

        Returns:
            Tuple of (optimal_phi, error_value)
        """

        def logThetaError(theta: float) -> float:
            return np.log(
                np.minimum(
                    (np.add.outer(xdata, -np.cos(theta))) ** 2,
                    (np.add.outer(ydata, -np.sin(theta))) ** 2,
                ).sum(0)
            )

        def opt_func(theta: float) -> float:
            if np.abs(theta) > np.pi:
                return 1e10
            else:
                return np.log(
                    np.sum(
                        np.minimum(
                            (xdata - np.cos(theta)) ** 2, (ydata - np.sin(theta)) ** 2
                        )
                    )
                )

        # Brute force search for candidates
        theta = np.linspace(-np.pi, np.pi, 50000)
        logThetaError_vals = logThetaError(theta)
        num_theta = 2
        mask = np.argpartition(logThetaError_vals, num_theta)[:num_theta]
        theta0 = theta[mask]

        # Optimize candidates
        fCandidate = []
        thetaCandidate = []
        for val in theta0:
            res = optimize.minimize(
                opt_func,
                x0=val,
                method='CG',
                tol=1e-10,
                options={'gtol': 1e-8, 'maxiter': 10000},
            )
            fCandidate.append(res.fun)
            thetaCandidate.append(res.x)

        fCandidate = np.asarray(fCandidate)
        thetaCandidate = np.asarray(thetaCandidate)

        if AltReturn:
            thetaFinal = thetaCandidate[np.argmax(fCandidate)]
            fFinal = np.max(fCandidate)
        else:
            thetaFinal = thetaCandidate[np.argmin(fCandidate)]
            fFinal = np.min(fCandidate)

        return np.arctan2(np.sin(thetaFinal), np.cos(thetaFinal)), fFinal
