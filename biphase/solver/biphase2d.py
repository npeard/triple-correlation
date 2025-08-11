#!/usr/bin/env python

from itertools import permutations
import logging

import numpy as np
from scipy import optimize

logger = logging.getLogger(__name__)


def PhiSolver(cosPhi, initial_phase=None, error_reject=-10):  # noqa: PLR0915,PLR0912,C901
    """Solves the phase for a given cosPhi from data and guesstimate of the
    first phase value. Uses all rows of Phi to solve the sign problem and
    obtain the correct phase slope.

    Keyword arguments:
            cosPhi (float) - 2D NumPy array, 2*num_pix-1 to an edge, contains the
            phase information to be retrieved. Usually should be computed using
            the "cosPhi_from_data" method in the Fluorescence_1D class

            initial_phase (float) - estimated value of the first pixel of phase
            to be retrieved. Accuracy of this estimate determines fidelity of
            phase retrieval.

    Returns:
            solved (float) - the solved phases out to qmax

            error (float) - the error values associated with the solved phases
    """
    num_pix = int((cosPhi.shape[0] + 1) / 2)
    cosPhi_sym = (
        cosPhi[
            num_pix - 1 : 2 * num_pix,
            num_pix - 1 : 2 * num_pix,
            num_pix - 1 : 2 * num_pix,
            num_pix - 1 : 2 * num_pix,
        ]
        + cosPhi[0:num_pix, 0:num_pix, 0:num_pix, 0:num_pix][::-1, ::-1, ::-1, ::-1]
    ) / 2

    if initial_phase is None:
        initial_phase = [0, 0]

    Phi = np.arccos(cosPhi_sym)
    solved = np.zeros(2 * (num_pix,))
    solved[0, 1] = initial_phase[0]
    solved[1, 0] = initial_phase[1]

    error = np.zeros_like(solved)

    n = 3
    diagonal_flag = 0
    suspect_num = (
        -1
    )  # Index for list of suspect pixels to be picked as alternates in re-solving
    num_pixels = 1  # To re-solve
    perm_num = (
        -1
    )  # The index in the list of permutations to use for alternates in re-solving
    perm = np.zeros(num_pix)
    while n < len(Phi[0, 0, 0, :]) + 1:
        # Generate list of points across the diagonal to be solved this round
        prev_solve_1 = np.arange(n - 1)
        prev_solve_2 = prev_solve_1[::-1]
        prev_solve = np.asarray([prev_solve_1, prev_solve_2])

        to_solve_1 = np.arange(n)
        to_solve_2 = to_solve_1[::-1]
        to_solve = np.asarray([to_solve_1, to_solve_2])

        for m in range(len(to_solve[0, :])):
            current_pair = to_solve[:, m]
            # Generate matrix of indices which fill the box defined by the origin and our current point
            # Find pairs of vectors which span the box and sum to the current
            # vector
            A = np.indices((current_pair[0] + 1, current_pair[1] + 1))
            B = np.indices((current_pair[0] + 1, current_pair[1] + 1))
            B[0, :, :] = current_pair[0] - B[0, :, :]
            B[1, :, :] = current_pair[1] - B[1, :, :]
            # Flatten in to list of pairs and remove trivial (0,0) + (n,m)
            # pairs
            A = A.reshape((2, -1))
            B = B.reshape((2, -1))
            A = A[:, 1:-1]
            B = B[:, 1:-1]

            plus = np.empty(len(A[0, :]))
            minus = np.empty(len(A[0, :]))
            for i in range(len(A[0, :])):
                # Find the positive and negative solutions
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

            logger.debug('Processing pixel pair: %s', current_pair)
            # If error flag has been triggered for the next diagonal, use the alternate value for trial positions
            # next_phi, error_val = self.find_next_phi(xdata=xdata, ydata=ydata)
            if diagonal_flag == n + 1 and perm[m] == 1:
                next_phi, error_val = find_next_phi(
                    xdata=xdata, ydata=ydata, AltReturn=True
                )
            else:
                next_phi, error_val = find_next_phi(xdata=xdata, ydata=ydata)

            solved[current_pair[0], current_pair[1]] = next_phi
            error[current_pair[0], current_pair[1]] = error_val

        # Loop mechanics
        # Reject any solution with a pixel that has error above error_reject
        if np.any(error[to_solve[0, :], to_solve[1, :]] > error_reject):
            # if (np.any( np.abs(np.subtract.outer(error[to_solve[0,:],
            # to_solve[1,:]], error[prev_solve[0,:], prev_solve[1,:]])) > 15)
            # and n>3):
            logger.debug('Previous errors: %s', error[prev_solve[0, :], prev_solve[1, :]])
            logger.debug('Current errors: %s', error[to_solve[0, :], to_solve[1, :]])
            logger.debug('Error differences: %s', np.abs(
                np.subtract.outer(
                    error[to_solve[0, :], to_solve[1, :]],
                    error[prev_solve[0, :], prev_solve[1, :]],
                )
            ))
            diagonal_flag = n
            logger.warning('Unacceptable error! Re-solving previous diagonal')
            # First, attempt to change pixels adjacent to pixel in current
            # diagonal with the largest error
            logger.debug('Current pixel errors: %s', error[to_solve[0, :], to_solve[1, :]])
            err_idx = np.argmax(error[to_solve[0, :], to_solve[1, :]])
            suspects = np.zeros((4, diagonal_flag - 0))  # The fourth row is
            # just a dummy case, only need 3 permutations for a 1 pixel error
            suspects[0, err_idx] = 1
            suspects[1, err_idx - 1] = 1
            suspects[2, err_idx - 1 : err_idx + 1] = 1
            suspect_num += 1
            logger.debug('Suspect pixels: %s', suspects)
            perm = suspects[suspect_num, :]

            # But if that fails, sort through all possible permutations
            if suspect_num > 2:
                suspect_num = 2
                elements = np.zeros(diagonal_flag - 1)
                elements[:num_pixels] = 1
                perms = np.asarray(list(set(permutations(elements))))
                perm_num += 1
                if perm_num >= len(perms[:, 0]):
                    logger.info('Adding additional pixel to re-solve')
                    num_pixels += 1
                    elements[:num_pixels] = 1
                    perms = np.asarray(list(set(permutations(elements))))
                    perm_num = 0
                # In case we have already been through every possible
                # permutation and still not met the error threshold
                if num_pixels > len(elements):
                    logger.critical('CRITICAL ERROR: Every possible permutation of alternate theta has been tested but the error threshold is still exceeded')
                    logger.critical('The error threshold is either too stringent or user intervention is needed')
                    # Then, go back to the default case (no alternates) and proceed anyways.
                    # For now, just exit.
                    import sys

                    sys.exit(1)

                logger.debug('Permutations: %s', perms)
                perm = perms[perm_num, :]
            n -= 2  # This is outside the "if suspect_num > 2:" statement
        elif diagonal_flag == n:
            diagonal_flag = 0
            suspect_num = -1
            perm_num = -1
            perm = np.zeros(num_pix)
        logger.debug('suspect_num: %d, perm_num: %d', suspect_num, perm_num)
        n += 1

    # Solve out to q_max, at this point error resolving should not be needed
    for n in range(1, len(Phi[0, 0, 0, :])):
        # Generate list of points across the diagonal to be solved this round
        to_solve_1 = np.arange(len(Phi[0, 0, 0, :]) - n) + n
        to_solve_2 = to_solve_1[::-1]

        to_solve = np.asarray([to_solve_1, to_solve_2])

        for m in range(len(to_solve[0, :])):
            current_pair = to_solve[:, m]
            # Generate matrix of indices which fill the box defined by the origin and our current point
            # Find pairs of vectors which span the box and sum to the current vector
            # current_pair[np.argmin(current_pair)] += 1
            # current_pair[np.argmax(current_pair)] -=1
            # A = np.indices(current_pair)
            # B = np.indices(current_pair)
            A = np.mgrid[0 : current_pair[0] + 1, 0 : current_pair[1] + 1]
            B = np.mgrid[0 : current_pair[0] + 1, 0 : current_pair[1] + 1]
            B[0, :, :] = current_pair[0] - B[0, :, :]
            B[1, :, :] = current_pair[1] - B[1, :, :]
            # Flatten in to list of pairs and remove trivial (0,0) + (n,m)
            # pairs
            A = A.reshape((2, -1))
            B = B.reshape((2, -1))
            A = A[:, 1:-1]
            B = B[:, 1:-1]

            plus = np.empty(len(A[0, :]))
            minus = np.empty(len(A[0, :]))
            for i in range(len(A[0, :])):
                # Find the positive and negative solutions
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

            logger.debug('Processing pixel pair: %s', current_pair)
            next_phi, error_val = find_next_phi(xdata=xdata, ydata=ydata)

            solved[current_pair[0], current_pair[1]] = next_phi
            error[current_pair[0], current_pair[1]] = error_val

    return solved, error


def PhiSolver_manualSelect(cosPhi, initial_phase=None, Alt=None):  # noqa: PLR0915
    """Solves the phase for a given cosPhi from data and guesstimate of the
    first phase value. Uses all rows of Phi to solve the sign problem and
    obtain the correct phase slope. Here, the user selects where resolving is used.

    Keyword arguments:
            cosPhi (float) - 2D NumPy array, 2*num_pix-1 to an edge, contains the
            phase information to be retrieved. Usually should be computed using
            the "cosPhi_from_data" method in the Fluorescence_1D class

            initial_phase (float) - estimated value of the first pixel of phase
            to be retrieved. Accuracy of this estimate determines fidelity of
            phase retrieval.

    Returns:
            solved (float) - the solved phases out to qmax

            error (float) - the error values associated with the solved phases
    """
    num_pix = int(cosPhi.shape[0])
    # cosPhi_sym = (cosPhi[num_pix - 1:2 * num_pix, num_pix - 1:2 * num_pix,
    # 			  num_pix - 1:2 * num_pix, num_pix - 1:2 * num_pix]
    # 			  + cosPhi[0:num_pix, 0:num_pix, 0:num_pix, 0:num_pix][::-1,
    # 				::-1, ::-1, ::-1]) / 2
    if initial_phase is None:
        initial_phase = [0, 0]

    Phi = np.arccos(cosPhi)

    solved = np.zeros(2 * (num_pix,))
    solved[0, 1] = initial_phase[0]
    solved[1, 0] = initial_phase[1]

    error = np.zeros_like(solved)

    n = 3
    while n < len(Phi[0, 0, 0, :]) + 1:
        # Generate list of points across the diagonal to be solved this round
        to_solve_1 = np.arange(n)
        to_solve_2 = to_solve_1[::-1]
        to_solve = np.asarray([to_solve_1, to_solve_2])

        for m in range(len(to_solve[0, :])):
            current_pair = to_solve[:, m]
            # Generate matrix of indices which fill the box defined by the
            # origin and our current point
            # Find pairs of vectors which span the box and sum to the
            # current vector
            A = np.indices((current_pair[0] + 1, current_pair[1] + 1))
            B = np.indices((current_pair[0] + 1, current_pair[1] + 1))
            B[0, :, :] = current_pair[0] - B[0, :, :]
            B[1, :, :] = current_pair[1] - B[1, :, :]
            # Flatten in to list of pairs and remove trivial (0,0) + (n,m)
            # pairs
            A = A.reshape((2, -1))
            B = B.reshape((2, -1))
            A = A[:, 1:-1]
            B = B[:, 1:-1]

            plus = np.empty(len(A[0, :]))
            minus = np.empty(len(A[0, :]))
            for i in range(len(A[0, :])):
                # Find the positive and negative solutions
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

            logger.debug('Processing pixel pair: %s', current_pair)
            # If an alternate has been requested by the user for the pixel,
            # choose the other value
            if Alt[current_pair[0], current_pair[1]] == 1:
                next_phi, error_val = find_next_phi(
                    xdata=xdata, ydata=ydata, AltReturn=True
                )
            else:
                next_phi, error_val = find_next_phi(xdata=xdata, ydata=ydata)

            solved[current_pair[0], current_pair[1]] = next_phi
            error[current_pair[0], current_pair[1]] = error_val
        n += 1

    # Solve phase out to q_max, at this point error resolving should not be
    # needed
    for n in range(1, len(Phi[0, 0, 0, :])):
        # Generate list of points across the diagonal to be solved this round
        to_solve_1 = np.arange(len(Phi[0, 0, 0, :]) - n) + n
        to_solve_2 = to_solve_1[::-1]

        to_solve = np.asarray([to_solve_1, to_solve_2])

        for m in range(len(to_solve[0, :])):
            current_pair = to_solve[:, m]
            # Generate matrix of indices which fill the box defined by the
            # origin and our current point
            # Find pairs of vectors which span the box and sum to the
            # current vector
            A = np.mgrid[0 : current_pair[0] + 1, 0 : current_pair[1] + 1]
            B = np.mgrid[0 : current_pair[0] + 1, 0 : current_pair[1] + 1]
            B[0, :, :] = current_pair[0] - B[0, :, :]
            B[1, :, :] = current_pair[1] - B[1, :, :]
            # Flatten in to list of pairs and remove trivial (0,0) + (n,m)
            # pairs
            A = A.reshape((2, -1))
            B = B.reshape((2, -1))
            A = A[:, 1:-1]
            B = B[:, 1:-1]

            plus = np.empty(len(A[0, :]))
            minus = np.empty(len(A[0, :]))
            for i in range(len(A[0, :])):
                # Find the positive and negative solutions
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

            logger.debug('Processing pixel pair: %s', current_pair)
            next_phi, error_val = find_next_phi(xdata=xdata, ydata=ydata)

            solved[current_pair[0], current_pair[1]] = next_phi
            error[current_pair[0], current_pair[1]] = error_val

    return solved, error


def find_next_phi(xdata=None, ydata=None, AltReturn=False):
    """Finds the nearest intersection of sets of possible theta by finding
    the pairs of vertical and horizontal lines that best fit the data when
    plotted as ordered pairs in the xy-plane.

    Keyword arguments:
            xdata (float) - cosine of the candidate theta value array
            ydata (float) - sine of the candidate theta value array
            AltReturn (bool) - when True, returns the alternate (less optimal)
            value of theta that fits the data

    Returns:
            (float) - the optimal value for the next value of the phase,
            given the input arguments

            fFinal (float) - the value of the error computed for that optimal
            phase value
    """

    # Samples the error function and starts minimization near the minimum
    def logThetaError(theta):
        return np.log(
            np.minimum(
                (np.add.outer(xdata, -np.cos(theta))) ** 2,
                (np.add.outer(ydata, -np.sin(theta))) ** 2,
            ).sum(0)
        )

    def opt_func(theta):
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

    # Find candidate theta by doing brute force search of the 1D parameter
    # space
    theta = np.linspace(-np.pi, np.pi, 50000)
    logThetaError = logThetaError(theta)
    num_theta = 2  # Number of candidates to accept. Two is optimal.
    mask = np.argpartition(logThetaError, num_theta)[:num_theta]
    logger.debug('Possible Theta = %s', theta[mask])
    theta0 = theta[mask]

    # Optimize candidate theta and choose the theta with smallest error
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
    logger.debug('Error = %s', fCandidate)
    thetaCandidate = np.asarray(thetaCandidate)
    thetaFinal = thetaCandidate[np.argmin(fCandidate)]
    fFinal = np.min(fCandidate)
    logger.debug('Final Theta = %s', thetaFinal)

    if AltReturn:
        thetaFinal = thetaCandidate[np.argmax(fCandidate)]
        fFinal = np.max(fCandidate)
        logger.info('Alternate triggered!')
        logger.debug('Final Theta = %s', thetaFinal)

    # Return ideal phi and the value of the error function at that phi
    return np.arctan2(np.sin(thetaFinal), np.cos(thetaFinal)), fFinal
