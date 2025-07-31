#!/usr/bin/env python

import numpy as np
from scipy import optimize


def simple_PhiSolver(cosPhi, initial_phase=0):
    """Solves the phase for a given cosPhi from data and guesstimate of the
    first phase value, but does not use all rows of Phi to solve the sign
    problem.

    Keyword arguments:
            cosPhi (float) - 2D NumPy array, 2*num_pix-1 to an edge, contains the
            phase information to be retrieved. Usually should be computed using
            the "cosPhi_from_data" method in the Fluorescence_1D class

            initial_phase (float) - estimated value of the first pixel of phase
            to be retrieved. Accuracy of this estimate determines fidelity of
            phase retrieval.

    Returns:
            solved (float) - the solved phases in a 1d array out to qmax
    """
    # Taking into account only "singles and doubles"
    num_pix = int((cosPhi.shape[0] + 1) / 2)
    cosPhi_sym = (
        cosPhi[num_pix - 1 : 3 * num_pix // 2, num_pix - 1 : 3 * num_pix // 2]
        + cosPhi[num_pix // 2 : num_pix, num_pix // 2 : num_pix][::-1, ::-1]
    ) / 2
    Phi = np.arccos(cosPhi_sym)

    solved = np.zeros(num_pix // 2)
    solved[1] = initial_phase
    # print(Phi[:, 2])
    for p in range(2, num_pix // 2):
        sum = 0
        for k in range(1, p):
            sum += Phi[k, 1]
        solved[p] = sum + p * solved[1]

    return solved


def PhiSolver(cosPhi, initial_phase=0):  # noqa: PLR0915
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
        cosPhi[num_pix - 1 : 3 * num_pix // 2, num_pix - 1 : 3 * num_pix // 2]
        + cosPhi[num_pix // 2 : num_pix, num_pix // 2 : num_pix][::-1, ::-1]
    ) / 2
    Phi = np.arccos(cosPhi_sym)

    # PHI SOLVER ALGORITHM
    # Initial conditions
    solved = np.zeros(num_pix)
    error = np.zeros_like(solved)
    solved[1] = initial_phase

    # Find phi out to KMAX
    error_threshold = 10
    n = 1
    useAlt = False
    while n < num_pix // 2:
        print('Pixel', n)
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

        next_phi, error_val = find_next_phi(xdata=xdata, ydata=ydata, AltReturn=useAlt)
        solved[n + 1] = next_phi
        error[n + 1] = error_val

        if error[n + 1] - error[n] > error_threshold:
            n -= 1
            useAlt = True
        else:
            useAlt = False
            n += 1

    # Find phi out to QMAX
    error_threshold = 10
    n = 0
    # for n in range(0, self.num_pix//2, 1):
    print('QMAX LOOP')
    while n < num_pix // 2:
        print('Pixel', n + num_pix // 2)
        branches = np.zeros((int((num_pix // 2 - n + 3) / 2) - 1, 2))
        for m in range(1, int((num_pix // 2 - n + 3) / 2), 1):
            plus = (
                Phi[num_pix // 2 - m + 1, m + n]
                + solved[num_pix // 2 - m + 1]
                + solved[m + n]
            )
            minus = (
                -Phi[num_pix // 2 - m + 1, m + n]
                + solved[num_pix // 2 - m + 1]
                + solved[m + n]
            )
            branches[m - 1, 0] = plus
            branches[m - 1, 1] = minus

        theta1 = np.append(branches[:, 0], branches[:, 1])
        theta2 = np.append(branches[:, 1], branches[:, 0])
        xdata = np.cos(theta1)
        ydata = np.sin(theta2)

        next_phi, error_val = find_next_phi(xdata=xdata, ydata=ydata, AltReturn=useAlt)
        solved[n + num_pix // 2 + 1] = next_phi
        error[n + num_pix // 2 + 1] = error_val

        if error[n + num_pix // 2 + 1] - error[n + num_pix // 2] > error_threshold:
            n -= 1
            useAlt = True
        else:
            useAlt = False
            n += 1

    # Return solved branches
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
    print('Possible Theta = ', theta[mask])
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
    print('Error = ', fCandidate)
    thetaCandidate = np.asarray(thetaCandidate)
    thetaFinal = thetaCandidate[np.argmin(fCandidate)]
    fFinal = np.min(fCandidate)
    print('Final Theta = ', thetaFinal)

    if AltReturn:
        thetaFinal = thetaCandidate[np.argmax(fCandidate)]
        fFinal = np.max(fCandidate)
        print('Alternate Triggered!')
        print('Final Theta = ', thetaFinal)

    # Return ideal phi and the value of the error function at that phi
    return np.arctan2(np.sin(thetaFinal), np.cos(thetaFinal)), fFinal
