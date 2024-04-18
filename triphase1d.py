#!/usr/bin/env python

import numpy as np
from scipy import optimize
import h5py
import speckle1d


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
	cosPhi_sym = (cosPhi[num_pix - 1:3 * num_pix // 2,
				  num_pix - 1:3 * num_pix // 2]
				  + cosPhi[num_pix // 2:num_pix,
					num_pix // 2:num_pix][::-1, ::-1]) / 2
	Phi = np.arccos(cosPhi_sym)

	solved = np.zeros(num_pix // 2)
	solved[1] = initial_phase
	print(Phi[:, 2])
	for p in range(2, num_pix // 2):
		sum = 0
		for k in range(1, p):
			sum += Phi[k, 1]
		solved[p] = sum + p * solved[1]

	return solved


def PhiSolver(cosPhi, initial_phase=0):
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
	cosPhi_sym = (cosPhi[num_pix - 1:3 * num_pix // 2,
							 num_pix - 1:3 * num_pix // 2]
							 + cosPhi[num_pix // 2:num_pix,
							   num_pix // 2:num_pix][::-1,::-1]) / 2
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
		print("Pixel", n)
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

		next_phi, error_val = find_next_phi(xdata=xdata, ydata=ydata,
												 AltReturn=useAlt)
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
	print("QMAX LOOP")
	while n < num_pix // 2:
		print("Pixel", n + num_pix // 2)
		branches = np.zeros((int((num_pix // 2 - n + 3) / 2) - 1, 2))
		for m in range(1, int((num_pix // 2 - n + 3) / 2), 1):
			plus = Phi[num_pix // 2 - m + 1, m + n] + solved[
				num_pix // 2 - m + 1] + solved[m + n]
			minus = -Phi[num_pix // 2 - m + 1, m + n] + solved[
				num_pix // 2 - m + 1] + solved[m + n]
			branches[m - 1, 0] = plus
			branches[m - 1, 1] = minus

		theta1 = np.append(branches[:, 0], branches[:, 1])
		theta2 = np.append(branches[:, 1], branches[:, 0])
		xdata = np.cos(theta1)
		ydata = np.sin(theta2)

		next_phi, error_val = find_next_phi(xdata=xdata, ydata=ydata,
												 AltReturn=useAlt)
		solved[n + num_pix // 2 + 1] = next_phi
		error[n + num_pix // 2 + 1] = error_val

		if error[n + num_pix // 2 + 1] - error[
			n + num_pix // 2] > error_threshold:
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
			np.minimum((np.add.outer(xdata, -np.cos(theta)))**2,
					   (np.add.outer(ydata, -np.sin(theta)))**2).sum(0))

	def opt_func(theta):
		if np.abs(theta) > np.pi:
			return 1e10
		else:
			return np.log(np.sum(np.minimum((xdata - np.cos(theta))**2,
											(ydata - np.sin(theta))**2)))

	# Find candidate theta by doing brute force search of the 1D parameter space
	theta = np.linspace(-np.pi, np.pi, 50000)
	logThetaError = logThetaError(theta)
	num_theta = 2  # Number of candidates to accept. Two is optimal.
	mask = (np.argpartition(logThetaError, num_theta)[:num_theta])
	print("Possible Theta = ", theta[mask])
	theta0 = theta[mask]

	# Optimize candidate theta and choose the theta with smallest error
	fCandidate = []
	thetaCandidate = []
	for val in theta0:
		res = optimize.minimize(opt_func, x0=val, method='CG', tol=1e-10,
								options={'gtol': 1e-8, 'maxiter': 10000})
		fCandidate.append(res.fun)
		thetaCandidate.append(res.x)
	fCandidate = np.asarray(fCandidate)
	print("Error = ", fCandidate)
	thetaCandidate = np.asarray(thetaCandidate)
	thetaFinal = thetaCandidate[np.argmin(fCandidate)]
	fFinal = np.min(fCandidate)
	print("Final Theta = ", thetaFinal)

	if AltReturn:
		thetaFinal = thetaCandidate[np.argmax(fCandidate)]
		fFinal = np.max(fCandidate)
		print("Alternate Triggered!")
		print("Final Theta = ", thetaFinal)

	# Return ideal phi and the value of the error function at that phi
	return np.arctan2(np.sin(thetaFinal), np.cos(thetaFinal)), fFinal


def append_to_h5file(cosPhi_marginal, phase,
					 filename="data.h5"):
	"""Appends training data consisting of an image stack, the associated
	marginalized cosPhi, and the structure phase to a file.

	Keyword arguments:
		image_stack (float) - the stack of square detector images
		cosPhi_marginal (float) - the computed marginalized cosPhi array
		phase (float) - the target phase for the structure
		filename (string) - the output HDF5 file where the data is to be
		appended
	"""
	with h5py.File(filename, 'a') as f:
		# Create datasets if they don't exist, otherwise append data

		if "cosPhi_marginal" in f.keys():
			f["cosPhi_marginal"].resize(
				(f["cosPhi_marginal"].shape[0] + 1), axis=0)
			new_data = np.expand_dims(cosPhi_marginal, axis=0)
			f["cosPhi_marginal"][-1:] = new_data
		else:
			f.create_dataset("cosPhi_marginal",
							 data=np.expand_dims(cosPhi_marginal, axis=0),
							 maxshape=(None, cosPhi_marginal.shape[0],
									   cosPhi_marginal.shape[1]),
							 compression="gzip", compression_opts=9,
							 chunks=True)

		if "phase" in f.keys():
			f["phase"].resize((f["phase"].shape[0] + 1),
									 axis=0)
			new_data = np.expand_dims(phase, axis=0)
			f["phase"][-1:] = new_data
		else:
			f.create_dataset("phase",
							 data=np.expand_dims(phase, axis=0),
							 maxshape=(None, phase.shape[0]),
							 compression="gzip", compression_opts=9,
							 chunks=True)


def generate_training_set_from_data(num_data=1000,
						 file="/Users/nolanpeard/Desktop/train-k3-shot1000.h5",
						 image_stack_depth = 0):
	"""Generates training data and writes it to a file.

	Keyword arguments:
		num_data (int) - the number of data and label pairs to generate and
		export
		file (string) - the file path where the data is to be exported
		image_stack_depth (int) - the number of images that should be
		generated per stack in each data/label set
		"""
	for _ in range(num_data):
		fluo = Speckle_1D.Fluorescence1D(kmax=3, num_pix=51, num_atoms=
											np.random.random_integers(3, high=10))
		phase_target = fluo.coh_phase_double
		cosPhi_from_dataPhase = fluo.cosPhi_from_data(num_shots=1000)

		num_pix = int((cosPhi_from_dataPhase.shape[0] + 1) / 2)
		cosPhi_sym_1 = cosPhi_from_dataPhase[num_pix - 1:3 * num_pix // 2,
					  	num_pix - 1:3 * num_pix // 2]

		append_to_h5file(cosPhi_sym_1, phase_target,
						 filename=file)

	# Check that the file opens and contains data of the expected size
	with h5py.File(file, 'r') as f:
		cosPhi_marginal_data = f["cosPhi_marginal"][:]
		phase_data = f["phase"][:]

	print("cosPhi_marginal_data: ", cosPhi_marginal_data.shape)
	print("phase_data: ", phase_data.shape)

def generate_training_set_from_reverse(num_data=1000,
						 file="/Users/nolanpeard/Desktop/train-k3-shot1000.h5",
						 image_stack_depth = 0):
	"""Generates training data and writes it to a file.

	Keyword arguments:
		num_data (int) - the number of data and label pairs to generate and
		export
		file (string) - the file path where the data is to be exported
		image_stack_depth (int) - the number of images that should be
		generated per stack in each data/label set
		"""
	for _ in range(num_data):
		fluo = Speckle_1D.Fluorescence1D(kmax=3,
										 num_pix=51,
										 num_atoms=np.random.random_integers(3,
																			  high=20))
		phase_target = fluo.coh_phase_double
		cosPhi_from_phase = fluo.cosPhi_from_phase()

		append_to_h5file(cosPhi_from_phase, phase_target,
						 filename=file)

	# Check that the file opens and contains data of the expected size
	with h5py.File(file, 'r') as f:
		cosPhi_marginal_data = f["cosPhi_marginal"][:]
		phase_data = f["phase"][:]

	print("cosPhi_marginal_data: ", cosPhi_marginal_data.shape)
	print("phase_data: ", phase_data.shape)