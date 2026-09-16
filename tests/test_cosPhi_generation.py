import numpy as np
import pytest

from fluo import Fluorescence1D, Fluorescence2D

# Fixed seed so the Monte Carlo below is reproducible. Chosen once, arbitrarily,
# and NOT tuned until the assertions passed -- a seed picked for passing would
# hide exactly the noise this test is supposed to bound. A sweep over seeds
# 0-9 in 1D and 0-4 in 2D showed every seed converges to the analytic result
# once num_shots is large enough, so nothing here depends on this value.
SEED = 20240617

# Tolerance is deliberately left at the original rtol=0.2 / atol=0.1. The two
# tests used to fail in roughly 2 runs out of 3 not because that band is too
# tight but because num_shots was too small, so the fix is shots, not a wider
# band. See the module docstring note below on why the error is concentrated.
RTOL = 0.2
ATOL = 0.1

# Why the discrepancy is not uniform across the array: cosPhi_from_data divides
# the measured closure by the triple product g1[n] * g1[m] * g1[n + m]. Wherever
# the coherent structure factor has a near-zero, that divisor is tiny and shot
# noise in the numerator is amplified by ~1/g1. Measured for the 1D case at
# seed 7: |g1| = 0.048 at one pixel (the smallest of 41), and the row through
# that pixel carried ~7x the mean error of every other row. The estimator is
# unbiased -- the error falls off with num_shots -- it is just far noisier at
# those pixels than elsewhere, which is what made a small-num_shots run look
# like a systematic disagreement.


def test_cosPhi_1d():
    """Test cosPhi_from_phase and cosPhi_from_data produce similar results in 1D."""
    fluo = Fluorescence1D(kmax=3, num_pix=21, num_atoms=5, rng=SEED)

    # 200000 shots, not 50000: at 50000 the noisiest pixels (see note above)
    # still land ~0.4 away from the analytic value, which is what made this
    # test flaky. At 200000 the worst element is ~0.03.
    cosPhi_from_data = fluo.cosPhi_from_data(num_shots=200000)
    cosPhi_from_data = cosPhi_from_data[
        fluo.num_pix - 1 : 3 * fluo.num_pix // 2,
        fluo.num_pix - 1 : 3 * fluo.num_pix // 2,
    ]

    cosPhi_from_phase = fluo.cosPhi_from_phase()

    assert cosPhi_from_data.shape == cosPhi_from_phase.shape, (
        'Shapes do not match for 1D model'
    )

    np.testing.assert_allclose(
        cosPhi_from_data,
        cosPhi_from_phase,
        rtol=RTOL,
        atol=ATOL,
        err_msg='1D cosPhi values differ too much between methods',
    )


def test_cosPhi_2d():
    """Test cosPhi_from_phase and cosPhi_from_data produce similar results in 2D."""
    fluo = Fluorescence2D(kmax=3, num_pix=11, num_atoms=5, rng=SEED)

    cosPhi_from_phase = fluo.cosPhi_from_phase()

    # 30000 shots, not 10000. At 10000 two of five seeds tried failed outright
    # (worst element 0.76 and 0.50, 6/36 and 3/36 elements out of tolerance).
    # The 4d correlation makes this the expensive test (~5 min), so the count
    # is not raised further than needed: measured worst element is 0.089 here
    # against 0.122 at 15000 and 0.171 at 20000.
    cosPhi_from_data = fluo.cosPhi_from_data(num_shots=30000)
    cosPhi_from_data = cosPhi_from_data[
        fluo.num_pix - 1 : 3 * fluo.num_pix // 2,
        fluo.num_pix - 1 : 3 * fluo.num_pix // 2,
        fluo.num_pix - 1 : 3 * fluo.num_pix // 2,
        fluo.num_pix - 1 : 3 * fluo.num_pix // 2,
    ]

    assert cosPhi_from_data.shape == cosPhi_from_phase.shape, (
        'Shapes do not match for 2D model'
    )

    # Take a 2D slice for comparison (same as in plot_cosPhi)
    cosPhi_from_data = cosPhi_from_data[1, :, 1, :]
    cosPhi_from_phase = cosPhi_from_phase[1, :, 1, :]

    np.testing.assert_allclose(
        cosPhi_from_data,
        cosPhi_from_phase,
        rtol=RTOL,
        atol=ATOL,
        err_msg='2D cosPhi values differ too much between methods',
    )


@pytest.mark.parametrize(
    ('cls', 'kwargs'),
    [
        (Fluorescence1D, {'kmax': 3, 'num_pix': 21, 'num_atoms': 5}),
        (Fluorescence2D, {'kmax': 3, 'num_pix': 11, 'num_atoms': 5}),
    ],
)
def test_same_seed_reproduces_geometry_and_shot_noise(cls, kwargs):
    """Two instances given the same seed must agree on both the atom array and
    the per-shot emitter phases. This is the property the rng argument buys,
    and the one the tests above rely on.
    """
    a = cls(rng=SEED, **kwargs)
    b = cls(rng=SEED, **kwargs)

    np.testing.assert_array_equal(a.coords, b.coords)
    # A single shot exercises the other draw site, which is separate from the
    # one that placed the atoms.
    np.testing.assert_array_equal(a.get_incoh_intens(), b.get_incoh_intens())
