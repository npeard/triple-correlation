#!/usr/bin/env python

import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from biphase.solver.differentiable import DifferentiableSolver
from biphase.transformer.lightning_decoder import GPTDecoder
from fluo import Fluorescence1D, Fluorescence2D


def test_1d_solver():
    """Test the 1D differentiable solver with synthetic data."""
    logger.info('Testing 1D DifferentiableSolver...')
    num_pix = 31
    fluo = Fluorescence1D(kmax=5, num_pix=num_pix, num_atoms=4)
    true_phase = fluo.coh_phase_double[fluo.num_pix - 1 :]
    cosPhi = fluo.cosPhi_from_data(num_shots=10000)

    # Initialize solver
    solver = DifferentiableSolver(cosPhi, epochs=500, lr=0.01, log_level='INFO')

    # Solve
    phase, loss_history = solver.solve()

    logger.debug('Final phase shape: %s', phase.shape)
    logger.debug('Final loss: %.6f', loss_history[-1])
    logger.debug('Loss decreased: %s', loss_history[0] > loss_history[-1])

    # Plot results
    fig, ((ax1, ax2, ax3), (ax4, ax5, _)) = plt.subplots(2, 3, figsize=(15, 8))

    # Plot true phase
    ax1.plot(true_phase, 'b-o', label='True Phase')
    ax1.set_title('True Phase (1D)')
    ax1.set_xlabel('Pixel Index')
    ax1.set_ylabel('Phase')
    ax1.grid(True)
    ax1.legend()

    # Plot solved phase
    solved_phase_np = phase.cpu().numpy()
    ax2.plot(solved_phase_np, 'r-s', label='Solved Phase')
    ax2.set_title('Solved Phase (1D)')
    ax2.set_xlabel('Pixel Index')
    ax2.set_ylabel('Phase')
    ax2.grid(True)
    ax2.legend()

    # Plot loss history
    ax3.semilogy(loss_history, 'g-')
    ax3.set_title('Training Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.grid(True)

    # Plot DifferentiableSolver target
    Phi_dim = int((num_pix + 1) / 2) - 1
    ax4.imshow(
        solver.target.cpu().numpy().reshape((Phi_dim, Phi_dim)),
        cmap='viridis',
        interpolation='nearest',
        origin='lower',
    )
    ax4.set_title('Target Phase (1D)')
    ax4.set_xlabel('Pixel Index')
    ax4.set_ylabel('Pixel Index')
    ax4.grid(True)

    # Plot DifferentiableSolver phase re-encoded
    encoded = (
        GPTDecoder._encode(solver.trainable_phase.phase.unsqueeze(0))
        .detach()
        .numpy()
        .reshape((Phi_dim, Phi_dim))
    )
    ax5.imshow(encoded, cmap='viridis', interpolation='nearest', origin='lower')
    ax5.set_title('Re-encoded Phase (1D)')
    ax5.set_xlabel('Pixel Index')
    ax5.set_ylabel('Pixel Index')
    ax5.grid(True)

    plt.tight_layout()
    # plt.savefig('1d_differentiable_solver_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    logger.info('1D test completed successfully!')
    return true_phase, solved_phase_np, loss_history


def test_2d_solver():
    """Test the 2D differentiable solver with synthetic data."""
    logger.info('Testing 2D DifferentiableSolver...')

    num_pix = 11
    fluo = Fluorescence2D(kmax=5, num_pix=num_pix, num_atoms=4)
    true_phase = fluo.coh_phase_double[fluo.num_pix - 1 :, fluo.num_pix - 1 :]
    cosPhi = fluo.cosPhi_from_data(num_shots=1000)

    # Initialize solver
    solver = DifferentiableSolver(cosPhi, epochs=500, lr=0.01, log_level='INFO')

    # Solve
    phase, loss_history = solver.solve()

    logger.debug('Final phase shape: %s', phase.shape)
    logger.debug('Final loss: %.6f', loss_history[-1])
    logger.debug('Loss decreased: %s', loss_history[0] > loss_history[-1])

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    # Plot true phase
    im1 = ax1.imshow(
        true_phase, cmap='viridis', interpolation='nearest', origin='lower'
    )
    ax1.set_title('True Phase (2D)')
    ax1.set_xlabel('X Pixel')
    ax1.set_ylabel('Y Pixel')
    plt.colorbar(im1, ax=ax1, label='Phase')

    # Plot solved phase
    solved_phase_np = phase.cpu().numpy()
    im2 = ax2.imshow(
        solved_phase_np, cmap='viridis', interpolation='nearest', origin='lower'
    )
    ax2.set_title('Solved Phase (2D)')
    ax2.set_xlabel('X Pixel')
    ax2.set_ylabel('Y Pixel')
    plt.colorbar(im2, ax=ax2, label='Phase')

    # Plot loss history
    ax3.semilogy(loss_history, 'g-')
    ax3.set_title('Training Loss')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.grid(True)

    plt.tight_layout()
    # plt.savefig('2d_differentiable_solver_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    logger.info('2D test completed successfully!')
    return true_phase, solved_phase_np, loss_history


if __name__ == '__main__':
    # true_1d, solved_1d, loss_1d = test_1d_solver()
    true_2d, solved_2d, loss_2d = test_2d_solver()
    logger.info('All tests passed!')
