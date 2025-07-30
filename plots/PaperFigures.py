#!/usr/bin/env python

import copy

import harminv
import matplotlib
import matplotlib.ticker
import numpy as np
import pylab as P
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from skimage.restoration import unwrap_phase


def highlight_cell(x, y, ax=None, **kwargs):
    """Highlights a cell in a given plot.

    Parameters:
        x (float): The x-coordinate of the cell to highlight.
        y (float): The y-coordinate of the cell to highlight.
        ax (Axes, optional): The axes object to add the rectangle patch to.
                             If not provided, the current axes will be used.
        **kwargs: Additional keyword arguments to pass to the Rectangle constructor.

    Returns:
        rect (Rectangle): The rectangle patch representing the highlighted cell.
    """
    # Create a rectangle patch with the given coordinates and size
    rect = P.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, **kwargs)

    # For figure 5, scale the coordinates and size of the rectangle
    rect = P.Rectangle(
        ((x - 0 * 0.5) * 6 / 11, (y - 0 * 0.5) * 6 / 11),
        6 / 11,
        6 / 11,
        fill=False,
        **kwargs,
    )

    # If an axes object is not provided, use the current axes
    ax = ax or P.gca()

    # Add the rectangle patch to the axes
    ax.add_patch(rect)

    # Return the rectangle patch
    return rect


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    """CustomFormatter class inherits from ScalarFormatter class

    Args:
        order (int): The order of magnitude for the tick labels (default: 0).
        fformat (str): The format string for the tick labels (default: "%1.1f").
        offset (bool): Whether to use an offset for the tick labels (default: True).
        mathText (bool): Whether to use math text for the tick labels (default: True).
    """

    def __init__(self, order=0, fformat='%1.1f', offset=True, mathText=True):
        """Initializes a CustomFormatter object.

        Args:
            order (int): The order of magnitude for the tick labels (default: 0).
            fformat (str): The format string for the tick labels (default: "%1.1f").
            offset (bool): Whether to use an offset for the tick labels (default: True).
            mathText (bool): Whether to use math text for the tick labels (default: True).
        """
        self.oom = order
        self.fformat = fformat
        super().__init__(useOffset=offset, useMathText=mathText)


def Figure_1():
    """Plot a schematic of diffraction from a group of atoms in the forward
    direction (coherent diffraction) and a set of incoherent frames collected
    from the side (incoherent diffraction).
    """
    np.random.seed(0x5EED)
    fluo = Speckle_2D.Fluorescence2D(kmax=5, num_pix=201, num_atoms=19)

    def draw_sphere(x_center, y_center, z_center, radius):
        """Draw a sphere centered at (x_center, y_center, z_center) with the given
        radius.

        Args:
            x_center (float): x-coordinate of the center of the sphere.
            y_center (float): y-coordinate of the center of the sphere.
            z_center (float): z-coordinate of the center of the sphere.
            radius (float): radius of the sphere.

        Returns:
            tuple: A tuple of arrays (x, y, z) representing the coordinates of
            points on the sphere.

        """
        # Draw the sphere
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)

        # Shift and scale the sphere
        x = radius * x + x_center
        y = radius * y + y_center
        z = radius * z + z_center

        return (x, y, z)

    coh_diff = np.abs(fluo.coh_ft)
    coh_diff[0, :] = coh_diff[:, 0] = 1
    coh_diff[-2, :] = coh_diff[:, -2] = 1
    incoh_diff = fluo.get_incoh_intens()
    incoh_diff[0, :] = incoh_diff[:, 0] = 1
    incoh_diff[-2, :] = incoh_diff[:, -2] = 1
    incoh_diff_1 = fluo.get_incoh_intens()
    incoh_diff_1[0, :] = incoh_diff_1[:, 0] = 1
    incoh_diff_1[-2, :] = incoh_diff_1[:, -2] = 1
    incoh_diff_2 = fluo.get_incoh_intens()
    incoh_diff_2[0, :] = incoh_diff_2[:, 0] = 1
    incoh_diff_2[-2, :] = incoh_diff_2[:, -2] = 1
    incoh_diff_3 = fluo.get_incoh_intens()
    incoh_diff_3[0, :] = incoh_diff_3[:, 0] = 1
    incoh_diff_3[-2, :] = incoh_diff_3[:, -2] = 1
    X = fluo.k_pix[0, :, :]
    Y = fluo.k_pix[1, :, :]
    Z = (np.max(X) + 2) * np.ones_like(X)

    # Vectors for labels
    # Coherent diffraction
    Kx = np.linspace(-(np.max(X) + 2), 1.25 * (np.max(X) + 2), 100)
    Ky = np.zeros_like(Kx)
    Kz = np.zeros_like(Kx)
    kx = np.linspace(-(np.max(X) + 2), 0, 100)
    ky = np.linspace(-0.35 * (np.max(X) + 2), 0, 100)
    kz = np.linspace(-0.35 * (np.max(X) + 2), 0, 100)

    # Incoherent diffraction
    k1y = np.linspace(0, (np.max(X) + 2), 100)
    k1x = np.linspace(0, -0.5 * (np.max(X) + 2), 100)
    k1z = np.linspace(0, 0.25 * (np.max(X) + 2), 100)

    k2y = np.linspace(0, (np.max(X) + 2), 100)
    k2x = np.linspace(0, 0.0625 * (np.max(X) + 2), 100)
    k2z = np.linspace(0, 0.5 * (np.max(X) + 2), 100)

    k3y = np.linspace(0, (np.max(X) + 2), 100)
    k3x = np.linspace(0, 0.35 * (np.max(X) + 2), 100)
    k3z = np.linspace(0, -0.125 * (np.max(X) + 2), 100)

    q1x = np.linspace((-0.5) * (np.max(X) + 2), (0.0625) * (np.max(X) + 2), 100)
    q1y = np.linspace((np.max(X) + 2), (np.max(X) + 2), 100)
    q1z = np.linspace(0.25 * (np.max(X) + 2), 0.5 * (np.max(X) + 2), 100)

    q2x = np.linspace(0.0625 * (np.max(X) + 2), 0.35 * (np.max(X) + 2), 100)
    q2y = np.linspace(1 * (np.max(X) + 2), 1 * (np.max(X) + 2), 100)
    q2z = np.linspace(0.5 * (np.max(X) + 2), -0.125 * (np.max(X) + 2), 100)

    q3x = np.linspace(0.35 * (np.max(X) + 2), -0.5 * (np.max(X) + 2), 100)
    q3y = np.linspace(1 * (np.max(X) + 2), 1 * (np.max(X) + 2), 100)
    q3z = np.linspace(-0.125 * (np.max(X) + 2), 0.25 * (np.max(X) + 2), 100)

    fig = P.figure(figsize=(10, 10))
    P.rcParams.update({'font.size': 22})
    # show the 3D rotated projection
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(
        X,
        Z,
        Y,
        rstride=1,
        cstride=1,
        facecolors=P.cm.binary(incoh_diff / np.max(incoh_diff)),
        shade=False,
        zorder=5,
    )
    ax1.plot_surface(
        X,
        Z + 1,
        Y,
        rstride=1,
        cstride=1,
        facecolors=P.cm.binary(incoh_diff_1 / np.max(incoh_diff_1)),
        shade=False,
        zorder=4,
    )
    ax1.plot_surface(
        X,
        Z + 2,
        Y,
        rstride=1,
        cstride=1,
        facecolors=P.cm.binary(incoh_diff_2 / np.max(incoh_diff_2)),
        shade=False,
        zorder=3,
    )
    ax1.plot_surface(
        -Z,
        X,
        Y,
        rstride=1,
        cstride=1,
        facecolors=P.cm.binary(coh_diff / np.max(coh_diff)),
        shade=False,
        zorder=1,
    )

    # plot vectors
    ax1.plot(Kx, Ky, Kz, 'r', zorder=50, lw=3)
    ax1.text(-0.4 * (np.max(X) + 2), 0, 0, r'$\vec{K}$', 'x', zorder=1000)
    ax1.plot(kx, ky, kz, 'r--', zorder=60, lw=1)
    ax1.text(
        -0.3 * (np.max(X) + 2),
        -0.15 * (np.max(X) + 2),
        -0.25 * (np.max(X) + 2),
        r'$\vec{k}$',
        (-(np.max(X) + 2), -0.35 * (np.max(X) + 2), -0.35 * (np.max(X) + 2)),
        zorder=1000,
    )
    ax1.plot(k1x, k1y, k1z, 'b--', zorder=70, lw=1)
    ax1.text(
        -0.27 * (np.max(X) + 2),
        0.55 * (np.max(X) + 2),
        0.1 * (np.max(X) + 2),
        r'$\vec{k}_1$',
        (-0.5 * (np.max(X) + 2), (np.max(X) + 2), 0.25 * (np.max(X) + 2)),
        zorder=1000,
    )
    ax1.plot(k2x, k2y, k2z, 'b--', zorder=80, lw=1)
    ax1.text(
        0.17 * (np.max(X) + 2),
        0.07 * (np.max(X) + 2),
        0.2 * (np.max(X) + 2),
        r'$\vec{k}_2$',
        (0.0625 * (np.max(X) + 2), (np.max(X) + 2), 0.5 * (np.max(X) + 2)),
        zorder=1000,
    )
    ax1.plot(k3x, k3y, k3z, 'b--', zorder=90, lw=1)
    ax1.text(
        0.25 * (np.max(X) + 2),
        0.25 * (np.max(X) + 2),
        -0.1 * (np.max(X) + 2),
        r'$\vec{k}_3$',
        (0.35 * (np.max(X) + 2), (np.max(X) + 2), -0.125 * (np.max(X) + 2)),
        zorder=1000,
    )
    ax1.scatter(-(np.max(X) + 2), 0, 0, c='red', zorder=40, s=30)
    # q
    ax1.plot(
        -(np.max(X) + 2) * np.ones_like(ky), ky - Ky, kz - Kz, 'g', zorder=100, lw=2
    )
    ax1.text(
        -(np.max(X) + 2),
        -0.45 * (np.max(X) + 2),
        -0.05 * (np.max(X) + 2),
        r'$\vec{q}$',
        (0, 1, 0.65),
        zorder=1000,
    )
    # q1,q2, q3
    ax1.plot(q1x, q1y, q1z, 'g', zorder=110, lw=2)
    ax1.text(
        -0.3 * (np.max(X) + 2),
        (np.max(X) + 2),
        0.4 * (np.max(X) + 2),
        r'$\vec{q}_1$',
        (1, 0, 0.4),
        zorder=1000,
    )
    ax1.plot(q2x, q2y, q2z, 'g', zorder=120, lw=2)
    ax1.text(
        0.2 * (np.max(X) + 2),
        (np.max(X) + 2),
        0.15 * (np.max(X) + 2),
        r'$\vec{q}_2$',
        (1, 0, -1.8),
        zorder=1000,
    )
    ax1.plot(q3x, q3y, q3z, 'g', zorder=130, lw=2)
    ax1.text(
        -0.2 * (np.max(X) + 2),
        (np.max(X) + 2),
        -0.15 * (np.max(X) + 2),
        r'$\vec{q}_3$',
        (1, 0, -0.5),
        zorder=1000,
    )

    # draw a sphere for each data point
    r = 0.25 * np.ones_like(fluo.coords[1, :])
    n = 500
    for xi, yi, zi, ri in zip(
        np.zeros_like(fluo.coords[0, :]),
        fluo.coords[0, :],
        fluo.coords[1, :],
        r,
        strict=False,
    ):
        (xs, ys, zs) = draw_sphere(xi, yi, zi, ri)
        ax1.plot_surface(xs, ys, zs, color='purple', zorder=n)
        n += 10

    # Hide grid lines
    ax1.axis('off')
    ax1.grid(False)

    # Hide axes ticks
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])

    P.tight_layout()
    P.show()


def Figure_S1():
    """Show how the inverse Fourier transform loses fidelity when the phases
    from the Fourier transform of two images are swapped.
    """
    np.random.seed(0x5EED)

    MITPhysics = np.array(Image.open('images/MITPhysics.jpeg').convert('L'))
    MITCamera = np.array(Image.open('images/MITCamera.jpeg').convert('L'))
    Mod_MITPhysics = np.abs(np.fft.fftshift(np.fft.fft2(MITPhysics)))
    # Mod_MITPhysics = np.ones_like(Mod_MITPhysics)
    Angle_MITPhysics = np.angle(np.fft.fftshift(np.fft.fft2(MITPhysics)))
    # Angle_MITPhysics = np.zeros_like(Angle_MITPhysics)
    Mod_MITCamera = np.abs(np.fft.fftshift(np.fft.fft2(MITCamera)))
    # Mod_MITCamera = np.ones_like(Mod_MITCamera)
    Angle_MITCamera = np.angle(np.fft.fftshift(np.fft.fft2(MITCamera)))
    # Angle_MITCamera = np.zeros_like(Angle_MITCamera)
    Inverted_1 = np.real(
        np.fft.fft2(
            np.fft.fftshift(Mod_MITPhysics)
            * np.exp(-1j * np.fft.fftshift(Angle_MITCamera))
        )
    )
    Inverted_2 = np.real(
        np.fft.fft2(
            np.fft.fftshift(Mod_MITCamera)
            * np.exp(-1j * np.fft.fftshift(Angle_MITPhysics))
        )
    )

    # Plot the original images
    fig = P.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    ax1.imshow(MITPhysics, cmap='gray')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    P.tight_layout()
    P.show()

    fig = P.figure(figsize=(7, 7))
    ax4 = fig.add_subplot(111)
    ax4.imshow(MITCamera, cmap='gray')
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    P.tight_layout()
    P.show()

    # Plot the Fourier Transforms
    fig = P.figure(figsize=(7, 7))
    ax2_1 = fig.add_subplot(111)
    ax2_1.imshow(np.log(Mod_MITPhysics))
    ax2_1.get_xaxis().set_visible(False)
    ax2_1.get_yaxis().set_visible(False)
    P.tight_layout()
    P.show()

    fig = P.figure(figsize=(7, 7))
    ax2_2 = fig.add_subplot(111)
    ax2_2.imshow(Angle_MITPhysics)
    ax2_2.get_xaxis().set_visible(False)
    ax2_2.get_yaxis().set_visible(False)
    P.tight_layout()
    P.show()

    fig = P.figure(figsize=(7, 7))
    ax5_1 = fig.add_subplot(111)
    ax5_1.imshow(Angle_MITCamera)
    ax5_1.get_xaxis().set_visible(False)
    ax5_1.get_yaxis().set_visible(False)
    P.tight_layout()
    P.show()

    fig = P.figure(figsize=(7, 7))
    ax5_2 = fig.add_subplot(111)
    ax5_2.imshow(np.log(Mod_MITCamera))
    ax5_2.get_xaxis().set_visible(False)
    ax5_2.get_yaxis().set_visible(False)
    P.tight_layout()
    P.show()

    # Plot the inverted images with swapped phases
    fig = P.figure(figsize=(7, 7))
    ax3 = fig.add_subplot(111)
    ax3.imshow(Inverted_1, cmap='gray')
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    P.tight_layout()
    P.show()

    fig = P.figure(figsize=(7, 7))
    ax6 = fig.add_subplot(111)
    ax6.imshow(Inverted_2, cmap='gray')
    ax6.get_xaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)
    P.tight_layout()
    P.show()


def BiSpectrumClosure():
    """Plot the bispectrum, closure, closure phase, and a non-redundant
    region of Phi to illustrate what experimental data we are using to
    retrieve the phase.
    """
    fluo = Speckle_1D.Fluorescence1D(kmax=5, num_pix=101, num_atoms=3)
    num_shots = 10000

    cmap = copy.copy(matplotlib.cm.get_cmap('viridis'))
    cmap.set_bad(color='white')

    fig = P.figure(figsize=(20, 5))
    P.rcParams.update({'font.size': 22})

    # Plot the bispectrum
    ax1 = fig.add_subplot(141)
    box_extent = 2 * fluo.kmax
    bispectrum = fluo.marginalize_g3(num_shots=num_shots)
    masked_bispectrum = np.ma.masked_where(fluo.weights_2d == 0, bispectrum)
    im = ax1.imshow(
        masked_bispectrum,
        origin='lower',
        cmap=cmap,
        extent=[-box_extent, box_extent, -box_extent, box_extent],
    )
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    cbar = P.colorbar(im, cax=cax1)
    ax1.set_ylabel(r'$q_2$ $[\mathrm{Length}]^{-1}$')
    ax1.set_xlabel(r'$q_1$ $[\mathrm{Length}]^{-1}$')
    ax1.set_title('Bispectrum')
    ax1.text(
        0.87,
        0.95,
        'A',
        transform=ax1.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='black',
    )
    ax1.set_xticks([-10, 0, 10])
    ax1.set_yticks([-10, 0, 10])

    # Plot the closure (the entire last term in the triple correlation)
    # The bispectrum is the entire triple correlation in q1,q2 space
    ax2 = fig.add_subplot(142)
    closure = fluo.closure_from_data(num_shots=num_shots)
    masked_closure = np.ma.masked_where(fluo.weights_2d == 0, closure)
    im = ax2.imshow(
        masked_closure,
        origin='lower',
        cmap=cmap,
        extent=[-box_extent, box_extent, -box_extent, box_extent],
    )
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes('right', size='5%', pad=0.05)
    cbar = P.colorbar(im, cax=cax2)
    ax2.set_xlabel(r'$q_1$ $[\mathrm{Length}]^{-1}$')
    ax2.set_title('Closure')
    ax2.text(
        0.87,
        0.95,
        'B',
        transform=ax2.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='black',
    )
    ax2.set_xticks([-10, 0, 10])
    ax2.set_yticks([-10, 0, 10])

    # Plot the closure phase label useful area of Phi
    ax3 = fig.add_subplot(143)
    closurePhase = np.arccos(fluo.cosPhi_from_data(num_shots=num_shots))
    masked_closurePhase = np.ma.masked_where(fluo.weights_2d == 0, closurePhase)
    im = ax3.imshow(
        masked_closurePhase,
        origin='lower',
        cmap=cmap,
        extent=[-box_extent, box_extent, -box_extent, box_extent],
    )
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes('right', size='5%', pad=0.05)
    cbar = P.colorbar(im, cax=cax3)
    cbar.set_ticks([0, np.pi])
    cbar.set_ticklabels(['0', r'$\pi$'])
    ax3.set_xlabel(r'$q_1$ $[\mathrm{Length}]^{-1}$')
    ax3.set_title('Closure Phase')
    ax3.text(
        0.87,
        0.95,
        'C',
        transform=ax3.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='black',
    )
    # Create a Rectangle patch
    rect = matplotlib.patches.Rectangle(
        (0, 0),
        box_extent / 2,
        box_extent / 2,
        linewidth=2,
        edgecolor='w',
        facecolor='none',
    )
    # Add the patch to the Axes
    ax3.add_patch(rect)
    ax3.set_xticks([-10, 0, 10])
    ax3.set_yticks([-10, 0, 10])

    # Plot Phi and label the diagonals processed for phase recovery
    box_extent = fluo.kmax
    ax4 = fig.add_subplot(144)
    Phi = np.arccos(
        fluo.cosPhi_from_data(num_shots=num_shots)[
            fluo.num_pix - 1 : 3 * fluo.num_pix // 2,
            fluo.num_pix - 1 : 3 * fluo.num_pix // 2,
        ]
    )
    im = ax4.imshow(Phi, origin='lower', extent=[0, box_extent, 0, box_extent])
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes('right', size='5%', pad=0.05)
    cbar = P.colorbar(im, cax=cax4)
    cbar.set_ticks([0, np.pi])
    cbar.set_ticklabels(['0', r'$\pi$'])
    ax4.set_xlabel(r'$q_1$ $[\mathrm{Length}]^{-1}$')
    ax4.set_title(r'$\left| \Phi \right|$')
    ax4.text(
        0.87,
        0.95,
        'D',
        transform=ax4.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='white',
    )
    ax4.set_xticks([0, 5])
    ax4.set_yticks([0, 5])
    # Draw diagonal
    ax4.plot(
        np.linspace(0, box_extent, 1000),
        np.linspace(0, box_extent, 1000),
        color='red',
        zorder=1,
    )
    # Draw diagonals processed up to kmax
    for x in np.linspace(0, box_extent, len(Phi[0, :]) + 1):
        # Create a Rectangle patch
        rect = matplotlib.patches.Rectangle(
            (
                x - 10 * box_extent / len(Phi[0, :]),
                box_extent - x - 10 * box_extent / len(Phi[0, :]),
            ),
            box_extent / len(Phi[0, :]),
            box_extent / len(Phi[0, :]),
            linewidth=1,
            edgecolor='w',
            facecolor='none',
            zorder=2,
        )
        # Add the patch to the Axes
        ax4.add_patch(rect)
    # Draw diagonals processed up to qmax
    for x in np.linspace(0, box_extent, len(Phi[0, :]) + 1):
        # Create a Rectangle patch
        rect = matplotlib.patches.Rectangle(
            (
                x + 10 * box_extent / len(Phi[0, :]),
                box_extent - x + 10 * box_extent / len(Phi[0, :]),
            ),
            box_extent / len(Phi[0, :]),
            box_extent / len(Phi[0, :]),
            linewidth=1,
            edgecolor='pink',
            facecolor='none',
            zorder=2,
        )
        # Add the patch to the Axes
        ax4.add_patch(rect)
    ax4.set_xlim([0, box_extent])
    ax4.set_ylim([0, box_extent])

    P.tight_layout()
    P.show()


def Figure_2():
    """Show how when the phase is retrieved from the first-order difference
    equation in Phi without the sign information, the phase solution has
    contours at the expected locations but the sign of the slope is incorrect.
    """
    np.random.seed(0x5EED + 1)
    fluo = Speckle_1D.Fluorescence1D(kmax=5, num_pix=201, num_atoms=3)
    cosPhi = fluo.cosPhi_from_data(num_shots=10000)
    initialPhase = fluo.coh_phase[fluo.num_pix // 2 :][0]

    solved = TriPhase_1D.simple_PhiSolver(cosPhi=cosPhi, initial_phase=initialPhase)
    solved = unwrap_phase(solved)
    real_phase = fluo.coh_phase[fluo.num_pix // 2 :]
    real_phase = unwrap_phase(real_phase)

    fig = P.figure(figsize=(7, 5))
    P.rcParams.update({'font.size': 22})
    # Plot the solved phase branch
    s = fig.add_subplot(111)
    plot_1 = P.plot(
        fluo.k_pix[fluo.num_pix // 2 :],
        real_phase,
        'o--',
        label=r'$\phi_{\mathrm{true}}$',
    )
    plot_2 = P.plot(
        fluo.k_pix[fluo.num_pix // 2 : fluo.num_pix - 1],
        solved,
        'o--',
        label=r'$\phi_{\Phi(\vec{k}) = \left|\Phi(\vec{k}) \right|}$',
    )
    s.axvspan(0, (27 / 101) * 5, alpha=0.3, color='red')
    s.axvspan((27 / 101) * 5, (64 / 101) * 5, alpha=0.3, color='blue')
    s.axvspan((64 / 101) * 5, (101 / 101) * 5, alpha=0.3, color='red')
    s.set_ylabel(r'$\phi(\vec{k})$')
    s.set_xlabel(r'$\vec{k}$')
    P.rcParams.update({'font.size': 16})
    s.set_xticks([0, 5])
    s.set_yticks([-3 * np.pi, -2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi, 3 * np.pi])
    s.set_yticklabels(
        [r'$-3\pi$', r'$-2\pi$', r'$-\pi$', r'$0$', r'$\pi$', r'$2\pi$', r'$3\pi$']
    )
    P.legend(loc='upper left')
    P.tight_layout()
    P.show()


def Figure_3():
    """Show how plotting candidate phases in xy-coordinates illustrates the
    intersection-finding optimization problem. Also show the error landscape
    for the optimization.
    """
    np.random.seed(0x5EED)
    fluo = Speckle_1D.Fluorescence1D(kmax=5, num_pix=51, num_atoms=3)
    # This code was previously a commented section in PhiSolver. This needs
    # to be retrieved from a previous commit.


def Figure_ResolvingDemo():
    """Show how the use of resolving to minimize the total error leads to a
    correct phase solution.
    """
    np.random.seed(0x5EED)
    fluo = Speckle_2D.Fluorescence2D(kmax=3, num_pix=11, num_atoms=3)
    real_phase = fluo.coh_phase_double[
        fluo.num_pix - 1 : 2 * fluo.num_pix - 1, fluo.num_pix - 1 : 2 * fluo.num_pix - 1
    ]

    num_shots = 1000
    cosPhi = fluo.cosPhi_from_data(num_shots=num_shots)
    initialPhase = [real_phase[0, 1], real_phase[1, 0]]
    solved, error = TriPhase_2D.PhiSolver(
        cosPhi, initial_phase=initialPhase, error_reject=50
    )

    solved = solved
    box_extent = 2 * fluo.kmax

    fig = P.figure(figsize=(20, 10))
    P.rcParams.update({'font.size': 22})
    # First, without resolving
    # Plot the solved phase branch
    ax1 = fig.add_subplot(241)
    im = ax1.imshow(
        solved,
        cmap='twilight_shifted',
        origin='lower',
        vmin=-np.pi,
        vmax=np.pi,
        extent=[0, box_extent, 0, box_extent],
    )
    ax1.set_title(r'$\phi$ Solved')
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    cbar = P.colorbar(im, cax=cax1)
    ax1.set_aspect('equal')
    ax1.get_xaxis().set_visible(False)
    # ax1.get_yaxis().set_visible(False)
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax1.set_ylabel(r'$k_y$ $[\mathrm{Length}]^{-1}$')
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

    ax2 = fig.add_subplot(242)
    im = ax2.imshow(
        real_phase,
        cmap='twilight_shifted',
        origin='lower',
        vmin=-np.pi,
        vmax=np.pi,
        extent=[0, box_extent, 0, box_extent],
    )
    ax2.set_title(r'$\phi$ Truth')
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes('right', size='5%', pad=0.05)
    cbar = P.colorbar(im, cax=cax2)
    ax2.set_aspect('equal')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax2.text(
        0.05,
        0.95,
        'B',
        transform=ax2.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='white',
    )

    ax3 = fig.add_subplot(245)
    im = ax3.imshow(
        error,
        cmap='coolwarm',
        origin='lower',
        extent=[0, box_extent, 0, box_extent],
        vmax=5,
        vmin=-25,
    )
    ax3.set_title(r'$\log \left[E(\phi) \right]$')
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes('right', size='5%', pad=0.05)
    cbar = P.colorbar(im, cax=cax3)
    ax3.set_aspect('equal')
    # ax3.get_xaxis().set_visible(False)
    # ax3.get_yaxis().set_visible(False)
    ax3.set_xlabel(r'$k_x$ $[\mathrm{Length}]^{-1}$')
    ax3.set_ylabel(r'$k_y$ $[\mathrm{Length}]^{-1}$')
    # Highlight the pixels where the error occurs
    highlight_cell(3, 0, ax=ax3, color='limegreen', linewidth=3)
    highlight_cell(2, 1, ax=ax3, color='limegreen', linewidth=3)
    highlight_cell(0, 3, ax=ax3, color='limegreen', linewidth=3)
    highlight_cell(1, 2, ax=ax3, color='red', linewidth=3)
    highlight_cell(0, 2, ax=ax3, color='blue', linewidth=3)
    ax3.text(
        0.05,
        0.95,
        'E',
        transform=ax3.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='white',
    )
    cbar.set_ticks([5, -5, -15, -25])

    ax4 = fig.add_subplot(246)
    im = ax4.imshow(
        real_phase - solved,
        cmap='PRGn',
        origin='lower',
        extent=[0, box_extent, 0, box_extent],
        vmin=-2 * np.pi,
        vmax=2 * np.pi,
    )
    ax4.set_title('Truth - Solved')
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes('right', size='5%', pad=0.05)
    cbar = P.colorbar(im, cax=cax4)
    ax4.set_aspect('equal')
    # ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4.set_xlabel(r'$k_x$ $[\mathrm{Length}]^{-1}$')
    ax4.text(
        0.05,
        0.95,
        'F',
        transform=ax4.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='black',
    )
    cbar.set_ticks([-2 * np.pi, 0, 2 * np.pi])
    cbar.set_ticklabels([r'$-2\pi$', '0', r'$2\pi$'])

    # Secondly, with resolving
    cosPhi = fluo.cosPhi_from_data(num_shots=num_shots)
    initialPhase = [real_phase[0, 1], real_phase[1, 0]]
    solved, error = TriPhase_2D.PhiSolver(
        cosPhi, initial_phase=initialPhase, error_reject=-1
    )
    # Plot the solved phase branch
    ax1 = fig.add_subplot(243)
    im = ax1.imshow(
        solved,
        cmap='twilight_shifted',
        origin='lower',
        vmin=-np.pi,
        vmax=np.pi,
        extent=[0, box_extent, 0, box_extent],
    )
    ax1.set_title(r'$\phi$ Solved')
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes('right', size='5%', pad=0.05)
    cbar = P.colorbar(im, cax=cax1)
    ax1.set_aspect('equal')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax1.text(
        0.05,
        0.95,
        'C',
        transform=ax1.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='white',
    )

    ax2 = fig.add_subplot(244)
    im = ax2.imshow(
        real_phase,
        cmap='twilight_shifted',
        origin='lower',
        vmin=-np.pi,
        vmax=np.pi,
        extent=[0, box_extent, 0, box_extent],
    )
    ax2.set_title(r'$\phi$ Truth')
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes('right', size='5%', pad=0.05)
    cbar = P.colorbar(im, cax=cax2)
    ax2.set_aspect('equal')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    cbar.set_ticks([-np.pi, 0, np.pi])
    cbar.set_ticklabels([r'$-\pi$', '0', r'$\pi$'])
    ax2.text(
        0.05,
        0.95,
        'D',
        transform=ax2.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='white',
    )

    ax3 = fig.add_subplot(247)
    im = ax3.imshow(
        error,
        cmap='coolwarm',
        origin='lower',
        extent=[0, box_extent, 0, box_extent],
        vmin=-25,
        vmax=5,
    )
    ax3.set_title(r'$\log \left[E(\phi) \right]$')
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes('right', size='5%', pad=0.05)
    cbar = P.colorbar(im, cax=cax3)
    ax3.set_aspect('equal')
    # ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set_xlabel(r'$k_x$ $[\mathrm{Length}]^{-1}$')
    highlight_cell(3, 0, ax=ax3, color='limegreen', linewidth=3)
    highlight_cell(2, 1, ax=ax3, color='limegreen', linewidth=3)
    highlight_cell(0, 3, ax=ax3, color='limegreen', linewidth=3)
    highlight_cell(1, 2, ax=ax3, color='limegreen', linewidth=3)
    ax3.text(
        0.05,
        0.95,
        'G',
        transform=ax3.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='white',
    )
    cbar.set_ticks([5, -5, -15, -25])

    ax4 = fig.add_subplot(248)
    im = ax4.imshow(
        real_phase - solved,
        cmap='PRGn',
        origin='lower',
        extent=[0, box_extent, 0, box_extent],
        vmin=-2 * np.pi,
        vmax=2 * np.pi,
    )
    ax4.set_title('Truth - Solved')
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes('right', size='5%', pad=0.05)
    cbar = P.colorbar(im, format=OOMFormatter(-4, mathText=True), cax=cax4)
    cbar.ax.yaxis.set_offset_position('left')
    ax4.set_aspect('equal')
    # ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4.set_xlabel(r'$k_x$ $[\mathrm{Length}]^{-1}$')
    ax4.text(
        0.05,
        0.95,
        'H',
        transform=ax4.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='black',
    )
    cbar.set_ticks([-2 * np.pi, 0, 2 * np.pi])
    cbar.set_ticklabels([r'$-2\pi$', '0', r'$2\pi$'])

    P.tight_layout()
    P.subplots_adjust(wspace=0.3, hspace=0.0)
    # Box to separate right plot cluster from left
    ax = ax4.axis()
    rec = P.Rectangle(
        (ax[0] - 8.65, ax[2] - 1.55),
        (ax[1] - ax[0]) + 10.65,
        (ax[3] - ax[2]) + 10.1,
        fill=False,
        lw=2,
        linestyle='solid',
    )
    rec = ax4.add_patch(rec)
    rec.set_clip_on(False)
    P.show()


def Figure_4_Rows():
    """Shows that exact ab initio phase solving on a 2D array is possible
    using manual (user-input) resolving to minimize the total error.
    """
    np.random.seed(0x5EED)
    plot = Plot_2D.Plot2D(num_pix=11, num_atoms=7, kmax=2)

    # First, user should do manual resolving to generate subfigures A-D
    plot.plot_PhiSolver_manualSelect(num_shots=20000)

    # Second, user should not do manual resolving to generate subfigures E-H
    plot.plot_PhiSolver_manualSelect(num_shots=20000, altLabel=True)


def Figure_S2():
    """Show how the double correlation may be used to obtain scattering
    intensities outside the detector area despite the sum of shots giving noise.
    """
    np.random.seed(0x5EED + 1)
    fluo = Speckle_2D.Fluorescence2D(kmax=5, num_pix=101, num_atoms=7)

    # Changing this will affect colorbar placement for the whole figure, be
    # careful
    fig = P.figure(figsize=(17, 10))
    P.rcParams.update({'font.size': 22})
    # Plot shots
    ax1 = fig.add_subplot(231)
    box_extent = fluo.kmax
    im = ax1.imshow(
        fluo.get_incoh_intens(),
        cmap='coolwarm',
        origin='lower',
        vmin=0,
        vmax=1,
        extent=[-box_extent, box_extent, -box_extent, box_extent],
    )
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
    # Anyway to add a shadow to this to indicate other shots?
    ax1.set_xlabel(r'$k_x$ $[\mathrm{Length}]^{-1}$')
    ax1.set_ylabel(r'$k_y$ $[\mathrm{Length}]^{-1}$')

    ax2 = fig.add_subplot(232)
    # Result from g2
    num_shots = 1000
    box_extent = 2 * fluo.kmax
    measured = fluo.marginalize_g2(num_shots=num_shots) - 1 + 1.0 / fluo.num_atoms
    im = ax2.imshow(
        measured,
        cmap='coolwarm',
        origin='lower',
        vmin=0,
        vmax=1,
        extent=[-box_extent, box_extent, -box_extent, box_extent],
    )
    ax2.text(
        0.05,
        0.95,
        'C',
        transform=ax2.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='white',
    )
    # Create a Rectangle patch
    rect = matplotlib.patches.Rectangle(
        (-box_extent / 2, -box_extent / 2),
        box_extent,
        box_extent,
        linewidth=2,
        edgecolor='w',
        facecolor='none',
    )
    # Add the patch to the Axes
    ax2.add_patch(rect)
    ax2.set_xlabel(r'$k_x$ $[\mathrm{Length}]^{-1}$')
    ax2.set_ylabel(r'$k_y$ $[\mathrm{Length}]^{-1}$')

    ax3 = fig.add_subplot(233)
    box_extent = np.max(
        np.fft.fftshift(
            np.fft.fftfreq(2 * fluo.num_pix, d=2 * (2 * fluo.kmax) / (2 * fluo.num_pix))
        )
    )
    # IFT = Autocorrelation enhanced
    measured_ift = np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(measured))))
    measured_ift = measured_ift / np.max(measured_ift)
    im = ax3.imshow(
        measured_ift,
        cmap='binary',
        origin='lower',
        vmin=0,
        vmax=0.1,
        extent=[-box_extent, box_extent, -box_extent, box_extent],
    )
    ax3.text(
        0.05,
        0.95,
        'E',
        transform=ax3.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='black',
    )
    ax3.set_xlabel('X [Length]')
    ax3.set_ylabel('Y [Length]')

    ax4 = fig.add_subplot(234)
    # Sum of shots
    box_extent = fluo.kmax
    incoh_sum = np.zeros(2 * (fluo.num_pix,))
    for n in range(num_shots):
        incoh_sum += fluo.get_incoh_intens()
    im = ax4.imshow(
        incoh_sum / num_shots,
        cmap='coolwarm',
        origin='lower',
        vmin=0,
        vmax=1,
        extent=[-box_extent, box_extent, -box_extent, box_extent],
    )
    ax4.text(
        0.05,
        0.95,
        'B',
        transform=ax4.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='white',
    )
    ax4.set_xlabel(r'$k_x$ $[\mathrm{Length}]^{-1}$')
    ax4.set_ylabel(r'$k_y$ $[\mathrm{Length}]^{-1}$')

    ax5 = fig.add_subplot(235)
    # Plot true, non-extended intensity
    box_extent = fluo.kmax
    true = np.abs(fluo.coh_ft) ** 2
    im = ax5.imshow(
        true,
        cmap='coolwarm',
        origin='lower',
        vmin=0,
        vmax=1,
        extent=[-box_extent, box_extent, -box_extent, box_extent],
    )
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
    ax5.set_xlabel(r'$k_x$ $[\mathrm{Length}]^{-1}$')
    ax5.set_ylabel(r'$k_y$ $[\mathrm{Length}]^{-1}$')

    ax6 = fig.add_subplot(236)
    box_extent = np.max(
        np.fft.fftshift(np.fft.fftfreq(fluo.num_pix, d=2 * fluo.kmax / fluo.num_pix))
    )
    # IFT = Autocorrelation non-enhanced
    true_ift = np.abs(np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(true))))
    true_ift = true_ift / np.max(true_ift)
    im = ax6.imshow(
        true_ift,
        cmap='binary',
        origin='lower',
        vmin=0,
        vmax=0.1,
        extent=[-box_extent, box_extent, -box_extent, box_extent],
    )
    ax6.text(
        0.05,
        0.95,
        'F',
        transform=ax6.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='black',
    )
    ax6.set_xlabel('X [Length]')
    ax6.set_ylabel('Y [Length]')

    # Add in a single vertical color bar for all figures
    P.tight_layout()
    # Create new axes according to image position
    # cax = fig.add_axes([ax3.get_position().x1 - 0.03,ax6.get_position().y0-0.002,0.01,2.33*ax3.get_position().height])
    # Plot vertical colorbar
    # P.colorbar(im, cax=cax, pad=0)
    # P.tight_layout()
    # P.subplots_adjust(right=0.9)
    P.show()


def Figure_S3():
    """Show how having extra k-space gives better real space resolution with
    and without phase information
    """
    np.random.seed(0x5EED + 2)
    fluo = Speckle_2D.Fluorescence2D(kmax=8, num_pix=201, num_atoms=5)
    P.rcParams.update({'font.size': 22})

    box_extent = np.max(fluo.x_pix[0])
    # Normal resolution object
    fig = P.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(231)
    # ax1.set_title("Object")
    ax1.imshow(
        fluo.object,
        extent=[-box_extent, box_extent, -box_extent, box_extent],
        origin='lower',
        cmap='gray',
    )
    ax1.scatter(
        fluo.coords[1, :], fluo.coords[0, :], facecolors='none', edgecolors='cyan'
    )
    ax1.get_xaxis().set_visible(False)
    # ax1.get_yaxis().set_visible(False)
    ax1.set_ylabel('Y [Length]')
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

    obj_NoPhase = np.fft.fftshift(fluo.coh_ft)
    obj_NoPhase = np.abs(obj_NoPhase)
    obj_Phase = np.fft.fftshift(fluo.coh_ft)
    phase = np.fft.fftshift(fluo.coh_phase)
    obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)

    obj_NoPhase = np.fft.ifftn(obj_NoPhase)
    obj_Phase = np.fft.ifftn(obj_Phase)

    obj_NoPhase = np.fft.fftshift(obj_NoPhase)
    obj_Phase = np.fft.fftshift(obj_Phase)

    box_extent = np.max(
        np.fft.fftshift(np.fft.fftfreq(fluo.num_pix, d=2 * fluo.kmax / fluo.num_pix))
    )
    # Normal resolution autocorrelation
    ax2 = fig.add_subplot(232)
    # ax2.set_title("Object from Intensity")
    ax2.imshow(
        np.abs(obj_NoPhase),
        extent=[-box_extent, box_extent, -box_extent, box_extent],
        origin='lower',
        cmap='binary',
        vmax=0.1,
    )
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.text(
        0.05,
        0.95,
        'C',
        transform=ax2.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='black',
    )

    ax3 = fig.add_subplot(233)
    # Normal resolution inversion
    # ax3.set_title("Object from Intensity + Phase")
    # Not sure why this needs to be transposed below, but it does give the
    # correct image
    ax3.imshow(
        np.abs(obj_Phase),
        extent=[-box_extent, box_extent, -box_extent, box_extent],
        origin='lower',
        cmap='hot',
    )
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.text(
        0.05,
        0.95,
        'E',
        transform=ax3.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='white',
    )

    # Double resolution
    box_extent = np.max(fluo.x_pix[0])
    ax4 = fig.add_subplot(234)
    # ax4.set_title("Object")
    ax4.imshow(
        fluo.object_double,
        extent=[-box_extent, box_extent, -box_extent, box_extent],
        origin='lower',
        cmap='gray',
    )
    ax4.scatter(
        fluo.coords[1, :], fluo.coords[0, :], facecolors='none', edgecolors='cyan'
    )
    # ax4.get_xaxis().set_visible(False)
    # ax4.get_yaxis().set_visible(False)
    ax4.set_xlabel('X [Length]')
    ax4.set_ylabel('Y [Length]')
    ax4.text(
        0.05,
        0.95,
        'B',
        transform=ax4.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='white',
    )

    obj_NoPhase = np.fft.fftshift(fluo.coh_ft_double)
    obj_NoPhase = np.abs(obj_NoPhase)
    obj_Phase = np.fft.fftshift(fluo.coh_ft_double)
    phase = np.fft.fftshift(fluo.coh_phase_double)
    obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)

    obj_NoPhase = np.fft.ifftn(obj_NoPhase)
    obj_Phase = np.fft.ifftn(obj_Phase)

    obj_NoPhase = np.fft.fftshift(obj_NoPhase)
    obj_Phase = np.fft.fftshift(obj_Phase)

    box_extent = np.max(
        np.fft.fftshift(
            np.fft.fftfreq(2 * fluo.num_pix, d=2 * fluo.kmax / fluo.num_pix)
        )
    )
    # Enhanced resolution autocorrelation
    ax5 = fig.add_subplot(235)
    # ax5.set_title("Object from Intensity")
    ax5.imshow(
        np.abs(obj_NoPhase),
        extent=[-box_extent, box_extent, -box_extent, box_extent],
        origin='lower',
        cmap='binary',
        vmax=0.1,
    )
    ax5.set_xlim([-1, 1])
    ax5.set_ylim([-1, 1])
    # ax5.get_xaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)
    ax5.set_xlabel('X [Length]')
    ax5.text(
        0.05,
        0.95,
        'D',
        transform=ax5.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='black',
    )

    # Enhanced resolution inversion
    ax6 = fig.add_subplot(236)
    # ax6.set_title("Object from Intensity + Phase")
    # Not sure why this needs to be transposed below, but it does give the
    # correct image
    ax6.imshow(
        np.abs(obj_Phase),
        extent=[-box_extent, box_extent, -box_extent, box_extent],
        origin='lower',
        cmap='hot',
    )
    ax6.set_xlim([-1, 1])
    ax6.set_ylim([-1, 1])
    # ax6.get_xaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)
    ax6.set_xlabel('X [Length]')
    ax6.text(
        0.05,
        0.95,
        'F',
        transform=ax6.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='white',
    )

    P.tight_layout()
    P.show()


def Figure_S5():
    np.random.seed(0x5EED + 2)
    # Show the effects of phase ramps, sign flips, and the use of harmonic
    # inversion to find a 1D structure
    fluo = Speckle_1D.Fluorescence1D(kmax=5, num_pix=501, num_atoms=3)

    fig = P.figure(figsize=(20, 5))
    P.rcParams.update({'font.size': 22})
    # First plot the normal phase, ramped, flipped, and both
    s = fig.add_subplot(141)
    truth = unwrap_phase(fluo.coh_phase_double)
    ramped = unwrap_phase(
        fluo.coh_phase_double + np.linspace(0, 50, len(fluo.coh_phase_double))
    )
    flipped = unwrap_phase(-fluo.coh_phase_double)
    rampFlipped = unwrap_phase(
        -fluo.coh_phase_double + np.linspace(0, -25, len(fluo.coh_phase_double))
    )
    s.plot(fluo.q_pix, truth, label='Truth')
    s.plot(fluo.q_pix, ramped, label='Ramped')
    s.plot(fluo.q_pix, flipped, label='Flipped')
    # s.plot(fluo.q_pix, rampFlipped, label="Ramped and Flipped")
    s.set_ylabel(r'$\phi(\vec{q})$')
    s.set_xlabel(r'$\vec{q}$')
    s.text(
        0.05,
        0.95,
        'A',
        transform=s.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='black',
    )
    P.rcParams.update({'font.size': 16})
    s.legend(loc='lower left')
    P.rcParams.update({'font.size': 22})

    # Plot the true phase Fourier Transform
    zero_pad = len(fluo.coh_phase_double) + 0
    obj_Phase = np.fft.fftshift(fluo.coh_ft_double)
    phase = np.fft.fftshift(truth)
    obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)
    obj_Phase = np.fft.ifft(obj_Phase, n=zero_pad)
    # When using DFT mode, why does this line need commenting?
    obj_Phase = np.fft.fftshift(obj_Phase)
    obj_scaled_x = np.fft.fftshift(
        np.fft.fftfreq(zero_pad, d=2 * 2 * fluo.kmax / (2 * fluo.num_pix))
    )
    # Harmonic inversion
    signal = np.sqrt(fluo.coh_ft_double) * np.exp(1j * truth)
    signal = np.real(signal)
    inversion = harminv.invert(
        signal, fmin=0, fmax=1, dt=2 * 2 * fluo.kmax / (2 * fluo.num_pix), nf=6
    )
    harm_x = inversion.frequency[inversion.amplitude > 0.1]
    print('Truth', fluo.coords)
    # frequencies
    print('Frequencies', inversion.frequency[inversion.amplitude > 0.1])
    # decay rates
    print('Decay Rates', inversion.decay[inversion.amplitude > 0.1])
    print('Q Factor', inversion.Q[inversion.amplitude > 0.1])  # Q factor
    # phase shift
    print('Phase Shift', inversion.phase[inversion.amplitude > 0.1])
    # absolute amplitudes
    print('Amplitude', inversion.amplitude[inversion.amplitude > 0.1])

    s = fig.add_subplot(142)
    s.plot(
        obj_scaled_x,
        np.abs(obj_Phase) / np.max(np.abs(obj_Phase)),
        label='Truth IFT',
        color='C0',
    )
    s.set_xlim([-1, 1])
    for x in fluo.coords:
        s.axvline(x=x, linestyle='dashed', color='C0')
    for x in harm_x:
        s.axvline(x=x, linestyle='dotted', color='C0')
    s.set_xlabel('X [Length]')
    s.set_ylabel('Brightness [a.u.]')
    # s.get_xaxis().set_visible(False)
    s.get_yaxis().set_visible(False)
    s.text(
        0.05,
        0.95,
        'B',
        transform=s.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='black',
    )
    P.rcParams.update({'font.size': 16})
    # s.legend()
    P.rcParams.update({'font.size': 22})
    s.set_xticks([-1, 0, 1])

    # Plot the ramped phase Fourier Transform
    obj_Phase = np.fft.fftshift(fluo.coh_ft_double)
    phase = np.fft.fftshift(ramped)
    obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)
    obj_Phase = np.fft.ifft(obj_Phase, n=zero_pad)
    # When using DFT mode, why does this line need commenting?
    obj_Phase = np.fft.fftshift(obj_Phase)
    obj_scaled_x = np.fft.fftshift(
        np.fft.fftfreq(zero_pad, d=2 * 2 * fluo.kmax / (2 * fluo.num_pix))
    )
    # Harmonic inversion
    signal = np.sqrt(fluo.coh_ft_double) * np.exp(1j * ramped)
    signal = np.real(signal)
    inversion = harminv.invert(
        signal[:], fmin=0, fmax=0.6, dt=2 * 2 * fluo.kmax / (2 * fluo.num_pix), nf=6
    )
    harm_x = inversion.frequency[inversion.amplitude > 0.1][1:] * np.sign(
        inversion.phase[inversion.amplitude > 0.1][1:]
    )
    print('Truth', fluo.coords)
    # frequencies
    print('Frequencies', inversion.frequency[inversion.amplitude > 0.1])
    # decay rates
    print('Decay Rates', inversion.decay[inversion.amplitude > 0.1])
    print('Q Factor', inversion.Q[inversion.amplitude > 0.1])  # Q factor
    # phase shift
    print('Phase Shift', inversion.phase[inversion.amplitude > 0.1])
    # absolute amplitudes
    print('Amplitude', inversion.amplitude[inversion.amplitude > 0.1])

    s = fig.add_subplot(143)
    s.plot(
        obj_scaled_x,
        np.abs(obj_Phase) / np.max(np.abs(obj_Phase)),
        label='Ramped IFT',
        color='C1',
    )
    s.set_xlim([-1, 1])
    for x in fluo.coords:
        s.axvline(x=x, linestyle='dashed', color='C0')
    for x in harm_x:
        s.axvline(x=x, linestyle='dotted', color='C1')
    s.set_xlabel('X [Length]')
    s.set_ylabel('Brightness [a.u.]')
    # s.get_xaxis().set_visible(False)
    s.get_yaxis().set_visible(False)
    s.text(
        0.05,
        0.95,
        'C',
        transform=s.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='black',
    )
    P.rcParams.update({'font.size': 16})
    # s.legend()
    P.rcParams.update({'font.size': 22})
    s.set_xticks([-1, 0, 1])

    # Plot the flipped phase Fourier Transform
    obj_Phase = np.fft.fftshift(fluo.coh_ft_double)
    phase = np.fft.fftshift(flipped)
    obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)
    obj_Phase = np.fft.ifft(obj_Phase, n=zero_pad)
    obj_Phase = np.fft.fftshift(obj_Phase)
    obj_scaled_x = np.fft.fftshift(
        np.fft.fftfreq(zero_pad, d=2 * 2 * fluo.kmax / (2 * fluo.num_pix))
    )
    # Harmonic inversion
    signal = np.sqrt(fluo.coh_ft_double) * np.exp(1j * flipped)
    signal = np.real(signal)
    inversion = harminv.invert(
        signal[fluo.num_pix :],
        fmin=0,
        fmax=1,
        dt=2 * 2 * fluo.kmax / (2 * fluo.num_pix),
        nf=6,
    )
    harm_x = -inversion.frequency[inversion.amplitude > 0.1]
    print('Truth', fluo.coords)
    # frequencies
    print('Frequencies', inversion.frequency[inversion.amplitude > 0.1])
    # decay rates
    print('Decay Rates', inversion.decay[inversion.amplitude > 0.1])
    print('Q Factor', inversion.Q[inversion.amplitude > 0.1])  # Q factor
    # phase shift
    print('Phase Shift', inversion.phase[inversion.amplitude > 0.1])
    # absolute amplitudes
    print('Amplitude', inversion.amplitude[inversion.amplitude > 0.1])

    s = fig.add_subplot(144)
    s.plot(
        obj_scaled_x,
        np.abs(obj_Phase) / np.max(np.abs(obj_Phase)),
        label='Flipped IFT',
        color='C2',
    )
    s.set_xlim([-1, 1])
    for x in fluo.coords:
        s.axvline(x=x, linestyle='dashed', color='C0')
    for x in harm_x:
        s.axvline(x=x, linestyle='dotted', color='C2')
    s.set_xlabel('X [Length]')
    s.set_ylabel('Brightness [a.u.]')
    # s.get_xaxis().set_visible(False)
    s.get_yaxis().set_visible(False)
    s.text(
        0.05,
        0.95,
        'D',
        transform=s.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='black',
    )
    P.rcParams.update({'font.size': 16})
    # s.legend()
    P.rcParams.update({'font.size': 22})
    s.set_xticks([-1, 0, 1])

    P.tight_layout()
    ax = s.axis()
    rec = P.Rectangle(
        (ax[0] - 4.85, ax[2] - 0.3),
        (ax[1] - ax[0]) + 5,
        (ax[3] - ax[2]) + 0.38,
        fill=False,
        lw=2,
        linestyle='solid',
    )
    rec = s.add_patch(rec)
    rec.set_clip_on(False)
    P.show()


def Figure_S4():
    np.random.seed(0x5EED + 2)
    fluo = Speckle_2D.Fluorescence2D(kmax=3, num_pix=101, num_atoms=5)

    np.random.seed(0x5EED + 2)
    fluo_coarse = Speckle_2D.Fluorescence2D(kmax=3, num_pix=15, num_atoms=5)

    P.rcParams.update({'font.size': 22})
    box_extent = 2 * fluo.kmax
    # Normal resolution object
    fig = P.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax1.imshow(
        unwrap_phase(fluo.coh_phase_double),
        origin='lower',
        extent=[-box_extent, box_extent, -box_extent, box_extent],
    )
    # ax1.get_xaxis().set_visible(False)
    # ax1.get_yaxis().set_visible(False)
    ax1.set_xlabel(r'$k_x$ [Length]$^{-1}$')
    ax1.set_ylabel(r'$k_y$ [Length]$^{-1}$')
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

    from skimage.transform import resize

    spline_order = 5
    ax2 = fig.add_subplot(222)
    ax2.imshow(
        unwrap_phase(fluo_coarse.coh_phase_double),
        origin='lower',
        extent=[-box_extent, box_extent, -box_extent, box_extent],
    )
    # ax2.imshow(resize(unwrap_phase(fluo_coarse.coh_phase_double), (2*fluo.num_pix-1,2*fluo.num_pix-1), order=spline_order), origin='lower')
    # ax2.imshow(pyramid_expand(unwrap_phase(fluo_coarse.coh_phase_double), upscale=(2*fluo.num_pix-1)/(2*fluo_coarse.num_pix-1), order=spline_order), origin='lower')
    # ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set_xlabel(r'$k_x$ [Length]$^{-1}$')
    # ax2.set_ylabel("Y [Length]")
    ax2.text(
        0.05,
        0.95,
        'B',
        transform=ax2.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='white',
    )

    # Plot the phase Fourier Transform
    obj_Phase = np.fft.fftshift(fluo.coh_ft_double)
    phase = np.fft.fftshift(fluo.coh_phase_double)
    obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)
    obj_Phase = np.fft.ifft2(obj_Phase)
    obj_Phase = np.fft.fftshift(obj_Phase)
    obj_scaled_x = np.fft.fftshift(
        np.fft.fftfreq(len(obj_Phase), d=2 * 2 * fluo.kmax / (2 * fluo.num_pix))
    )
    box_extent = np.max(
        np.fft.fftshift(
            np.fft.fftfreq(2 * fluo.num_pix, d=2 * fluo.kmax / fluo.num_pix)
        )
    )
    ax3 = fig.add_subplot(223)
    ax3.imshow(
        np.abs(obj_Phase),
        origin='lower',
        extent=[-box_extent, box_extent, -box_extent, box_extent],
        cmap='gray',
    )
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    # ax3.get_xaxis().set_visible(False)
    # ax3.get_yaxis().set_visible(False)
    ax3.set_xlabel('X [Length]')
    ax3.set_ylabel('Y [Length]')
    ax3.text(
        0.05,
        0.95,
        'C',
        transform=ax3.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='white',
    )

    # Plot the phase Fourier Transform
    obj_Phase = np.fft.fftshift(fluo.coh_ft_double)
    phase = np.fft.fftshift(
        resize(
            unwrap_phase(fluo_coarse.coh_phase_double),
            (2 * fluo.num_pix - 1, 2 * fluo.num_pix - 1),
            order=spline_order,
        )
    )
    # phase = np.fft.fftshift(pyramid_expand(unwrap_phase(fluo_coarse.coh_phase_double), upscale=(2*fluo.num_pix-1)/(2*fluo_coarse.num_pix-1), order=spline_order))
    obj_Phase = np.abs(obj_Phase) * np.exp(1j * phase)
    obj_Phase = np.fft.ifft2(obj_Phase)
    obj_Phase = np.fft.fftshift(obj_Phase)
    obj_scaled_x = np.fft.fftshift(
        np.fft.fftfreq(len(obj_Phase), d=2 * 2 * fluo.kmax / (2 * fluo.num_pix))
    )
    box_extent = np.max(
        np.fft.fftshift(
            np.fft.fftfreq(2 * fluo.num_pix, d=2 * fluo.kmax / fluo.num_pix)
        )
    )
    ax4 = fig.add_subplot(224)
    ax4.imshow(
        np.abs(obj_Phase),
        origin='lower',
        extent=[-box_extent, box_extent, -box_extent, box_extent],
        cmap='gray',
    )
    ax4.set_xlim([-1, 1])
    ax4.set_ylim([-1, 1])
    # ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4.set_xlabel('X [Length]')
    ax4.text(
        0.05,
        0.95,
        'D',
        transform=ax4.transAxes,
        fontsize=22,
        fontweight='bold',
        va='top',
        c='white',
    )

    P.tight_layout()
    ax = ax4.axis()
    rec = P.Rectangle(
        (ax[0] - 0.15, ax[2] - 0.5),
        (ax[1] - ax[0]) + 0.3,
        (ax[3] - ax[2]) + 3.3,
        fill=False,
        lw=2,
        linestyle='solid',
    )
    rec = ax4.add_patch(rec)
    rec.set_clip_on(False)
    P.show()
