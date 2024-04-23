#!/usr/bin/env python

import numpy as np
import ase


class Xtal:
    """Xtal class representing crystal structure"""

    def __init__(self, super_lattice=(2, 2, 2)):
        """Xtal class initializer

        Args:
            super_lattice (tuple): Size of super lattice
        """
        import ase.lattice.cubic as alc
        import ase.lattice.hexagonal as alh

        # self.atoms = alc.FaceCenteredCubic(size=super_lattice,
        #                                   symbol='Cu',
        #                                   pbc=(1, 1, 1))

        self.atoms = alh.HexagonalClosedPacked(size=super_lattice,
                                               symbol='Cu',
                                               pbc=(1, 1, 1),
                                               latticeconstant=(2., 2))

        # Convert positions to meters
        self.atoms.set_cell(np.identity(3) * self.atoms.get_cell())

    def get_positions(self):
        """Get atomic positions in meters

        Returns:
            np.ndarray: Array of atomic positions
        """
        return self.atoms.get_positions()

    def get_number_atoms(self):
        """Get number of atoms

        Returns:
            int: Number of atoms
        """
        return len(self.atoms)
