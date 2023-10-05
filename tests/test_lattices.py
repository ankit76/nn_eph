import numpy as np
import pytest

from nn_eph import lattices


def test_size():
    n_sites = 4
    lattice = lattices.one_dimensional_chain(n_sites)
    assert len(lattice.sites) == 4
    assert len(lattice.bonds) == 4

    n_sites = 2
    lattice = lattices.one_dimensional_chain(n_sites)
    assert len(lattice.sites) == 2
    assert len(lattice.bonds) == 1

    l_x, l_y = 5, 4
    lattice = lattices.two_dimensional_grid(l_x, l_y)
    assert len(lattice.sites) == l_x * l_y
    assert len(lattice.shell_distances) == 6
    assert len(lattice.bonds) == 2 * l_x * l_y
    assert len(lattice.bond_shell_distances) == 11

    l_x, l_y, l_z = 3, 2, 2
    lattice = lattices.three_dimensional_grid(l_x, l_y, l_z)
    assert len(lattice.sites) == l_x * l_y * l_z
    assert len(lattice.shell_distances) == 4


def test_get_distance():
    n_sites = 4
    lattice = lattices.one_dimensional_chain(n_sites)
    assert lattice.get_distance((0,), (3,)) == 1
    assert (
        lattice.get_bond_distance(
            (
                0,
                0,
            ),
            (
                0,
                3,
            ),
        )
        == 1
    )

    l_x, l_y = 5, 4
    lattice = lattices.two_dimensional_grid(l_x, l_y)
    assert lattice.get_distance((0, 0), (1, 4)) == 2
    assert lattice.get_bond_distance((1, 0, 0), (0, 3, 4)) == 4

    l_x, l_y, l_z = 3, 2, 2
    lattice = lattices.three_dimensional_grid(l_x, l_y, l_z)
    assert lattice.get_distance((0, 0, 0), (0, 0, 2)) == 1


def test_get_neighbors():
    n_sites = 4
    lattice = lattices.one_dimensional_chain(n_sites)
    assert np.all(lattice.get_neighboring_bonds((0,)) == np.array([(0, 3), (0, 0)]))
    assert lattice.get_neighboring_sites((0, 1)) == [(1,), (2,)]

    # n_sites = 2
    # lattice = lattices.one_dimensional_chain(n_sites)
    # assert list(lattice.get_neighboring_bonds(1)) == [ 0 ]

    l_x, l_y = 5, 4
    lattice = lattices.two_dimensional_grid(l_x, l_y)
    assert len(list(lattice.get_neighboring_bonds((0, 0)))) == 4


if __name__ == "__main__":
    test_size()
    test_get_distance()
    test_get_neighbors()
