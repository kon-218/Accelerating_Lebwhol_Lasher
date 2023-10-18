from cython.CythonLebwohlLasher import initdat, one_energy, all_energy
import numpy as np

def test_initdat():
    nmax = 5
    lattice = initdat(nmax)
    assert lattice.shape == (nmax, nmax)

def test_one_energy():
    nmax = 5
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    en = one_energy(arr, 1, 1, nmax)
    assert isinstance(en, float)

def test_all_energy():
    # Define a specific lattice configuration with known energy
    nmax = 3
    arr = np.array([[0.0, 1.0, 2.0],
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0]], dtype=np.float64)
    expected_energy = 4.373435645621068  # This is the expected energy for the given arr

    # Calculate the energy using the function
    calculated_energy = all_energy(arr, nmax)

    # Assert that the calculated energy matches the expected energy
    assert calculated_energy == expected_energy