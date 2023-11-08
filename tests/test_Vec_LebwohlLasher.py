from LL_MPI_Vec import initdat, one_energy, all_energy, MC_step, get_order, get_order,main
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

    result = np.zeros_like(arr)
    # Calculate the energy using the function
    all_energy(arr, nmax, result)

    # Assert that the calculated energy matches the expected energy
    assert result[0][0] == expected_energy

def test_acceptance_ratio():
    # Create a dummy lattice
    nmax = 5
    arr = np.zeros((nmax, nmax))
    Ts = 0.5
    
    # Call MC_step function
    acceptance_ratio = MC_step(arr, Ts, nmax)
    
    # Check if acceptance ratio is between 0 and 1
    assert 0 <= acceptance_ratio <= 1
