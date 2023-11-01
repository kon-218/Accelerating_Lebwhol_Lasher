from Vectorized_LebwohlLasher import initdat, one_energy, all_energy, MC_step, get_order, get_order_vec,main
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

def test_acceptance_ratio():
    # Create a dummy lattice
    nmax = 5
    arr = np.zeros((nmax, nmax))
    Ts = 0.5
    
    # Call MC_step function
    acceptance_ratio = MC_step(arr, Ts, nmax)
    
    # Check if acceptance ratio is between 0 and 1
    assert 0 <= acceptance_ratio <= 1

def test_get_order():
    # Create a random lattice
    np.random.seed(0)
    arr = np.random.random((5, 5)) * 2 * np.pi
    
    # Calculate the order using the original function
    original_order = get_order(arr, 5)
    
    # Calculate the order using the new function
    new_order = get_order_vec(arr, 5)
    
    # Check if the orders match within a small tolerance (e.g., 1e-6)
    assert np.isclose(original_order, new_order, atol=1e-6)