"""
Basic Python Lebwohl-Lasher code.  Based on the paper
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
 
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""

from mpi4py import MPI
import sys
import time
import datetime
import numpy as np
from numba import guvectorize, float64, vectorize, int64, jit, njit
import numba as nb
from math import exp

# import matplotlib.pyplot as plt
# import matplotlib as mpl

#=======================================================================
@njit()
def initdat(nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
      initialised with random orientations in the range [0,2pi].
    Returns:
      arr (float(nmax,nmax)) = array to hold lattice.
    """
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================
# def plotdat(arr,pflag,nmax):
#     """
#     Arguments:
#     arr (float(nmax,nmax)) = array that contains lattice data;
#     pflag (int) = parameter to control plotting;
#       nmax (int) = side length of square lattice.
#     Description:
#       Function to make a pretty plot of the data array.  Makes use of the
#       quiver plot style in matplotlib.  Use pflag to control style:
#         pflag = 0 for no plot (for scripted operation);
#         pflag = 1 for energy plot;
#         pflag = 2 for angles plot;
#         pflag = 3 for black plot.
#     The angles plot uses a cyclic color map representing the range from
#     0 to pi.  The energy plot is normalised to the energy range of the
#     current frame.
#   Returns:
#       NULL
#     """
#     if pflag==0:
#         return
#     u = np.cos(arr)
#     v = np.sin(arr)
#     x = np.arange(nmax)
#     y = np.arange(nmax)
#     cols = np.zeros((nmax,nmax))
#     if pflag==1: # colour the arrows according to energy
#         mpl.rc('image', cmap='rainbow')
#         for i in range(nmax):
#             for j in range(nmax):
#                 cols[i,j] = one_energy(arr,i,j,nmax)
#         norm = plt.Normalize(cols.min(), cols.max())
#     elif pflag==2: # colour the arrows according to angle
#         mpl.rc('image', cmap='hsv')
#         cols = arr%np.pi
#         norm = plt.Normalize(vmin=0, vmax=np.pi)
#     else:
#         mpl.rc('image', cmap='gist_gray')
#         cols = np.zeros_like(arr)
#         norm = plt.Normalize(vmin=0, vmax=1)

#     quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
#     fig, ax = plt.subplots()
#     q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
#     ax.set_aspect('equal')
#     plt.show()
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
      arr (float(nmax,nmax)) = array that contains lattice data;
      nsteps (int) = number of Monte Carlo steps (MCS) performed;
      Ts (float) = reduced temperature (range 0 to 2);
      ratio (float(nsteps)) = array of acceptance ratios per MCS;
      energy (float(nsteps)) = array of reduced energies per MCS;
      order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
    Returns:
      NULL
    """
    # Create filename based on current date and time
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")

    sys.stdout.write("#=====================================================\n")
    sys.stdout.write('# MPI Monte Carlo\n')
    sys.stdout.write('# Num Procs:           {:d}\n'.format(comm.Get_size()))
    sys.stdout.write("# File created:        {:s}\n".format(current_datetime))
    sys.stdout.write("# Size of lattice:     {:d}x{:d}\n".format(nmax, nmax))
    sys.stdout.write("# Number of MC steps:  {:d}\n".format(nsteps))
    sys.stdout.write("# Reduced temperature: {:5.3f}\n".format(Ts))
    sys.stdout.write("#=====================================================\n")
    # sys.stdout.write("# MPI MC step:  Ratio:     Energy:   Order:\n")
    # sys.stdout.write("#=====================================================\n")

    sys.stdout.write("# Run time (s):        {:8.6f}\n".format(runtime))

    # Write the columns of data
    for i in range(nsteps+1):
        line = "   {:05d}    {:f} {:12.4f}  {:6.4f} \n".format(i,ratio[i],energy[i],order[i])
        sys.stdout.write(line)
#=======================================================================
@njit(fastmath=True)
def one_energy(arr,ix,iy,nmax):
    """
    Arguments:
      arr (float(nmax,nmax)) = array that contains lattice data;
      ix (int) = x lattice coordinate of cell;
      iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
    Returns:
      en (float) = reduced energy of cell.
    """
    res = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#
    ang = arr[ix,iy]-arr[ixp,iy]
    res += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    res += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    res += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    res += 0.5*(1.0 - 3.0*np.cos(ang)**2)

    return res

#=======================================================================
@guvectorize([(float64[:,:], int64, float64[:,:])], '(n,m),()->(n,m)', target='cpu')
def all_energy(arr,nmax, result):
    """
    Arguments:
      arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
    Returns:
      enall (float) = reduced energy of lattice.
    """
    result[0][0] = 0.0
    for i in range(nmax):
        for j in range(nmax):
            result_energy = one_energy(arr,i,j,nmax)
            result[0][0] += result_energy
#=======================================================================
@njit(fastmath=True)
def get_order(arr,nmax):
    """
    Arguments:
      arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes.  Function returns S_lattice = max(eigenvalues(Q_ab)).
    Returns:
      max(eigenvalues(Qab)) (float) = order parameter for lattice.
    """
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()
#======================================================================
def MC_parallel_step(arr,Ts,nmax,comm):
    """
    Arguments:
      arr (float(nmax,nmax)) = array that contains lattice data;
      Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
      comm (MPI communicator) = MPI communicator.
    Description:
      Parallelized version of MC_step, each process handles a part of the lattice.
    Returns:
      accept/(local_nmax**2) (float) = acceptance ratio for current MCS.
    """
    scale = 0.1 + Ts
    accept = 0
    xran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    yran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    aran = np.random.normal(scale=scale, size=(nmax, nmax))

    rank = comm.Get_rank()
    size = comm.Get_size()
    local_nmax = nmax // size

    # Determine which rows of the lattice each process will work on
    start_row = rank * local_nmax
    end_row = (rank + 1) * local_nmax if rank != size - 1 else nmax

    local_accept = 0

    # Similar to MC_step, but loop over local section of lattice
    for i in range(start_row, end_row):
        for j in range(nmax):
            #print(i,j)
            ix = xran[i, j]
            iy = yran[i, j]
            ang = aran[i, j]
            en0 = one_energy(arr, ix, iy, nmax)
            arr[ix, iy] += ang
            en1 = one_energy(arr, ix, iy, nmax)
            if en1 <= en0:
                local_accept += 1
            else:
                boltz = exp(-(en1 - en0) / Ts)
                if boltz >= np.random.uniform(0.0,1.0):
                    local_accept += 1
                else:
                    arr[ix, iy] -= ang
    
    total_accept = comm.reduce(local_accept, op=MPI.SUM, root=0)

    if rank == 0:
        return total_accept / (nmax**2)

#=======================================================================

def main(program, nsteps, nmax, temp, pflag, comm):
    """
    Arguments:
        program (string) = the name of the program;
        nsteps (int) = number of Monte Carlo steps (MCS) to perform;
        nmax (int) = side length of square lattice to simulate;
        temp (float) = reduced temperature (range 0 to 2);
        pflag (int) = a flag to control plotting.
        comm (MPI communicator) = MPI communicator.
    Description:
        This is the main function running the Lebwohl-Lasher simulation.
    Returns:
        NULL
    """
    np.random.seed(42)
    rank = comm.Get_rank()
    #print(f"{rank} rank")
    # Create and initialise lattice
    lattice = initdat(nmax)
    result_view = np.zeros_like(lattice, dtype = float)

    if rank == 0:
        # Plot initial frame of lattice
        #plotdat(lattice,pflag,nmax)
        # Create arrays to store energy, acceptance ratio and order parameter
        energy = np.zeros(nsteps+1,dtype=np.dtype)
        ratio = np.zeros(nsteps+1,dtype=np.dtype)
        order = np.zeros(nsteps+1,dtype=np.dtype)
        # Set initial values in arrays
        all_energy(lattice,nmax, result_view)
        energy[0] = result_view[0][0]
        ratio[0] = 0.5 # ideal value
        order[0] = get_order(lattice,nmax)
        # Begin doing and timing some MC steps.
        initial = time.time()

    for it in range(1,nsteps+1):
        accept_ratio = MC_parallel_step(lattice,temp,nmax,comm)
        if rank == 0:
            ratio[it] = accept_ratio
            all_energy(lattice,nmax,result_view)
            energy[it] = result_view[0][0]
            order[it] = get_order(lattice,nmax)

    if rank == 0:
        final = time.time()
        runtime = final-initial
        # Final outputs
        print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
        # Plot final frame of lattice and generate output file
        savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
        #plotdat(lattice,pflag,nmax)




if __name__ == '__main__':
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, comm)
    else:
        if rank == 0:
            print("Usage: mpiexec -n <num_processes> python3 {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================