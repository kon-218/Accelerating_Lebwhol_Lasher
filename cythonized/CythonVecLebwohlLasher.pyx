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

""" """
import sys
import time
import datetime
import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib as mpl
cimport numpy as np
from cython.parallel cimport prange
from libc.math cimport exp, cos, sin, sqrt, pi
cimport libc.stdlib
from libc.stdlib cimport rand, RAND_MAX

from mpi4py import MPI
from mpi4py.MPI cimport Intracomm as IntracommType


#=======================================================================
cdef np.ndarray initdat(int nmax):
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
    arr = np.random.random_sample((nmax,nmax))*2.0*pi
    return arr



#=======================================================================

cdef plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()

#=======================================================================
cdef void savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
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
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "logs/LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================

cdef double one_energy(double[:, ::1] arr, int ix, int iy, int nmax):
    cdef double en = 0.0
    cdef int ixp = (ix+1)%nmax
    cdef int ixm = (ix-1)%nmax
    cdef int iyp = (iy+1)%nmax
    cdef int iym = (iy-1)%nmax
    cdef double ang

    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)
    ang = arr[ix,iy]-arr[ixp,iym]
    en += 0.5*(1.0 - 3.0*cos(ang)**2)

    return en

#=======================================================================
cdef double all_energy(double[:, ::1] arr,int nmax):
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
    cdef double enall = 0.0
    cdef int i,j
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall

#=======================================================================
cdef double get_order(double[:, ::1] arr,int nmax):
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
    cdef np.ndarray Qab = np.zeros((3,3))
    cdef np.ndarray delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    cdef np.ndarray lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    cdef int a,b,i,j
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = Qab/(2*nmax*nmax)
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()

#=======================================================================
cdef MC_parallel_step(np.ndarray arr,float Ts,int nmax, comm):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
    #
    # Pre-compute some random numbers.  This is faster than
    # using lots of individual calls.  "scale" sets the width
    # of the distribution for the angle changes - increases
    # with temperature.
    cdef double scale=0.1+Ts
    cdef double accept = 0
    cdef np.ndarray xran = np.random.randint(0,high=nmax, size=(nmax,nmax))
    cdef np.ndarray yran = np.random.randint(0,high=nmax, size=(nmax,nmax))
    cdef np.ndarray aran = np.random.normal(scale=scale, size=(nmax,nmax))
    cdef int i, j

    rank = comm.Get_rank()
    size = comm.Get_size()
    local_nmax = nmax // size

    # Determine which rows of the lattice each process will work on
    start_row = rank * local_nmax
    end_row = (rank + 1) * local_nmax if rank != size - 1 else nmax

    local_accept = 0

    for i in range(nmax):
        for j in range(nmax):
            ix = xran[i,j]
            iy = yran[i,j]
            ang = aran[i,j]
            en0 = one_energy(arr,ix,iy,nmax)
            arr[ix,iy] += ang
            en1 = one_energy(arr,ix,iy,nmax)
            if en1<=en0:
                local_accept += 1
            else:
            # Now apply the Monte Carlo test - compare
            # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                boltz = exp( -(en1 - en0) / Ts )

                if boltz >= random_uniform():
                    accept += 1
                else:
                    arr[ix,iy] -= ang
    
    # Combine acceptance ratios from all processes
    total_accept = comm.reduce(local_accept, op=MPI.SUM, root=0)

    if rank == 0:
        return accept/(nmax*nmax)


cdef double random_uniform() nogil:
    return <double>rand() / RAND_MAX

#=======================================================================
cpdef main(str program, int nsteps, int nmax, float temp, int pflag,comm):
    """
    Arguments:
      program (string) = the name of the program;
      nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
      temp (float) = reduced temperature (range 0 to 2);
      pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    # Create and initialise lattice
    cdef np.ndarray lattice = initdat(nmax)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print("Comm complete")
    print(rank)
    cdef np.ndarray result_view, energy, ratio, order
    cdef int it
    cdef float runtime
    cdef float initial, final

    if rank == 0:
        initial = MPI.Wtime()
        result_view = np.zeros_like(lattice, dtype = float)
        # Plot initial frame of lattice
        plotdat(lattice, pflag, nmax)
        # Create arrays to store energy, acceptance ratio and order parameter
        energy = np.zeros(nsteps+1, dtype=np.float64)
        ratio = np.zeros(nsteps+1, dtype=np.float64)
        order = np.zeros(nsteps+1, dtype=np.float64)
        # Set initial values in arrays
        energy[0] = all_energy(lattice,nmax)
        ratio[0] = 0.5 # ideal value
        order[0] = get_order(lattice, nmax)

        # Begin doing and timing some MC steps.

    for it in range(1,nsteps+1):
        accept_ratio = MC_parallel_step(lattice,temp,nmax,comm)
        if rank == 0:
            ratio[it] = accept_ratio
            energy[it] = all_energy(lattice,nmax)
            order[it] = get_order(lattice,nmax)
    
    if rank == 0:
        final = MPI.Wtime()
        runtime = final - initial
        # Final outputs
        print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax, nsteps, temp, order[nsteps-1], runtime))
        # Plot final frame of lattice and generate output file
        savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)
        plotdat(lattice, pflag, nmax)