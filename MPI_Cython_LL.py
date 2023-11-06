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

import sys
import time 
from mpi4py import MPI

#import Cython.CythonLebwohlLasher as cythonLL
from cythonized.cythonized.CythonVecLebwohlLasher import main


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if int(len(sys.argv)) == 5:
    PROGNAME = sys.argv[0]
    ITERATIONS = int(sys.argv[1])
    SIZE = int(sys.argv[2])
    TEMPERATURE = float(sys.argv[3])
    PLOTFLAG = int(sys.argv[4])
    if rank == 0:
        initial = MPI.Wtime()
        print(initial)

    main(PROGNAME,ITERATIONS,SIZE,TEMPERATURE,PLOTFLAG,comm)
    
    if rank == 0:
        final = MPI.Wtime()
        time_taken = final - initial
        print(f"Time taken: {time_taken:.5f}s")
else:
    print("Usage: mpiexec -n <num_processes> python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
    #=======================================================================
