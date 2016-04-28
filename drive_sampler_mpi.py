"""
used in conjunction with toolbox.drive_sampler_mpi() to run sampler in parallel
"""
from mpi4py import MPI
import sys
import emcee
import cPickle as pkl

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
sys.stdout.write("...running mpi process %d of %d on %s.\n"% (rank, size, name))

# Load workspace and initial positions of walkers
file = open('Workspace.pkl','rb')
input = pkl.Unpickler(file)
W = input.load()['W']
file.close()

# Initialize sampler
W.sampler_init(W.sampler_init_kwargs)

# Run chains
W.drive_sampler(W.pos[rank],step_num=W.step_num,burn_num=W.burn_num,**W.sampler_kwargs)

# Output chains
file = open(W.dir_pycape+'/mpi_chains/mpi_chain_rank%s.pkl'%rank,'wb')
output = pkl.Pickler(file)
output.dump({'rank%s_chain'%rank:W.S.sampler.chain})
file.close()
