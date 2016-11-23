"""
used in conjunction with toolbox.drive_sampler_mpi() to run sampler in parallel
"""
from mpi4py import MPI
import sys
import emcee
import cPickle as pkl
import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)

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
W.samp_init(W.sampler_init_kwargs,lnprob_kwargs=W.lnprob_kwargs,sampler_kwargs=W.sampler_kwargs)

# Run chains
W.samp_drive(W.pos[rank],step_num=W.step_num,burn_num=W.burn_num)

# Output chains
file = open(W.dir_pycape+'/mpi_chains/mpi_chain_rank%s.pkl'%rank,'wb')
output = pkl.Pickler(file)
output.dump({'rank%s_chain'%rank:W.S.sampler.chain})
file.close()