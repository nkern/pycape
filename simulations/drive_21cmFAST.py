"""
drive_21cmFAST.py : driver for 21cmFAST simulation
See Mesinger et al. 2011MNRAS.411..955M
"""

import os, sys, numpy as np
from .DictEZ import create as ezcreate

class drive_21cmFAST():

	def __init__(self,dic):
		self.__dict__.update(dic)

	def init_machine(self):
		"""
		- initialize how simulation will be run: on a single processor, on a cluster, etc.
		"""
		pass

	def init_files(self):
		"""
		- build necessary directories and files for jobs to be run
		"""
		pass

	def send_jobs(self):
		"""
		- run simulation
		"""
		pass

	def collect_results(self):
		"""
		- load results from various runs and print to file
		"""
		pass



