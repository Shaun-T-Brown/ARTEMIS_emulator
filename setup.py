from distutils.core import setup

setup(name='ARTEMIS_emulator',
      version='0.1',
      description='Python package for using an building ARTMEIS cosmological emulators',
      author='Shaun T. Brown',
      author_email='shaunb866@outlook.com',
      url='',
      py_modules=['src/'],

      requires = ['numpy',
                  'scipy',
                  'matplotlib',
                  'dill',
                  'h5py',

                  ]
     )



import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import dill
import h5py as h5
import eagle_IO.eagle_IO as E
import os 
from matplotlib.colors import LogNorm
from itertools import compress