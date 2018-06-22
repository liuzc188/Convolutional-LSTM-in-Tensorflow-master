import matplotlib
from matplotlib.pylab import *    
import numpy
import scipy
from numpy import *                
from scipy import *               
import pdb
text_load = load
from std import *



try: del set
except: pass


#No need to overload dot anymore.
#dot_std = dot
#def dot(A,B):
#    "dot(A,B): works the usual way for ndarrays, but in a user-defined way when A is a vec. "
#    assert(isinstance(B, ndarray))
#
#    if isinstance(A, ndarray):
#        return dot_std(A,B)
#    else:
#        return A.dot(B)


shape_std=shape
def shape(A):
    if isinstance(A, ndarray):
        return shape_std(A)
    else:
        return A.shape()

size_std = size
def size(A):
    if isinstance(A, ndarray):
        return size_std(A)
    else:
        return A.size()

det = linalg.det
