from mats.std_mat import std_mat
from mats.bias_up_mat import bias_up_mat

def bstd_mat(v,h):
    return bias_up_mat(std_mat(v,h))
