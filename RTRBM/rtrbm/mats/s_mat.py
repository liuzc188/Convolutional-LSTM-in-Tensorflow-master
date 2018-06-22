"""
s_mat is a transformer that receives a set of matrices and essentially computes their
sum. It is extremely convenient when we want, e.g., a patch and a convolutional matrix,
or a convolutional and a sparse matrix or even more of them. 
"""
from mats.base_mat import base_mat
from pylab import equal, concatenate, array

class s_mat(base_mat):
    def __init__(self, w_list):
        assert( array([x.h==w_list[0].h for x in w_list]).all())
        assert( array([x.v==w_list[0].v for x in w_list]).all())

        self.w = w_list
        self.h = w_list[0].h
        self.v = w_list[0].v
    def soft_copy(self):
        m = s_mat(self.w[:])
        m.TR = self.TR
        return m 

    def __mul__(self, X):
        if not self.TR:  # do the usual thing
            H = self.w[0] * X
            for i in range(1, len(self.w)):
                H += self.w[i] * X
            return H
        else:
            V = self.w[0].T() * X
            for i in range(1, len(self.w)):
                V += self.w[i].T() * X
            return V
    def outp(self, V, H):
        if self.TR:
           TMP= 1*V
           V=H
           H=TMP

        o = 0 * self
        ### this is the s-mat, a sum_matrix. Great!
        for i in range(len(self)):
            o.w[i] = self.w[i].outp(V, H)

        return o

