"""
m_mat is a transformer that splits the input vector into chunks
and feeds it to the matrix. It ought to subsume std_mat_m
and bi_conv_mat. That's about it. 
"""
from mats.base_mat import base_mat
from pylab import equal, concatenate, array

class m_mat(base_mat):
    def __init__(self, w_list):
        assert( array([x.h==w_list[0].h for x in w_list]).all())
        self.v = sum(x.v for x in w_list)
        self.w = w_list
        self.h = w_list[0].h

    def soft_copy(self):
        m = m_mat(self.w[:])
        m.TR = self.TR
        return m 

    def __mul__(self, X):
        if not self.TR:  # do the usual thing
            a = 0
            b = self.w[0].v
            H = self.w[0] * X[:,a:b]
            for i in range(1, len(self)):
                a = b
                b += self.w[i].v
                H += self.w[i] * X[:,a:b]                
            return H
        else:
            return concatenate([x.transpose()*X for x in self.w], 1)
    def outp(self, V, H):
        if self.TR:
           TMP= 1*V
           V=H
           H=TMP
        o = 0 * self

        a = 0
        b = 0
        for i in range(len(self)):
            a = b
            b += self.w[i].v
            o.w[i] = self.w[i].outp(V[:,a:b],H)
        return o

