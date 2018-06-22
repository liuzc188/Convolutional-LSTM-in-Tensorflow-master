from pylab import *
from std import *
from base_mat import base_mat

class std_mat_m(base_mat):

    def __init__(self, v, h, m, init=True):
        self.v = v
        self.h = h
        self.m = m
        if init:
            self.w = [randn(v, h)]
        assert(self.v % self.m == 0)

    def soft_copy(self): 
        s = std_mat_m(self.v, self.h, self.m, init=False)

        s.w = [x for x in self.w] 

        s.TR = self.TR
        return s

    def __mul__(self, x):
        if not self.TR:
            return dot(x, self.w[0])
        else:
            return dot(x, self.w[0].transpose())

    def outp(self, v, h):
        if self.TR: tmp=v; v=h; h=tmp;

        s = self.zero_copy() 
        s.w[0] =  dot(v.transpose(), h)
        return s

    def show_W(self):
        def print_aligned_m(W):
            L = []
            v = self.v / self.m
            assert(v*self.m == self.v)

            for u in range(self.m):
                L.append(print_aligned(W[v*u:v*(u+1),:]))
            C = concatenate(L,1)
            return C

        def nor(x): 
            a = x-x.min()
            return a/float(a.max())
        if not self.TR:
            return print_aligned_m(nor(self.w[0]))
        else:
            return print_aligned_m(nor(self.w[0].transpose()))

