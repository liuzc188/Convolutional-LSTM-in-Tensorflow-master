from pylab import *
from std import *
from base_mat import base_mat
class std_mat(base_mat):

    def __init__(self, v, h, init=True):
        self.v = v
        self.h = h

        if init:
            self.w = [randn(v, h)]

    def soft_copy(self): 
        s = std_mat(self.v, self.h, init=False)

        s.w = [x for x in self.w] 

        s.TR = self.TR
        return s

    def __mul__(self, x):
        if not self.TR:
            return dot(x, self.w[0])
        else:
            return dot(x, self.w[0].T)

    def outp(self, v, h):
        if self.TR:
            print 'shucks'
            tmp=v; v=h; h=tmp;

        s = self.soft_copy() 
        s.w[0] =  dot(v.transpose(), h)
        return s

    def show_W(self):
        def nor(x): 
            a = x-x.min()
            return a/float(a.max())
        try:
            if not self.TR:
                return print_aligned(nor(self.w[0]))
            else:
                return print_aligned(nor(self.w[0].transpose()))
        except:
            return self.w[0]

