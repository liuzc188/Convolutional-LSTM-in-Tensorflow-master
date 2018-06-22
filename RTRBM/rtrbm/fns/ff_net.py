
from std import *
from mats.base_mat import base_mat

# constructed from a sequence of feedforward networks.
class ff_net(base_mat):
    def __init__(self, w): 
        self.w = w

    def soft_copy(self):
        s = ff_net(self.w)
        s.w = [x for x in self.w]

        return s
    def __call__(self, V):

        for f in self:
            V=f(V)

        return V

    def grad(self, dE_dV): 
        d = 0*self
        for i in reversed(range(len(self))):
            d[i]=self[i].grad(dE_dV)
            dE_dV=self[i].dE_dY
        return d
            

class FFNet(base_mat):
    def __init__(self, *w):
        self.w = w
    def soft_copy(self):
        s = FFNet()
        s.w = [x for x in self.w]
        return s
    ## TODO: Complete this class.
        
