
from std import *
from mats.base_mat import base_mat

class sig_net(base_mat):
    def __init__(self, w):
        self.w = [w]
        self.V = None
        self.H = None

    def soft_copy(self):
        s = sig_net(None)
        s.w = [x for x in self.w]
        s.V, s.H = self.V, self.H

        return s

    def __call__(self, V):
        self.V = V
        self.H = sigmoid(self[0]*V)
        return self.H

    def grad(self, dE_dY, V=None):
        if V==None: V=self.V
        dE_dX = dE_dY * self.H * (1-self.H)

        d = self.copy()


        d[0]=self[0].outp_up(V, dE_dX)

        self.dE_dY= self[0].transpose()  * dE_dX

        return d
