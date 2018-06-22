from mats.base_mat import base_mat
from mats.std_mat import std_mat
from mats.bias_mat import bias_mat

from pylab import sigmoid, Rsigmoid, stochastic
class rbm(base_mat):
    def __init__(self, v, h, init=True):
        """
        We currently support only CD1
        """
        self.v = v; self.h = h
        if init:
            self.w = [bias_mat(std_mat(v,h))]
    def soft_copy(self):
        A = rbm(self.v, self.h)
        A.w = [x for x in self.w]
        return A


    def grad(self, V1):
        batch_size, v = V1.shape
        assert(v==self.v)
        self.V1 = V1

        W = self.w[0]
        G = 0 * self
        dW = G[0]

        H1 = sigmoid(W * V1)

        dW+= W.outp(V1, H1)

        H1 = stochastic(H1)
        
        V2 = Rsigmoid(W.T() * H1)
        H2 = sigmoid(W * V2)

        dW -= W.outp(V2, H2)
        
        recon = abs(V1 - V2).sum(1).sum()
        
        return G, dict(loss=recon)

    def show_stuff(self, mf=True):
        from pylab import concatenate, show, print_aligned
        sample_fn = [Rsigmoid, sigmoid][mf]
        W  = self[0]
        V1 = self.V1
        H1 = sample_fn(W * V1)
        V2 = sample_fn(W.T() * H1)        
        
        V_io = concatenate([[x,y] for x,y in zip(V1, V2)])
        
        show(print_aligned(V_io.T))
