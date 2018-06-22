import rbm
from mats.base_mat import base_mat
from pylab import randn, sigmoid, Rsigmoid, newaxis, zeros
from mats.bias_mat import up, down

class rnn_trbm(base_mat):
    def __init__(self, VH, HH, CD_n, vis_gauss=False):
        self.vis_gauss = vis_gauss
        v, h =VH.v, VH.h
        assert(HH.h == HH.v == h)

        self.v, self.h = v, h

        self.w = [VH, HH, randn(h)] ## initial biases for all.

        self.CD_n = CD_n

    def soft_copy(self):
        VH, HH, b_init = self
        A = rnn_trbm(VH, HH, self.CD_n, self.vis_gauss)
        A[2] = self[2]
        return A

    def grad(self, V):
        T, v = V.shape
        assert(v == self.v)
        h = self.h
        VH, HH, b_init = self

        G = 0 * self
        d_VH, d_HH, d_b_init = G

        H = zeros((T, h))
        B = zeros((T, h))
        
        H[0] = sigmoid(VH * V[[0]]+ b_init[newaxis, :])

        for t in range(1, T):
            B[[t]] = HH*H[[t-1]]
            H[[t]] = sigmoid(VH*V[[t]] + B[[t]])    
    
        dB  = zeros((T, h))
        dBL = zeros((T, h))
        
        F_t = zeros(h)

        loss =0 

        VH_t = 1 * VH
        for t in reversed(range(T)):
            dB[t] = H[t] * (1-H[t]) * F_t
            
            VH_t[2] = B[t] + VH[2]  

            if self.CD_n > 0:
                dVH_t, dict_loss = rbm.rbm_grad_cd   (VH_t, V[[t]], self.CD_n, self.vis_gauss)
            else:
                dVH_t, dict_loss = rbm.rbm_grad_exact(VH_t, V[[t]], self.vis_gauss)
            loss += dict_loss['loss']
            
            dBL[t] = dVH_t[2] 
            
            d_VH += dVH_t



            HH.direction = up
            F_t[:] = HH.T() * (dB[[t]] + dBL[[t]])
            HH.direction = None


        d_b_init += dB[0]


        HH.direction = up
        VH.direction = up
        for t in range(1,T):
            d_HH += HH.outp_up(H[[t-1]], dB[[t]] + dBL[[t]])
            d_VH += VH.outp_up(V[[t]], dB[[t]])
        d_VH += VH.outp_up(V[[0]], dB[[0]])

        HH.direction = None
        VH.direction = None

        return G, dict(loss=loss)

    def show_W(self):
        return self[0].show_W()

    def sample(self, T, g, g0=None):
        if g0==None:
            g0 = g

        v, h = self.v, self.h
        VH, HH, b_init = self

        V = zeros((T, v))
        H = zeros((T, h))
        B = zeros((T, h))
        
        VH_t = 1*VH

        VH_t[2] = VH[2] + b_init
        V[[0]], H_t_stoch = rbm.sample(VH_t, g0, 1, self.vis_gauss)
        H[[0]] = sigmoid(VH_t * V[[0]])
        if self.vis_gauss:
            V[[0]] = VH_t.T() * H_t_stoch
        else:
            V[[0]] = sigmoid(VH_t.T() * H_t_stoch)
        for t in range(1, T):
            B[[t]] = HH*H[[t-1]]

            VH_t[2] = VH[2] + B[t]
            V[[t]], H_t_stoch = rbm.sample(VH_t, g, 1, self.vis_gauss)

            H[[t]] = sigmoid(VH_t * V[[t]])
            if self.vis_gauss:
                V[[t]] = VH_t.T() * H_t_stoch
            else:
                V[[t]] = sigmoid(VH_t.T() * H_t_stoch)
        return V
            
