from mats.base_mat import base_mat
from pylab import randn, shape
from mats import up, down

class bias_mat(base_mat):
    def __init__(self, w, init=True, direction=None):
        "just give it a mat, and it will, very conveniently, add biases." 
        self.v, self.h = w.v, w.h
        # If it fails, try using std_mats instead.
        if init:
            self.w = [w, randn(self.v), randn(self.h)]

        self.direction = direction
    def soft_copy(self):
        m = bias_mat(self.w[0], init=False)
        m.w = [x for x in self.w]
        m.TR = self.TR
        m.direction = self.direction
        return m

    def outp(self, V, H): 
        if self.TR:
            TMP = H.copy()
            H = V.copy()
            V = TMP

        d     = self.soft_copy()
        d.w[0]  = self.w[0].outp(V,H)
        
        if self.direction==None:
            d.w[1]  = V.sum(0)
            d.w[2]  = H.sum(0)
        if self.direction==up:
            d.w[1]  = 0 * V.sum(0)
            d.w[2]  = H.sum(0)
        if self.direction==down:
            d.w[1]  = V.sum(0)
            d.w[2]  = 0 * H.sum(0)
        return d


    def outp_up(self, V, H): 
        if self.TR:
            TMP = H.copy()
            H = V.copy()
            V = TMP

        d     = self.soft_copy()
        d.w[0]  = self.w[0].outp_up(V,H)
        d.w[1]  = 0 * self.w[1]
        assert(self.direction==up) 
        d.w[2]  = H.sum(0)                

        return d

    def outp_down(self, V, H): 
        if self.TR:
            TMP = H.copy()
            H = V.copy()
            V = TMP

        d     = self.soft_copy()
        d.w[0]  = self.w[0].outp_down(V,H)

        assert(self.direction==down) 
        d.w[1]  = V.sum(0)
        d.w[2]  = 0 * self.w[2]
        return d


    def __mul__(self, V):
        if not self.TR:
            batchsize, v = shape(V)
            assert(v==self.v)
            if self.direction!=down:
                return self.w[0]*V + self.w[2] 
            else:
                return self.w[0]*V
        else:
            batchsize, h = shape(V)
            assert(h==self.h)            
            if self.direction!=up: #i.e., we're not in feedforward mode, so we want transpose this way.
                return self.w[0].transpose()*V + self.w[1]
            else:
                return self.w[0].transpose()*V
            
    def show_W(self):
        return self.w[0].show_W()
