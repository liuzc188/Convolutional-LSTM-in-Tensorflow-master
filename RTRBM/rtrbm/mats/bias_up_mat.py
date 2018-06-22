from mats.base_mat import base_mat
from pylab import randn, shape
from mats import up, down

class bias_up_mat(base_mat):
    def __init__(self, w, init=True): #, direction=None):
        "just give it a mat, and it will, very conveniently, add biases." 
        self.v, self.h = w.v, w.h
        # If it fails, try using std_mats instead.
        if init:
            self.w = [w, randn(self.h)]

    def soft_copy(self):
        m = bias_up_mat(self.w[0], init=False)
        m.w = [x for x in self.w]
        m.TR = self.TR
        return m

    def outp(self, V, H): 
        if self.TR:
            raise Exception ("bias_mat_up does not deal with transposed outerproducts")
            TMP = H.copy()
            H = V.copy()
            V = TMP

        d     = self.zero_copy()
        d[0]  = self[0].outp(V,H)
        d[1]  = H.sum(0)
        return d


    def __mul__(self, V):
        if not self.TR:
            batchsize, v = shape(V)
            return self[0]*V + self[1] 
        else:
            batchsize, h = shape(V)
            assert(h==self.h)            
            return self[0].transpose()*V
            
    def show_W(self):
        return self.w[0].show_W()
