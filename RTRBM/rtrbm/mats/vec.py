
from mats.base_mat import base_mat


class vec(base_mat):
    def __init__(self, v):
        self.w = [v]

    def dot(self, w):
        if not self.TR: 
            return dot(v, w)
        else:
#its interpreted as an outer product. Yup.
# Otherwise the whole thing will fail.
            return dot(v.transpose())

    def soft_copy(self): 
        s = vec(self.w[0])
        s.TR = self.TR
        return s

    
