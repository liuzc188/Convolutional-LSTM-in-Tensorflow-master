from std.basic import *
import sys

def printf(s):
    sys.stdout.flush()
    sys.stdout.write(s)

class itera:
    def __init__(self, list_of_sizes):
        self.list_of_sizes = list_of_sizes
        self.tot_len = sum(self.list_of_sizes)
        self.STATE = 0
        self.tot_len_arr = 2**(self.tot_len)
    def init(self):
        self.STATE = 0
    def is_done(self):
        return self.STATE > 2**(self.tot_len)-1
    def next(self):
        if self.is_done():
            raise Exception ("itera is done and will not proceed anymore")
        self.STATE+=1
    def get(self):
        if self.is_done():
            raise Exception ("itera is done and will not proceed anymore")
        STR = binary_repr(self.STATE).zfill(self.tot_len)
        lst = array(map(int,list(STR)),'i')
        b, a = 0, 0
        ANS = []
        for l in self.list_of_sizes:
            b += l
            ANS.append(lst[a:b])
            a = b
        return ANS

class iterm:
    def __init__(self, list_of_sizes):
        self.list_of_sizes = list_of_sizes
        self.tot_len = sum(self.list_of_sizes)
        self.STATE = 0
        self.tot_len_arr = prod(self.list_of_sizes)
    def init(self):
        self.STATE = 0
    def is_done(self):
        return self.STATE >= self.tot_len_arr
    def next(self):
        if self.is_done():
            raise Exception ("iterm is done and will not proceed anymore")
        self.STATE+=1
    def get(self):
        if self.is_done():
            raise Exception ("iterm is done and will not proceed anymore")
        st  = self.STATE
        ANS = []
        for l in self.list_of_sizes:
            cur_ii = st % l
            cur_bits = ieye(l, cur_ii)
            ANS.append(cur_bits)
            st /= l


        return ANS
