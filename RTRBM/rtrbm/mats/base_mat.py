from mats import up, down

from pylab import *


class base_mat:
    TR = False
    w  = None
    direction = None

    v = None
    h = None

    #def __init__(self, w):
    #    self.w=w

    def soft_copy(self):
        raise Exception ("base_mat: soft_copy: unimplemented error")

    def copy(self):
        s = self.soft_copy()
        s.w = [x.copy() for x in self.w]
        return s
    
    def zero_copy(self):
        "New style: use 0*x."
        return 0 * self.copy()
    
    def __add__(self, a):
        s = self.copy()
        s += a
        return s
            
    def __iadd__(self, a):
        if isscalar(a):
            for i in xrange(len(self.w)):
                self.w[i]+=a
        else:
            assert(len(self.w)==len(a.w))
            for i in xrange(len(self)):
                self.w[i]+=a.w[i]
        return self
        
    def __sub__(self, a):
        s = self.copy()
        s -= a
        return s

    def __isub__(self, a): 
        if isscalar(a):
            for i in xrange(len(self.w)):
                self.w[i]-=a
        else:
            assert(len(self.w)==len(a.w))
            for i in xrange(len(self)):
                self.w[i]-=a.w[i]

        return self

    def __imul__(self, C):  
        if isscalar(C):
            for i in xrange(len(self.w)):
                self.w[i]*=C
        else:
            assert(len(self.w)==len(C.w))
            for i in xrange(len(self.w)):
                self.w[i]*=C.w[i]
        return self

    def __rmul__(self, c):
        s = self.copy()
        s *= c
        return s

    def __radd__(self, c):
        s = self.copy()
        s += c
        return s
    

    def transpose(self):
        a = self.soft_copy()
        a.TR = not a.TR
        return a

    def T(self):
        "same as transpose"
        a = self.soft_copy()
        a.TR = not a.TR
        return a

    def show_W(self): 
        # TODO: add rescaling of images, so that matrices of
        # different dimensions could be plotted. 
        conlist = [x.show_W() for x in self.w]
        return concatenate(conlist)
        
    def get_W0(self): 
        if self.w[0].__class__==ndarray:
            return self.w[0]
        else:
            return self.w[0].get_W0()


    def __ipow__(self, C, modulo=None): 
        if isscalar(C):
            for i in xrange(len(self.w)):
                self.w[i]**=C
        else:
            assert(len(self.w)==len(C.w))
            for i in xrange(len(self.w)):
                self.w[i]**=C.w[i]
        return self

    def __pow__(self, c, modulo=None):
        s = self.copy()
        s**=c
        return s

    def sum(self):
        return sum(x.sum() for x in self.w)

    def mean(self):
        return self.sum() / float(self.size())


    def __repr__(self):
        return '<[\n'+'\n'.join([x.__repr__() for x in self.w])+']>\n'

    def flatten(self):
        z = concatenate(tuple(x.flatten() for x in self.w))
        return z

    def size(self): #unfortunately, this is not really ndarray compatible; they use directly size
        self.siz = sum(size_aux(x) for x in self.w)
        return self.siz

    def max(self):
        return max(x.max() for x in self.w)
    
    def min(self):
        return min(x.min() for x in self.w)
    
    def all(self):
        for x in self:
            if not x.all():
                return False
        return True

    def any(self):
        for x in self:
            if x.any():
                return True
        return False
            
    def unpack(self, x): #the inverse of flatten.
        a, b = 0, 0
        d = self.zero_copy()
        for i in range(len(self)):
            b += size_aux(self[i])
            x1 = x[a:b]

            if isinstance(self[i],numpy.ndarray):
                d[i]=x1.reshape(*shape(self[i])).copy()
            else:
                d[i]=self[i].unpack(x1)
            a = b
        return d

    def __abs__(self):
        d = self.copy()
        for i in range(len(d.w)):
            d.w[i]=abs(d.w[i])
        return d

    def sign(self):
        d = self.copy()
        for i in range(len(d.w)):
            if isinstance(d.w[i], ndarray):
                d.w[i]=sign(d.w[i])
            else:
                d.w[i]=d.w[i].sign()
        return d


    def __contains__(self, y):
        for x in self.w:
            if y in x: return True
        return False

    def __eq__(self, y):
        e = self.copy()
        if not isscalar(y):
            assert(len(y)==len(self.w))
            for i in xrange(len(self.w)):
                e.w[i]=(self.w[i]==y.w[i])
        else:
            for i in xrange(len(self.w)):
                e.w[i]=self.w[i]==y
        return e

    def __le__(self, y):
        e = self.copy()
        if not isscalar(y):
            assert(len(y.w)==len(self.w))
            for i in xrange(len(self.w)):
                e.w[i]=(self.w[i].__le__(y.w[i]))
        else:
            for i in xrange(len(self)):
                e.w[i]=self.w[i].__le__(y)
        return e

    def __lt__(self, y):
        e = self.copy()
        if not isscalar(y):
            assert(len(y.w)==len(self.w))
            for i in xrange(len(self.w)):
                e.w[i]=(self.w[i].__lt__(y.w[i]))
        else:
            for i in xrange(len(self)):
                e.w[i]=self.w[i].__lt__(y)
        return e

    def __ge__(self, y):
        e = self.copy()
        if not isscalar(y):
            assert(len(y)==len(self))
            for i in xrange(len(self)):
                e[i]=(self[i].__ge__(y[i]))
        else:
            for i in xrange(len(self.w)):
                e.w[i]=self.w[i].__ge__(y)
        return e

    def __gt__(self, y):
        e = self.copy()
        if not isscalar(y):
            assert(len(y.w)==len(self.w))
            for i in xrange(len(self.w)):
                e.w[i]=(self.w[i].__gt__(y.w[i]))
        else:
            for i in xrange(len(self.w)):
                e.w[i]=self.w[i].__gt__(y)
        return e

    def bool2float(self):
        """
        transform a bool array to 0., 1. array.
        works assuming the 'leafs' are numpy.ndarrays
        """
        e = self.copy()
        for i in range(len(self)):
            if isinstance(self[i], numpy.ndarray):
                e[i]=array(self[i],'d')
            else:
                e[i]=self[i].bool2float()
        return e

    def float2bool(self):
        """
        if self is 0,1 array, return a similar array with true/false
        values, for indexing. assums that 'leafs' are numpy.ndarrays.
        """
        e = self.copy()
        for i in range(len(self)):
            if isinstance(self.w[i], numpy.ndarray):
                e[i]=array(self.w[i],'bool')
            else:
                e[i]=self.w[i].float2bool()
        return e
        
    
    def __getitem__(self, item):
        "__getitem__: slow, copy full array. returns numpy.ndarray"
        if isinstance(item, base_mat):
            assert(len(self)==len(item))
            tmp = [v[i].flatten() for v,i in zip(self.w, item.w)]
            return concatenate(tmp, 0)
        elif isscalar(item):
            return self.w[item]
        else:
            return self.flatten()[item]
    

    def set(self, value):
        self *= 0
        self += value
        return self
    def set_unpack(self, value):
        self *= 0
        self += self.unpack(value)
        return self

            

    def __setitem__(self, items, vals):
        "__setitem__:  slow, copies all array."
        if isinstance(items, base_mat):
            assert(len(self.w)==len(items.w))
            if not isscalar(vals): #then vals is a flat array. 
                self0 = self.flatten() #then modify self. 
                inds0 = array(items.flatten() ,'bool') 
                self0[inds0]=vals 
                self.w = self.unpack(self0).w
                
            else:
                for i in xrange(len(self)):
                    self.w[i][items.w[i]]=vals
        elif isscalar(items):
            self.w[items]=vals
        else: 
            f = self.flatten()
            f[items]=vals
            self.w = self.unpack(f).w
            
        return self

    def __len__(self):
        return len(self.w)

    

    def outp(self, V, H):
        raise Exception("outp is not implemented!")

    def outp_up(self, V, H):
        return self.outp(V,H)
    
    def outp_down(self, V, H):
        return self.outp(V,H)

    def __neg__(self):
        return (-1)*self

    def shape(self):
        return (self.v, self.h)

def size_aux(x):

    if isinstance(x,numpy.ndarray): return x.size
    return x.size()

