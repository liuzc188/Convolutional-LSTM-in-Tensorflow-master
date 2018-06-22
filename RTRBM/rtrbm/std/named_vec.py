from pylab import *

class named_vec:
    def __init__(self, **d):
        self.d = d
    


class named_vec_not_ready:
    def __init__(self, **d):
        object.__setattr__(self,'d',d)

    def d(self):
        return object.__getattribute__(self,d)

    def keys(self):
        return self.d.keys()

    def has_key(self,k):
        return self.d.has_key(k)
        
    def soft_copy(self):
        X = named_vec(**self.d)
        object.__setattr__(X,'d',dict())
        for k in self.d.keys():
            X.d[k] = self.d[k]
        return X

    def __getattribute__(self, attr):
        d = object.__getattribute__(self, 'd')
        if attr=='d':
            return d

        

        if d.has_key(attr):
            return self.d[attr]

        else:
            return object.__getattribute__(self, attr)
            
    def __getitem__(self, item):
        "__getitem__: slow, copy full array. returns numpy.ndarray"
        d = object.__getattribute__(self, d)
        if d.has_key(item):
            return self.d[item]
        else:
            return object.__getattribute__(self, item)

    def __setattr__(self, item, val):
        if item=='d':
            object.___setattr__(self,'d',val)
        else:
            self.d[item]=val

    def __setitem__(self, item, val):
        "__getitem__: slow, copy full array. returns numpy.ndarray"
        self.d[item]=val




    def copy(self):
        X = self.soft_copy()
        for k in X.d.keys():
            X[k] = X[k].copy()

    def __add__(self, a):
        s = self.copy()
        s += a
        return s
    def __iadd__(self, a):
        if isscalar(a):
            for k in self.d.keys():
                self.d[k]+=a
        else:
            for k in self.d.keys():
                self.d[k]+=a.d[k]
            for k in set(a.d.keys()) - set(self.d.keys()):
                self.d[k]=a.d[k]
        return self
    

    def __sub__(self, a):
        s = self.copy()
        s -= a
        return s
    def __isub__(self, a):
        if isscalar(a):
            for k in self.d.keys():
                self.d[k]-=a
        else:
            for k in self.d.keys():
                self.d[k]-=a.d[k]
            for k in set(a.d.keys()) - set(self.d.keys()):
                self.d[k]=-a.d[k]
        return self
        

    def __imul__(self, a):
        if isscalar(a):
            for k in self.d.keys():
                self.d[k]*=a
        else:
            for k in self.d.keys():
                self.d[k]*=a.d[k]
            for k in set(a.d.keys()) - set(self.d.keys()):
                self.d[k]=a.d[k] ### unclear: Do I want it to do what?
        return self


    def __rmul__(self, c):
        s = self.copy()
        s *= c
        return s

    def __radd__(self, c):
        s = self.copy()
        s += c
        return s
    
    
    ### hacks
    def transpose(self):
        a = self.soft_copy()
        a.TR = not a.TR
        return a
    def T(self):
        "same as transpose"
        a = self.soft_copy()
        a.TR = not a.TR
        return a

    
    def __pow__(self, a):
        s = self.copy()
        s **= a
        return s
    def __ipow__(self, a):
        if isscalar(a):
            for k in self.d.keys():
                self.d[k]**=a
        else:
            for k in self.d.keys():
                self.d[k]**=a.d[k]
            for k in set(a.d.keys()) - set(self.d.keys()):
                self.d[k]=1 #-a.d[k]
        return self
    
    def __repr__(self):
        return '<[\n'+'\n'.join(['%s: %s' % (k, x.__repr__()) for (k,x) in self.d.iteritems()])+']>\n'

    def flatten(self):
        z = concatenate(tuple(x.flatten() for x in self.d.itervalues()))
        return z

    
    def max(self):
        return max(x.max() for x in self.d.itervalues())
    
    def min(self):
        return min(x.min() for x in self.d.itervalues())

    def all(self):
        for x in self.d.itervalues():
            if not x.all():
                return False
        return True

    def any(self):
        for x in self.d.itervalues():
            if x.any():
                return True
        return False


    def size(self): #unfortunately, this is not really ndarray compatible; they use directly size
        self.siz = sum(size_aux(x) for x in self.d.itervalues())
        return self.siz


    def unpack(self, x): #the inverse of flatten.
        a, b = 0, 0
        d = self.zero_copy()
        for k in self.d.iterkeys():
            b += size_aux(self.d[k])
            x1 = x[a:b]

            if isinstance(self[i],numpy.ndarray):
                d.d[k]=x1.reshape(*shape(self[k])).copy()
            else:
                d.d[k]=self.d[k].unpack(x1)
            a = b
        return d

    
    def __abs__(self):
        d = self.copy()
        for i in d.d.iterkeys():
            d.d[i]=abs(d.d[i])
        return d

    def __contains__(self, y):
        for x in self.d.values():
            if y in x: return True
        return False


    def __eq__(self, y):
        e = self.copy()
        if not isscalar(y):
            for k in self.d.iterkeys():
                e.w[i]=(self.d[k]==y.d[k])
        else:
            for k in self.d.iterkeys():
                e.w[k]=self.w[k]==y
        return e

    def __le__(self, y):
        e = self.copy()
        if not isscalar(y):
            for k in self.d.iterkeys():
                e.w[i]=(self.d[k].__le__(y.d[k]))
        else:
            for k in self.d.iterkeys():
                e.w[k]=self.w[k].__le__(y)
        return e


    def __lt__(self, y):
        e = self.copy()
        if not isscalar(y):
            for k in self.d.iterkeys():
                e.w[i]=(self.d[k].__lt__(y.d[k]))
        else:
            for k in self.d.iterkeys():
                e.w[k]=self.w[k].__lt__(y)
        return e

    def __ge__(self, y):
        e = self.copy()
        if not isscalar(y):
            for k in self.d.iterkeys():
                e.w[i]=(self.d[k].__ge__(y.d[k]))
        else:
            for k in self.d.iterkeys():
                e.w[k]=self.w[k].__ge__(y)
        return e


    def __gt__(self, y):
        e = self.copy()
        if not isscalar(y):
            for k in self.d.iterkeys():
                e.w[i]=(self.d[k].__gt__(y.d[k]))
        else:
            for k in self.d.iterkeys():
                e.w[k]=self.w[k].__gt__(y)
        return e

    def __len__(self):
        return len(self.d.keys())

    



def size_aux(x):

    if isinstance(x,numpy.ndarray): return x.size
    return x.size()
