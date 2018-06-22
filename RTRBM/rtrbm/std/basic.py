from pylab import *
from loader import gload, load, save, gsave, fload

gray()

def LOG(x):
    return log(x+1e-300)



def show_mat(x,vmin=None,vmax=None):
    hold(False)
    imshow(x,interpolation='nearest',vmin=vmin,vmax=vmax)


def show(V,vmin=None,vmax=None):
    if V.ndim==1:
        res = int(sqrt(shape(V)[0]))
        if abs(res**2 - shape(V)[0])>0.00001: #i.e., if its not an integer
            print "check the dimensionality of the vector, not a square"
        show_mat(V.reshape(res, res),vmin,vmax)
    if V.ndim==2:
        show_mat(V,vmin,vmax)
    if V.ndim==3:
        imshow(V,interpolation='nearest',vmin=vmin, vmax=vmax)
    if V.ndim>3:
        print "don't know how to print such a vector!"

def show_seq(V,vmin=None,vmax=None):
    T   = len(V)
    res = int(sqrt(shape(V)[1]))
    for t in range(T):
        print t
        show(V[t],vmin,vmax)



def unsigmoid(x): return log (x) - log (1-x)


def print_aligned(w):
    n1 = int(ceil(sqrt(shape(w)[1])))
    n2 = n1
    r1 = int(sqrt(shape(w)[0]))
    r2 = r1
    Z = zeros(((r1+1)*n1, (r1+1)*n2), 'd')
    i1, i2 = 0, 0
    for i1 in range(n1):
        for i2 in range(n2):
            i = i1*n2+i2
            if i>=shape(w)[1]: break
            Z[(r1+1)*i1:(r1+1)*(i1+1)-1, (r2+1)*i2:(r2+1)*(i2+1)-1] = w[:,i].reshape(r1,r2)
    return Z


def aux(a):
    (s1, s2)=shape(a)
    mu=.5*ones((s1,1),'d')
    return concatenate((a,mu),1)
def aux2(a):
    (s1, s2)=shape(a)
    mu=.5*ones((1,s2),'d')
    return concatenate((a,mu),0)


def print_seq(x): # x is assumed to be a sequence.

    (T, s0)=shape(x)
    s1 = int(sqrt(s0))
    assert(s1**2 == s0)
    
    Z=aux2(concatenate([aux(f) for f in x.reshape(T,s1,s1)],1))
    return Z

def stochastic(x): return array(rand(*shape(x))<x,'d')
def Rsigmoid(x): return stochastic(sigmoid(x))
def sigmoid(x):  return 1./(1. + exp(-x))
def Dsigmoid(x):
    s = sigmoid(x)
    return s*(1-s)

def monomial(n, shape=(1,)): return array(floor(rand(*shape)*n),'i')
multinomial = monomial
def id(x): return x



def true_multinomial(p):
    return (p.cumsum(1)<rand(len(p),1)).sum(1)



from pylab import amap
def softmax_slow(a):
    def softmax_1(a):
        e=exp(a-a.max())
        return e/e.sum()
    return amap(softmax_1,a)
def softmax(a):
    a=a-a.max(1).reshape(-1,1)
    b=exp(a)
    c=b.sum(1)
    if isnan(a).any():
        import pdb
        pdb.set_trace()
    ans = b/c.reshape(-1,1)
    if isnan(ans).any():
        import pdb
        pdb.set_trace()
    return ans
    


def generic_softmax_deriv(dE_dY, y):
    """        shape(dE_dY)=[n,k]
     input: shape(x)=[n,k]
            shape(y)=[n,k]
     note: the total input, x, isn't used.
    """

    # for a given n, the output is diag(y)-y*y'
    Y0 = dE_dY * y 
    return Y0 - y*Y0.sum(1).reshape(-1,1)
    
### these guys are so standard... they deserve better! 
def to_list_lens(vec, lens):
    assert(len(vec)==sum(lens))
    ans = [None]*len(lens)
    a = 0
    for i in range(len(lens)):
        b = a + lens[i]
        ans[i]=vec[a:b]
        a = b
    return ans
def from_list_lens(v_l, l):
    # we need to concatentae vectors.
    assert([len(z) for z in v_l]==l)
    return concatenate(v_l)
###

def expand(labels, n_labels):
    a=zeros((len(labels), n_labels))
    for i in range(len(labels)):
        a[i,labels[i]]=1
    return a

def e_true_multinomial(p):
    return expand(true_multinomial(p), shape(p)[1])

def ieye(n,i):
    z= zeros(n)
    z[i]=1
    return z
