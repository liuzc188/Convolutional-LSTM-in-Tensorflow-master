
"""
The common_functions implement functions that are useful for most trainers;
It contains different functions that return the gradient and the cost functions of various
"""


from std import *
from pylab import *


def const(c): return lambda y: c
def dict_to_f(d):
    try:
        d[0]  #this has to be tried, clearly.
    except:
        raise Exception ("dict_to_find requires that d has the key 0")
    k = array(d.keys(),'d')
    def res(n):
        return d[k[find(k <= n).max()]]
    return res
def make_parameter(x):
    if isscalar(x):
        return const(x)
    elif type(x)==dict:
        return dict_to_f(x)
    else:
        return x

def rms(x):
    return sqrt(  (x**2).sum() / float(size(x))   )



### these are the important functions for me right now. Yeah.
def make_std_bp_softmax(W):
    """
    W is an ff_net of matrices, which are propagated through the sigmoids.
    The final layer uses the softmax, however.
    """
    st = [None]*(len(W)+1)

    def grad(W, d):
        X, Y = d
        batch_size = len(X)
        st[0] = X
        for i in range(len(W)-1):
            st[i+1]=empty((batch_size, W[i].h),'d')

        for i in range(len(W)-1):
            fast_sigmoid(W[i]*st[i], st[i+1])

        st[-1] = softmax(W[-1]*st[-2])
        P      = st[-1]
        L      = -(log(P)*Y).sum()
        L01    = (P.argmax(1)!=Y.argmax(1)).sum()    

        dW     = 0 * W

        dx     = Y - P
        for i in reversed(range(len(W))):
            dW[i] = W[i].outp(st[i], dx)
            dy    = W[i].T() * dx
            dx    = dy * st[i] * (1-st[i])
        #import pdb
        #pdb.set_trace()

        return dW, dict(loss=L, zero_one_loss=L01)

    def loss(W, d):
        X, Y = d
        batch_size = len(X)
        st[0] = X
        for i in range(len(W-1)):
            st[i+1]=empty((batch_size, W[i].h),'d')

        for i in range(len(W)-1):
            fast_sigmoid(W[i]*st[i], st[i+1])

        P = softmax(W[-1]*st[-2])
        L      = -(log(P)*Y).sum()
        L01    = (P.argmax(1)!=Y.argmax(1)).sum()    
        
        return dict(loss=L, zero_one_loss=L01)
        
    return grad, loss



### these are the important functions for me right now. Yeah.
def make_std_bp_squared(W):
    """
    W is an ff_net of matrices, which are propagated through the sigmoids.
    The final layer uses the squared loss, yay!.
    """
    st = [None]*(len(W)+1)

    def grad(W, d):
        X, Y = d
        batch_size = len(X)
        st[0] = X
        for i in range(len(W)-1):
            st[i+1]=empty((batch_size, W[i].h),'d')

        for i in range(len(W)-1):
            fast_sigmoid(W[i]*st[i], st[i+1])

        st[-1] = (W[-1]*st[-2]) # linear regression.
        P      = st[-1]
        L      = .5 * ((P - Y)**2).sum()

        #L01    = (P.argmax(1)!=Y.argmax(1)).sum()    

        dW     = 0 * W

        dx     = Y - P
        for i in reversed(range(len(W))):
            dW[i] = W[i].outp(st[i], dx)
            dy    = W[i].T() * dx
            dx    = dy * st[i] * (1-st[i])

        return dW, dict(loss=L) #, zero_one_loss=L01)

    def loss(W, d):
        X, Y = d
        batch_size = len(X)
        st[0] = X
        for i in range(len(W-1)):
            st[i+1]=empty((batch_size, W[i].h),'d')

        for i in range(len(W)-1):
            fast_sigmoid(W[i]*st[i], st[i+1])

        P = (W[-1]*st[-2])
        L      = .5 * ((P - Y)**2).sum()
        #L01    = (P.argmax(1)!=Y.argmax(1)).sum()    
        
        return dict(loss=L) #, zero_one_loss=L01)
        
    return grad, loss




def make_std_bp_sigmoid(W):
    """
    W is an ff_net of matrices, which are propagated through the sigmoids.
    The final layer uses the softmax, however.
    """
    st = [None]*(len(W)+1)

    def grad(W, d):
        X, Y = d
        batch_size = len(X)
        st[0] = X
        for i in range(len(W)-1):
            st[i+1]=empty((batch_size, W[i].h),'d')

        for i in range(len(W)-1):
            fast_sigmoid(W[i]*st[i], st[i+1])

        st[-1] = sigmoid(W[-1]*st[-2])
        P      = st[-1]
        L      = -(log(P)*Y+log(1-P)*(1-Y)).sum()
        L01    = (array(P>.5,'i')!=array(Y,'i')).sum()    

        dW     = 0 * W

        dx     = Y - P
        for i in reversed(range(len(W))):
            dW[i] = W[i].outp(st[i], dx)
            dy    = W[i].T() * dx
            dx    = dy * st[i] * (1-st[i])

        return dW, dict(loss=L, zero_one_loss=L01)

    def loss(W, d):
        X, Y = d
        batch_size = len(X)
        st[0] = X
        for i in range(len(W)-1):
            st[i+1]=empty((batch_size, W[i].h),'d')

        for i in range(len(W)-1):
            fast_sigmoid(W[i]*st[i], st[i+1])

        st[-1] = sigmoid(W[-1]*st[-2])
        P      = st[-1]
        L      = -(log(P)*Y+log(1-P)*(1-Y)).sum()
        L01    = (array(P>.5,'i')!=array(Y,'i')).sum()    

        return dict(loss=L, zero_one_loss=L01)
    return grad, loss




class make_greedy:
    """
    The function takes W[0], W[1],..,W[k-2], W[k-1],W[k]
    uses W[0:k-1] to do inference, and returns the gradient with 
    respect to W[k].
    """
    def __init__(self, W, cd=1):
        self.W = W
        self.cd = cd

        self.st = [None]*(len(self.W)+1)

    def grad(self, W_k, d, CD=None):
        if CD==None: CD=self.cd
        ### this time, d is simply the raw data

        batch_size = len(d)
        self.st[0] =  1 * d
        for i in range(len(self.W)):
            self.st[i+1]=empty((batch_size, self.W[i].h),'d')

        for i in range(len(self.W)):
            fast_sigmoid(self.W[i]*self.st[i], self.st[i+1]) # let us hope that this will work!
        
        # dW_k = 0 * W_k
        X = self.st[-1]

        ### now, do CD
        H = empty((batch_size, W_k.h), 'd')
        V = empty((batch_size, W_k.v), 'd')
        fast_sigmoid(W_k * X, H)
        # H is now known
        
        dW_k = W_k.outp(X, H)

        H = stochastic(H)
        for k in range(CD-1):
            fast_Rsigmoid(W_k.T() * H, V)
            fast_Rsigmoid(W_k * V, H)
        fast_Rsigmoid(W_k.T() * H, V)
        fast_sigmoid(W_k * V, H)            

        dW_k -= W_k.outp(V, H)        

        recon_error = abs(X-V).sum()
        return dW_k, dict(loss=recon_error)

    def recon(self, W_k, d, CD=None):
        if CD==None: CD=self.cd
        batch_size = len(d)
        self.st[0] =  1 * d
        for i in range(len(self.W)):
            self.st[i+1]=empty((batch_size, self.W[i].h),'d')

        for i in range(len(self.W)):
            fast_sigmoid(self.W[i]*self.st[i], self.st[i+1]) # let us hope that this will work!
        
        X = self.st[-1]

        ### now, do CD
        H = empty((batch_size, W_k.h), 'd')
        V = empty((batch_size, W_k.v), 'd')
        fast_sigmoid(W_k * X, H)
        # H is now known
        
        H = stochastic(H)
        for k in range(CD-1):
            fast_Rsigmoid(W_k.T() * H, V)
            fast_Rsigmoid(W_k * V, H)
        fast_Rsigmoid(W_k.T() * H, V)
        fast_sigmoid(W_k * V, H)            


        recon_error = abs(X-V).sum()
        return dict(loss=recon_error)

    
    
    def make_greedy(self):
        def grad(w,d):
            return self.grad(w,d)
        def loss(w,d):
            return self.recon(w,d)

        return grad, loss

