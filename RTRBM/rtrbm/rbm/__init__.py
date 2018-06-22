from pylab import newaxis, amap, binary_repr, zeros, log, exp, log_sum_exp, sigmoid, stochastic, Rsigmoid, rand, randn
from pylab import dot
def int_to_bin(i, v):
    assert(0 <= i < 2**v)
    return (amap(int,list(binary_repr(i).zfill(v)))[newaxis,:]).astype('f')

def log1exp(b):
    return log(1+exp(-abs(b))) + b * (b>0).astype('f');

def free_energy(w, x):
    return log1exp(w*x).sum(1) + (w[1][newaxis,:] * x).sum(1)

def brute_force_Z(W):
    """
    For binary visibles and hiddens.
    
    """
    v, h = W.v, W.h
    if v>h:
        W1 = 0 * W
        W1.w = [W1.w[0].T(), W1.w[2], W1.w[1]]

        return brute_force_Z(W1)

    if v>20:
        print "v = ", v, " may take a long time. "

    Z = zeros(2**v)
    for i in xrange(2**v):
        V    = int_to_bin(i, v)
        Z[i] = free_energy(W, V)

    z = log_sum_exp(Z)
    return z
    
def brute_force_Z_vis_gauss(W):
    v,h = W.v, W.h
    Z = zeros(2**h)
    for i in xrange(2**h):
        H = int_to_bin(i, h)
        b = W.T() * H
        Z[i] = .5 * dot(b,b.T) + dot(H,W[2])
    return log_sum_exp(Z)
def rbm_grad_exact(W, x, vis_gauss=False):

    batch_size = float(len(x))
    v, h = W.v, W.h
    G = 0 * W
    H = sigmoid(W*x)
    G += 1./batch_size * W.outp(x, H) 
    if not vis_gauss:
        Z = brute_force_Z(W)

        def prob(V):
            return exp( free_energy(W, V)[0] - Z )

        for i in xrange(2**v):
            V = int_to_bin(i, v)
            H = sigmoid(W* V)
            G -= prob(V) * W.outp(V, H) 

            loss = - ( free_energy(W, x).mean(0) - Z )

    if vis_gauss:
        Z = brute_force_Z_vis_gauss(W)

        def prob(H):
            b = W.T() * H
            z = .5*dot(b,b.T) + dot(H,W[2])
            return float(exp(z-Z) )

        for i in xrange(2**h):
            H = int_to_bin(i, h)
            b = W.T()*H
            G -= prob(H) * W.outp(b,H)
        loss = - (-.5*amap(dot,x,x).mean() + free_energy(W, x).mean(0) - Z)

    return batch_size * G, dict(loss=batch_size * loss)            

def rbm_grad_cd(W, x, cd, vis_gauss=False):
    batch_size = float(len(x))
    v, h, = W.v, W.h
    
    V = x
    H = sigmoid(W * x)
    G = W.outp(V, H) 
    for g in range(cd):
        H = stochastic(H)
        if vis_gauss:
            V = W.T()*H + randn(batch_size,v)
        else:
            V = Rsigmoid(W.T() * H)
        H = sigmoid(W * V)
    G -= W.outp(V, H)

    loss = abs(V - x).sum()
    return G, dict(loss=loss)
        
def sample(W,g,batch_size, vis_gauss=False):
    v,h = W.v, W.h
    V = rand(batch_size,v)
    for gg in range(g):
        H = Rsigmoid(W*V)

        if vis_gauss:
            V = W.T()*H + randn(batch_size,v)
        else:
            V = Rsigmoid(W.T()*H)
    return V,H


def sample_last_mf(W,g,batch_size, vis_gauss=False):
    v,h = W.v, W.h
    V = rand(batch_size,v)
    for gg in range(g):
        H = Rsigmoid(W*V)

        if vis_gauss:
            V = W.T()*H + randn(batch_size,v)
        else:
            V = Rsigmoid(W.T()*H)

    if vis_gauss:	
        V = W.T()*H 
    else:
        V = sigmoid(W.T()*H)

    return V,H
