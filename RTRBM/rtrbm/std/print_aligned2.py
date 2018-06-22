from pylab import sqrt, ceil, zeros
def print_aligned2(w, r1, r2, n1=None, n2=None):
    if n1==None or n2==None:
        n1 = int(ceil(sqrt(w.shape[1])))
        n2 = n1
        

    assert(r1*r2==w.shape[0])

    Z = zeros(((r1+1)*n1, (r2+1)*n2), 'd')
    i1, i2 = 0, 0
    for i1 in range(n1):
        for i2 in range(n2):
            i = i1*n2+i2
            if i>=w.shape[1]: break
            Z[(r1+1)*i1:(r1+1)*(i1+1)-1, (r2+1)*i2:(r2+1)*(i2+1)-1] = w[:,i].reshape(r1,r2)
    return Z
