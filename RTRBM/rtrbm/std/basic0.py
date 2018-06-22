from pylab import *

### slow and crappy version.
def ind2sub(inds, dims):
    assert(len(dims)==2)
    ans = zeros((len(dims),  len(inds)),'i')
    inds = array(inds,'i')


    dims=map(int,dims)
    for i in range(len(inds)):
        ind = inds[i]
        ## himiya
        ans[0, i] = ind / dims[1]
        ans[1, i] = ind % dims[1] 
    return ans


def min_N(A, N):
    B=[(x,i) for i,x in enumerate(A)]
    B.sort()

    min_N = B[:N]

    inds = array([i for (x,i) in min_N])
    vals = array([x for (x,i) in min_N])

    return inds, vals

def max_N(A, N):
    inds, vals = min_N(-A, N)
    return inds, -vals
