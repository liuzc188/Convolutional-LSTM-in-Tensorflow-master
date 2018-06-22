from basic import *
def rev(x): return list(reversed(x))

def random_subset(n,k):
    s=set([])
    while len(s)<k:
        s.add(monomial(n)[0])
        
    return list(s)


#from scipy.misc import imresize
def imresize(*args):
    raise Exception ("imresize is not implemented in the new python (?)")

def normalize_im(im, a, b):

    im1 = (im-float(a))/(float(b)-float(a))
    im1[im1<0]=0
    im1[im1>1]=1
    return im1

def im_show_resize(IMGS, AXIS=0, LOCAL=False, a=None, b=None, concat=True):
    from pylab import array
    if not LOCAL:
        if b==None:
            b = array([im.max() for im in IMGS],'f').max()
        if a==None:
            a = array([im.min() for im in IMGS],'f').min()
        #if a.__class__ != list:
        #    A = [a] * len(IMGS)
        #else:
        #    assert(len(a)==len(IMGS))
        #    A= a
        #if b.__class__ != list:
        #    B = [b] * len(IMGS)
        #else:
        #    assert(len(b)==len(IMGS))
        #    B = b

        IMGS1 = [normalize_im(im,a,b) for im in IMGS]
    else:
        assert(a==None and b==None)
        IMGS1 = [normalize_im(im,im.min(),im.max()) for im in IMGS]

    IMGS = IMGS1
    for im in IMGS:
        assert((0<=im).all() and (im<=1).all())

    if AXIS==1:
        IMGS = [im.T for im in IMGS]
    
    ANS = []

    MAX_height = max(len(im) for im in IMGS)
    for im in IMGS:
        height, width = im.shape
        
        new_height = MAX_height
        new_width  = int(float(width) / float(height) * MAX_height)+1
        new_size = new_height, new_width

        im1 = im.min() + (imresize(im, new_size).astype('f')/255)*(im.max()-im.min())
        ANS.append(im1)


    if concat:
        c = concatenate(ANS,1)

        if AXIS==0:
            return c
        else:
            return c.T
    else:
        return [x.T for x in ANS]
    
    
