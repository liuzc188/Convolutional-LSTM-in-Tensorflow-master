from pylab import *
from mats.eq_conv_mat import eq_conv_mat
from mats.conv_mat import conv_mat
def frac_shift_2(img, v):
    assert(((v>=0) & (v<1)).all())

    
    c0 = array([0, 1-v[0], v[0]])
    c1 = array([0, 1-v[1], v[1]])
    
    W = c0.reshape(1,-1)*c1.reshape(-1,1)

    sh_img = shape(img)

    c = eq_conv_mat([sh_img[0], sh_img[1]],
                    [3,3],
                    [1,1],
                    [1,1])
    c.w[0]=W

    ans=(c*img.reshape(1,-1)).reshape(shape(img))
                    
    return ans

def gen_shift_2(img, v):
    v_int  = floor(v)
    v_frac = v - floor(v)

    a,b = shape(img)
    i,j = arange(a), arange(b)
    [J,I]=meshgrid(j,i)

    Ish = array(I + v_int[1],'i')
    Jsh = array(J + v_int[0],'i')

    Ish[Ish>=a]=a-1
    Jsh[Jsh>=b]=b-1
    Ish[Ish<0]=0
    Jsh[Jsh<0]=0

    img2 = img[Ish, Jsh]

    img3 = frac_shift_2(img2, v_frac)

    return img3
    
