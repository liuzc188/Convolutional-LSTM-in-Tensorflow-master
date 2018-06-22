#import pdb
from pylab import *
from PIL import Image

def imread(file):
    #pdb.set_trace()
    f = Image.open(file)
    diter =f.getdata()
    d    = array(diter)
    size = f.size
    return d.reshape(size[1],size[0],3)


