import numpy

def l_p(x,p):
    y = x.copy()
    y *= 2*(y>0).bool2float()-1
    return (y**p).sum()


def dl_p(x,p):
    return p*(abs(x)**(p-1))
    
