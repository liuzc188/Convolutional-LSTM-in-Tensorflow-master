import cPickle
from gzip import GzipFile as gfile


    
def gsave(p, filename):
    f=gfile(filename,'wb')
    cPickle.dump(p,f,cPickle.HIGHEST_PROTOCOL) 
    f.close()

def gload(filename):
    f=gfile(filename,'rb')
    y=cPickle.load(f)
    f.close()
    return y

def save(p, filename):
    f=file(filename,'wb')
    cPickle.dump(p,f,cPickle.HIGHEST_PROTOCOL) 
    f.close()

def load(filename): #a unified loading interface :) 
    try: 
        return gload(filename)
    except: None
    
    f=file(filename,'rb')
    y=cPickle.load(f)
    f.close()
    return y

def fload(filename):
    
    f=file(filename,'rb')
    y=cPickle.load(f)
    f.close()
    return y
