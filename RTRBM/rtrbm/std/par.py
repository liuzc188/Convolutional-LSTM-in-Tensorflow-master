


from math import ceil

from numpy import mod, inf, double, isnan, fromstring
from pylab import *
import os, sys, select, operator

try:
    import cPickle
    pickle = cPickle
except ImportError:
    import pickle


def waitall(n):
    for i in xrange(n):
        (pid,status) = os.wait()
        print "pid %d removed" % pid



def pipeload(fd):
    """Unpickle an object from the stream given by a numeric descriptor."""
#    return pickle.load(os.fdopen(fd,'r'))
#    return fromfile(os.fdopen(fd,'r'))
    return fromstring(os.fdopen(fd,'r').read())
    
def pipedump(object, fd):
    """Pickle an object to the stream given by a numeric descriptor."""
#    pickle.dump(object,os.fdopen(fd,'w'),pickle.HIGHEST_PROTOCOL)
#    object.tofile(os.fdopen(fd,'w'))
    os.fdopen(fd,'w').write(object.tostring())

def chunkify(data,numchunks):
    chunksize = int(ceil(data.shape[1]/float(numchunks)))
    result = [None]*numchunks
    for i in range(numchunks):
      result[i] = data[:,i*chunksize:(i+1)*chunksize]
    return result


#### the first functions: just applies f to small subsets of the data,
#### returns the sum of the gradient. Useful if we want to use CG.
def pipe_par(f, data, numprocesses):
    chunksize = int(ceil(len(data)/float(numprocesses)))

    isparent = True
    readpipes = []
    children = []
    fdchild = {}
    nn = 0

    ### giving "birth" to the children :)
    while nn < numprocesses-1:
        parentread, childwrite = os.pipe()
        pid = os.fork()
        # child
        if not pid:
            # child has the variable nn, which tells it its number.
            isparent = False
            os.close(parentread)

            sys.stdin.close()
            os.close(0)


            break
        
        # parent
        else:
            os.close(childwrite)
            readpipes.append(parentread)
            children.append(pid)
            fdchild[parentread] = nn
            nn += 1

    # work on piece nn of the data; both parents and children.
    start = nn*chunksize
    stop = min(start + chunksize, len(data))

    # this step is done by each of the children and the parent.
    gradient = f(data[start:stop])
        
    # if child return your grad and exit
    if not isparent:
        pipedump(gradient.flatten(), childwrite)
        # the children have killed themselves, and are now zombies.
        os._exit(0)
        
    # if parent:
    # (by that point all the children are dead, so)
    assert(isparent)
    # 1) collect the output
    # (grad is its "own" gradient)
    grad = ([None]*(numprocesses-1)) + [gradient.flatten()]
    while readpipes:
        rfds, wfds, efds = select.select(readpipes, [], [], None)
        for fd in rfds:
            ii = fdchild[fd]
            grad[ii] = pipeload(fd)
            readpipes.remove(fd)
    ## i.e.: grad now has whatever the children sent. 

    totalgrad = grad[0]
    for g in grad[1:]:
        totalgrad += g
                
    # 2) wait (i.e., bury) for all the children
    # how to kill the children? Hmmm...
    for nn in range(numprocesses-1):
        try:
            (pid,status) = os.wait()
            if pid not in children:
                print "unrecognized child %d has died" % pid
            if not status:
                pass
            else:
                print "pid %d failed somehow" % pid
                children.remove(pid)
        except OSError, (errno, strerror):
            print "caught OSError", errno, ":", strerror

    return gradient.unpack(totalgrad)

def pipe_par_labs_reducer(f, reducer, data, labs, numprocesses):
    chunksize = int(ceil(len(data)/float(numprocesses)))

    isparent = True
    readpipes = []
    children = []
    fdchild = {}
    nn = 0

    ### giving "birth" to the children :)
    while nn < numprocesses-1:
        parentread, childwrite = os.pipe()
        pid = os.fork()
        # child
        if not pid:
            # child has the variable nn, which tells it its number.
            isparent = False
            os.close(parentread)

            sys.stdin.close()
            os.close(0)


            break
        
        # parent
        else:
            os.close(childwrite)
            readpipes.append(parentread)
            children.append(pid)
            fdchild[parentread] = nn
            nn += 1

    # work on piece nn of the data; both parents and children.
    start = nn*chunksize
    stop = min(start + chunksize, len(data))

    # this step is done by each of the children and the parent.
    gradient = f(data[start:stop], labs[start:stop])
        
    # if child return your grad and exit
    if not isparent:
        pipedump(gradient.flatten(), childwrite)
        # the children have killed themselves, and are now zombies.
        os._exit(0)
        
    # if parent:
    # (by that point all the children are dead, so)
    assert(isparent)
    # 1) collect the output
    # (grad is its "own" gradient)
    grad = ([None]*(numprocesses-1)) + [gradient.flatten()]
    while readpipes:
        rfds, wfds, efds = select.select(readpipes, [], [], None)
        for fd in rfds:
            ii = fdchild[fd]
            grad[ii] = pipeload(fd)
            readpipes.remove(fd)
    ## i.e.: grad now has whatever the children sent. 

    totalgrad = reducer(grad)
                
    # 2) wait (i.e., bury) for all the children
    # how to kill the children? Hmmm...
    for nn in range(numprocesses-1):
        try:
            (pid,status) = os.wait()
            if pid not in children:
                print "unrecognized child %d has died" % pid
            if not status:
                pass
            else:
                print "pid %d failed somehow" % pid
                children.remove(pid)
        except OSError, (errno, strerror):
            print "caught OSError", errno, ":", strerror

    return totalgrad



def pipe_par_reducer(f, reducer, data, numprocesses):
    chunksize = int(ceil(len(data)/float(numprocesses)))

    isparent = True
    readpipes = []
    children = []
    fdchild = {}
    nn = 0

    ### giving "birth" to the children :)
    while nn < numprocesses-1:
        parentread, childwrite = os.pipe()
        pid = os.fork()
        # child
        if not pid:
            # child has the variable nn, which tells it its number.
            isparent = False
            os.close(parentread)

            sys.stdin.close()
            os.close(0)


            break
        
        # parent
        else:
            os.close(childwrite)
            readpipes.append(parentread)
            children.append(pid)
            fdchild[parentread] = nn
            nn += 1

    # work on piece nn of the data; both parents and children.
    start = nn*chunksize
    stop = min(start + chunksize, len(data))

    # this step is done by each of the children and the parent.
    gradient = f(data[start:stop])
        
    # if child return your grad and exit
    if not isparent:
        pipedump(gradient.flatten(), childwrite)
        # the children have killed themselves, and are now zombies.
        os._exit(0)
        
    # if parent:
    # (by that point all the children are dead, so)
    assert(isparent)
    # 1) collect the output
    # (grad is its "own" gradient)
    grad = ([None]*(numprocesses-1)) + [gradient] #.flatten()]
    while readpipes:
        rfds, wfds, efds = select.select(readpipes, [], [], None)
        for fd in rfds:
            ii = fdchild[fd]
            grad[ii] = gradient.unpack(pipeload(fd))
            readpipes.remove(fd)
    ## i.e.: grad now has whatever the children sent. 

    totalgrad = reducer(grad)
                
    # 2) wait (i.e., bury) for all the children
    # how to kill the children? Hmmm...
    for nn in range(numprocesses-1):
        try:
            (pid,status) = os.wait()
            if pid not in children:
                print "unrecognized child %d has died" % pid
            if not status:
                pass
            else:
                print "pid %d failed somehow" % pid
                children.remove(pid)
        except OSError, (errno, strerror):
            print "caught OSError", errno, ":", strerror

    return totalgrad





def pipe_par_labs2(f, data, data2, labs, numprocesses):
    "f(data1, data2, labs) returns a gradient (that may contain a loss function in it)"
    chunksize = int(ceil(len(data)/float(numprocesses)))

    isparent = True
    readpipes = []
    children = []
    fdchild = {}
    nn = 0

    ### giving "birth" to the children :)
    while nn < numprocesses-1:
        parentread, childwrite = os.pipe()
        pid = os.fork()
        # child
        if not pid:
            # child has the variable nn, which tells it its number.
            isparent = False
            os.close(parentread)

            sys.stdin.close()
            os.close(0)


            break
        
        # parent
        else:
            os.close(childwrite)
            readpipes.append(parentread)
            children.append(pid)
            fdchild[parentread] = nn
            nn += 1

    # work on piece nn of the data; both parents and children.
    start = nn*chunksize
    stop = min(start + chunksize, len(data))

    # this step is done by each of the children and the parent.
    gradient = f(data[start:stop],data2[start:stop], labs[start:stop])
        
    # if child return your grad and exit
    if not isparent:
        pipedump(gradient.flatten(), childwrite)
        # the children have killed themselves, and are now zombies.
        os._exit(0)
        
    # if parent:
    # (by that point all the children are dead, so)
    assert(isparent)
    # 1) collect the output
    # (grad is its "own" gradient)
    grad = ([None]*(numprocesses-1)) + [gradient.flatten()]
    while readpipes:
        rfds, wfds, efds = select.select(readpipes, [], [], None)
        for fd in rfds:
            ii = fdchild[fd]
            grad[ii] = pipeload(fd)
            readpipes.remove(fd)
    ## i.e.: grad now has whatever the children sent. 

    totalgrad = grad[0]
    for g in grad[1:]:
        totalgrad += g
                
    # 2) wait (i.e., bury) for all the children
    # how to kill the children? Hmmm...
    for nn in range(numprocesses-1):
        try:
            (pid,status) = os.wait()
            if pid not in children:
                print "unrecognized child %d has died" % pid
            if not status:
                pass
            else:
                print "pid %d failed somehow" % pid
                children.remove(pid)
        except OSError, (errno, strerror):
            print "caught OSError", errno, ":", strerror

    return gradient.unpack(totalgrad)

def pipe_par_labs(f, data,  labs, numprocesses):
    "f(data,  labs) returns a gradient (that may contain a loss function in it)"
    chunksize = int(ceil(len(data)/float(numprocesses)))

    isparent = True
    readpipes = []
    children = []
    fdchild = {}
    nn = 0

    ### giving "birth" to the children :)
    while nn < numprocesses-1:
        parentread, childwrite = os.pipe()
        pid = os.fork()
        # child
        if not pid:
            # child has the variable nn, which tells it its number.
            isparent = False
            os.close(parentread)

            sys.stdin.close()
            os.close(0)


            break
        
        # parent
        else:
            os.close(childwrite)
            readpipes.append(parentread)
            children.append(pid)
            fdchild[parentread] = nn
            nn += 1

    # work on piece nn of the data; both parents and children.
    start = nn*chunksize
    stop = min(start + chunksize, len(data))

    # this step is done by each of the children and the parent.
    gradient = f(data[start:stop], labs[start:stop])
        
    # if child return your grad and exit
    if not isparent:
        pipedump(gradient.flatten(), childwrite)
        # the children have killed themselves, and are now zombies.
        os._exit(0)
        
    # if parent:
    # (by that point all the children are dead, so)
    assert(isparent)
    # 1) collect the output
    # (grad is its "own" gradient)
    grad = ([None]*(numprocesses-1)) + [gradient.flatten()]
    while readpipes:
        rfds, wfds, efds = select.select(readpipes, [], [], None)
        for fd in rfds:
            ii = fdchild[fd]
            grad[ii] = pipeload(fd)
            readpipes.remove(fd)
    ## i.e.: grad now has whatever the children sent. 

    totalgrad = grad[0]
    for g in grad[1:]:
        totalgrad += g
                
    # 2) wait (i.e., bury) for all the children
    # how to kill the children? Hmmm...
    for nn in range(numprocesses-1):
        try:
            (pid,status) = os.wait()
            if pid not in children:
                print "unrecognized child %d has died" % pid
            if not status:
                pass
            else:
                print "pid %d failed somehow" % pid
                children.remove(pid)
        except OSError, (errno, strerror):
            print "caught OSError", errno, ":", strerror

    return gradient.unpack(totalgrad)



def pipe_par_learn(f, w, data_getter, LR=.01, maxepoch=40, momentum=.9, numprocesses=8):
    """
    do some learning starting with w. f(w,data_getter()) returns a gradient
    that should include weight decay and normalizing for batch_size.
    Default values: maxepoch=40, momentum=.9, numprocesses=8, LR=.01.
    """

    isparent = True
    readpipes = []
    children = []
    fdchild = {}
    nn = 0

    ### giving "birth" to the children :)
    print 'forking...'
    while nn < numprocesses-1:
        parentread, childwrite = os.pipe()
        pid = os.fork()
        random.seed(int(os.times()[-1]*1000))
        # reseed, please!
        # child
        if not pid:
            # child has the variable nn, which tells it its number.
            isparent = False
            os.close(parentread)

            sys.stdin.close()
            os.close(0)


            break
        
        # parent
        else:
            os.close(childwrite)
            readpipes.append(parentread)
            children.append(pid)
            fdchild[parentread] = nn
            nn += 1
    print 'done forking.'

    # work on piece nn of the data; both parents and children.



    w = w.copy() #so that each has its own! right?! ;(

    # these steps is done by each of the children and the parent.
    v = 0 * w
    for epoch in xrange(maxepoch):
        print 'proc:%d, epoch:%d' % (nn, epoch)
        dat = data_getter()
        v *= momentum
        v += LR * f(w, dat)
        w += v


    #### the end of the joy. from this
    # if child return your grad and exit
    if not isparent:
        pipedump(w.flatten(), childwrite)
        # the children have killed themselves, and are now zombies.
        os._exit(0)
        
    # if parent:
    # (by that point all the children are dead, so)
    assert(isparent)
    # 1) collect the output
    # (grad is its "own" gradient)
    all_of_w = ([None]*(numprocesses-1)) + [w.flatten()]
    while readpipes:
        rfds, wfds, efds = select.select(readpipes, [], [], None)
        for fd in rfds:
            ii = fdchild[fd]
            all_of_w[ii] = pipeload(fd)
            readpipes.remove(fd)
    

    ### collect results from parents: add the parameters and average the results.
    new_w = all_of_w[0]
    for u in all_of_w[1:]:
        new_w += u
    new_w *= (1. / len(all_of_w)) 
    # an average it is.
              


    # 2) wait (i.e., bury) for all the children
    # how to kill the children? Hmmm...
    for nn in range(numprocesses-1):
        try:
            (pid,status) = os.wait()
            if pid not in children:
                print "unrecognized child %d has died" % pid
            if not status:
                pass
            else:
                print "pid %d failed somehow" % pid
                children.remove(pid)
        except OSError, (errno, strerror):
            print "caught OSError", errno, ":", strerror

    # and unpack the result to its usual shape.
    return w.unpack(new_w)





### its still not ready!
class par:
    def __init__(self, f, numprocesses=8):

        self.isparent = True
        isparent = True
        self.readpipes = []
        self.writepipes = []
        self.children = []
        self.readpipe_to_child = {}
        self.writepipe_to_child = {}
        self.nn = 0

        self.numprocesses = numprocesses

        # fork all the children.
        print 'forking...'
        while self.nn < self.numprocesses: #the parnet wont actually do anything, actually.
            parentread, childwrite  = os.pipe()
            childread,  parentwrite = os.pipe() ## make two pipes--one forward, one backward.
            pid = os.fork()
            random.seed(int(os.times()[-1]*1000))
            # reseed, please!
            # child
            if not pid:
                # child has the variable nn, which tells it its number.
                isparent = False
                os.close(parentread)
                os.close(parentwrite)
                sys.stdin.close()
                os.close(0)
                self.isparent = False
                self.childread  = childread
                self.childwrite = childwrite
                break
        
            # parent
            else:
                os.close(childread)
                os.close(childwrite)
                os.setpgid(pid, 0)
                self.readpipes.append(parentread)
                self.writepipes.append(parentwrite)
                self.children.append(pid)
                self.readpipe_to_child[parentread] = self.nn
                self.writepipe_to_child[parentwrite] = self.nn
                self.nn += 1
        print '%d says hi' % self.nn
        print 'done forking.'

        #### interestingly, the code for the children goes here! Yup. 
        #### 
        #### ok. all children wait
        if not self.isparent:
            ### start waiting for data.

            readpipes = [self.childread]
            while readpipes:
                rfds, wfds, efds = select.select(readpipes, [], [], None)
                for fd in rfds:
                    ii = fdchild[fd]
                    grad[ii] = pipeload(fd)
                    readpipes.remove(fd)            

        ## good. At this point, there should be many processes.
    def killchildren(self):
        ### it still does not work. 
        for child in self.children:
            os.kill(-child, 9)
            os.kill(child, 9)
