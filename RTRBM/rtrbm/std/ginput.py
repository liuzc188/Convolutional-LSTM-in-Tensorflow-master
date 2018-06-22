


import wx as _wx
import pylab as _pylab
import time as _time


class GaelInput(object):
    """
    Class that creates a callable object to retrieve mouse click in a
    blocking way, a la MatLab. Based on Gael Varoquaux's almost-working
    object. Thanks Gael! I've been trying to get this working for years!

    -Jack
    """

    debug = False
    cid   = None # event connection

    def on_click(self, event):
        """
        Event handler that will be passed to the current figure to
        retrive clicks.
        """
        # if it's a valid click, append the coordinates to the list
        if event.inaxes:
            self.clicks.append((event.xdata, event.ydata))
            if self.debug: print "boom: "+str(event.xdata)+","+str(event.ydata)

    def __call__(self, n=1, timeout=30, debug=False):
        """
        Blocking call to retrieve n coordinate pairs through mouse clicks.
        """

        # just for printing the coordinates
        self.debug = debug

        # make sure the user isn't messing with us
        assert isinstance(n, int), "Requires an integer argument"

        # connect the click events to the on_click function call
        self.cid = _pylab.connect('button_press_event', self.on_click)

        # initialize the list of click coordinates
        self.clicks = []

        # wait for n clicks
        counter = 0
        while len(self.clicks) < n:
            # key step: yield the processor to other threads
            _wx.Yield();

            # rest for a moment
            _time.sleep(0.1)

            # check for a timeout
            counter += 1
            if counter > timeout/0.1: print "ginput timeout"; break;

        # All done! Disconnect the event and return what we have
        _pylab.disconnect(self.cid)
        self.cid = None
        return self.clicks

def ginput(n=1, timeout=30, debug=False):
    """
    Simple functional call for physicists. This will wait for n clicks
from the user and
    return a list of the coordinates of each click.
    """

    x = GaelInput()
    return x(n, timeout, debug) 
