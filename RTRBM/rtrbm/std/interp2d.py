""" Class for interpolating values
    - coded just like the octave algorithm for this problem.
    - this is nasty code !!!
    - it just does, what I needed

    2004-10-18 00:30;   Gerald Richter

    THOUGHTS:
    !! Need to find argument for keeping initialize.  If it isn't
    !! found, get rid of it!
    There is some arguments for keeping initialize.
    - it may be convenient, to keep the data in the object, forgetting
      about it in the instanciating routine, 
    - when using the call to the object, it behaves just like a function 
      that is smoothly defined.
    - an interpolation is usually required many times, so the
      instanciation and size-checking stuff is run only once.
"""

# TODO: - what happens, if I supply shapes of (x_new, y_new)
#           (1,1) or (2,1) or (1,2) in Grid mode ?
#       - add nearest method
#       - supply an option for clipping the values outside the range.
#           in fill_value = Clip
#           will return reduced grid of interpolated, plus the new grid

__all__ = ['interp2']

from scipy_base import *
from scipy_base.fastumath import *
from scipy import *
# need masked arrays if stuff gets too big.
#import MA

# The following are cluges to fix brain-deadness of take and
# sometrue when dealing with 0 dimensional arrays.
# Shouldn't they go to scipy_base??

_sometrue = sometrue
def sometrue(a,axis=0):    
    x = asarray(a)
    if shape(x) == (): x = x.flat
    return _sometrue(x)

def reduce_sometrue(a):
    all = a
    while len(shape(all)) > 1:    
        all = sometrue(all)
    return all

## indices does that too in some way
def meshgrid( a, b):
    a = asarray(a)
    b = asarray(b)
    return resize(a,(len(b),len(a))), \
        transpose(resize(b,(len(a),len(b))))


class interp2:
    
    # define the modes for interpolation
    #   'Grid'  gives results for a full combination of 
    #       all x-points with all y-points
    #   'Point' gives results only for the pairs of points
    op_modes = ('Grid','Point')
    #self.avail_methods = ('linear', 'nearest')
    avail_methods = ('linear')

    # initialization
    def __init__(self,x,y,z,kind='linear',
                 copy=1,bounds_error=0,fill_value=None):
        """Initialize a 2d linear interpolation class
        
        Input:
          x,y  - 1-d arrays: defining 2-d grid or 2-d meshgrid 
                             have to be ascending order
          z    - 2-d array of grid values
          kind - interpolation type ('nearest', 'linear', 'cubic', 'spline')
          copy - if true then data is copied into class, otherwise only a
                   reference is held.
          bounds_error - if true, then when out_of_bounds occurs, an error is
                          raised otherwise, the output is filled with
                          fill_value.
          fill_value - if None, then NaN, otherwise the value to fill in
                        outside defined region.
        """
        self.copy = copy
        self.bounds_error = bounds_error
        if fill_value is None:
            self.fill_value = array(0.0) / array(0.0)
        else:
            self.fill_value = fill_value

        if kind not in self.avail_methods:
            raise NotImplementedError, "Only "+ \
                str(self.avail_methods)+ \
                "supported for now."

        ## not sure if the rest of method is kosher
        # checking the input ranks
        # shape z:
        #   x: columns, y: rows
        if rank(z) != 2:
            raise ValueError, "z Grid values is not a 2-d array."
        (rz, cz) = shape(z)
        if min(shape(z)) < 3:
            raise ValueError, "2d fitting a Grid with one extension < 3"+\
                    "doesn't make too much of a sense, don't you think?"
        if (rank(x) > 1) or (rank(y) > 1):
            raise ValueError, "One of the input arrays is not 1-d."
        if (len(x) != rz) or (len(y) != cz):
            print "len of x: ", len(x)
            print "len of y: ", len(y)
            print "shape of z: ", shape(z)
            raise ValueError, "Length of X and Y must match the size of Z."

        # TODO: could check for x,y input as grids, and check dimensions

        # TODO: check the copy-flag
        #       offer some spae-saving alternatives        
        #self.x = atleast_1d(x).copy()
        self.x = atleast_1d(x).astype(x.typecode())
        self.x.savespace(1)
        #self.y = atleast_1d(y).copy()
        self.y = atleast_1d(y).astype(y.typecode())
        self.y.savespace(1)
        #self.z = array(z, copy=self.copy)
        self.z = array(z, z.typecode(), copy=self.copy, savespace=1)


    # the call
    def __call__(self, xi, yi, mode='Grid'):
        """
        Input:
          xi, yi      : 1-d arrays defining points to interpolate.
          mode        : 'Grid': (default)
                                calculate whole grid of x_new (x) y_new
                                points, returned as such
                        'Point' : take the [x_new, y_new] tuples and
                                return result for each
        Output:
          z : 2-d array (grid) of interpolated values; mode = 'Grid'
              1-d array of interpol. values on points; mode = 'Point'
        """
        
        if mode not in self.op_modes:
            raise NotImplementedError, "Only "+ \
                str(self.op_modes)+ \
                "operation modes are supported for now."

        # save some space
        # TODO: is this typing good?
        xi = atleast_1d(xi).astype(Float32)
        yi = atleast_1d(yi).astype(Float32)
        # TODO: check dimensions of xi, yi?
        #XI = MA.array(xi);
        #YI = MA.array(yi);
        XI = xi; YI = yi;
        X = self.x; Y = self.y;
        Z = self.z

        # TODO: move this to init ?
        xtable = X;
        ytable = Y;
        ytlen = len (ytable);
        xtlen = len (xtable);

        # REMARK: the octave routine sets the indices one higher if
        #       values are equal, not lower, as searchsorted() does.
        #       changed and verified behaviour, result only 
        #           differed at O(1e-16). 
        #   this is the more exact and octave identical approach
        eqx = sum(X == reshape(repeat(XI,(len(X))), (len(X), len(XI))))
        # get x index of values in XI
        xidx = clip( (searchsorted(X,XI) + eqx),1,len(X)-1 )-1
        eqy = sum(Y == reshape(repeat(YI,(len(Y))), (len(Y), len(YI))))
        # get y index of values in YI
        yidx = clip( (searchsorted(Y,YI) + eqy),1,len(Y)-1 )-1
        
        # get the out of bounds
        (out_of_xbounds, out_of_ybounds) = \
                                self._check_bounds(XI, YI)

        # generate an mgrid from the vectors
        #   transforming to full grid shapes
        ( X, Y) = meshgrid( X, Y)
        ( XI, YI) = meshgrid( XI, YI)
        """
        if mode == 'Point':
            XI = MA.masked_array( XI, 
                mask=eye(shape(XI)[0], shape(XI)[1]).astype('b') )
            YI = MA.masked_array( YI, 
                mask=eye(shape(YI)[0], shape(YI)[1]).astype('b') )
            X = MA.masked_array( X, 
                mask=eye(shape(X)[0], shape(X)[1]).astype('b') )
            Y = MA.masked_array( Y, 
                mask=eye(shape(Y)[0], shape(Y)[1]).astype('b') )
        print X.mask()
        print X.compressed()
        """

        # calculating the shifted squares
        a = (Z[:-1, :-1]);
        b = ((Z[:-1, 1:]) - a);
        c = ((Z[1:, :-1]) - a);
        d = ((Z[1:, 1:]) - a - b - c);

        # TODO: write an index take method
        it1 = take(take(X, xidx,axis=1), yidx, axis=0)
        Xsc = (XI - it1) / \
              ( take(take(X,(xidx+1),axis=1), yidx, axis=0) - it1 )
        Xsc = transpose(Xsc)
        it2 = take(take(Y, xidx,axis=1), yidx, axis=0)
        Ysc = (YI - it2) / \
              ( take(take(Y,xidx,axis=1), (yidx+1), axis=0) - it2 )
        Ysc = transpose(Ysc)
        #it1 = take(take(MA.filled(X,0), xidx,axis=1), yidx, axis=0)
        #Xsc = (MA.filled(XI,0) - it1) / \
        #      ( take(take(MA.filled(X,0),(xidx+1),axis=1), yidx, axis=0) - it1 )
        #Xsc = MA.transpose(Xsc)
        #it2 = take(take(MA.filled(Y,0), xidx,axis=1), yidx, axis=0)
        #Ysc = (MA.filled(YI,0) - it2) / \
        #      ( take(take(MA.filled(Y,0),xidx,axis=1), (yidx+1), axis=0) - it2 )
        #Ysc = MA.transpose(Ysc)

        # apply plane equation
        ZI = take( take(a,yidx,axis=1), xidx, axis=0) + \
                take(take(b,yidx,axis=1), xidx, axis=0) * Ysc + \
                take(take(c,yidx,axis=1), xidx, axis=0) * Xsc + \
                take(take(d,yidx,axis=1), xidx, axis=0) * (Ysc * Xsc)

        # do the out of boundary masking
        oob_mask = logical_or( transpose(resize(out_of_xbounds, 
                        (shape(ZI)[1], shape(ZI)[0])) ),
                    resize(out_of_ybounds, shape(ZI)) )
        #print "oob mask: \n", oob_mask, shape(oob_mask)
        # blind the oob vals i
        # - NOT NEEDED ANYMORE?
        #ZI = ZI*logical_not(oob_mask)
        # set the fill values
        putmask( ZI, oob_mask, self.fill_value)

        # correction for the scalar behaviour in calculations
        #   (dont return full interpolation grid for single values 
        #   in xi or yi)
        ZI = take( take( ZI, range(len(xi)), 0), range(len(yi)), 1)
        #ZI[:len(xi),:len(yi)]

        if mode == 'Point':
            ZI = diag( ZI)

        return (ZI)


    def _check_bounds(self, x_new, y_new):
        # If self.bounds_error = 1, we raise an error if any x_new values
        # fall outside the range of x.  
        # Otherwise, we return arrays indicating
        # which values are outside the boundary region.  

		# TODO: better use min() instead of [0],[-1]?
        below_xbounds = less(x_new, self.x[0])
        above_xbounds = greater(x_new,self.x[-1])
        below_ybounds = less(y_new, self.y[0])
        above_ybounds = greater(y_new,self.y[-1])
        #  Note: sometrue has been redefined to handle length 0 arrays
        # !! Could provide more information about which values are out of bounds
        if self.bounds_error and sometrue(below_xbounds):
            raise ValueError, " A value in x_new is below the"\
                              " interpolation range."
        if self.bounds_error and sometrue(above_xbounds):
            raise ValueError, " A value in x_new is above the"\
                              " interpolation range."
        if self.bounds_error and sometrue(below_ybounds):
            raise ValueError, " A value in y_new is below the"\
                              " interpolation range."
        if self.bounds_error and sometrue(above_ybounds):
            raise ValueError, " A value in y_new is above the"\
                              " interpolation range."
        # !! Should we emit a warning if some values are out of bounds.
        # !! matlab does not.
        out_of_xbounds = logical_or(below_xbounds,above_xbounds)
        out_of_ybounds = logical_or(below_ybounds,above_ybounds)

        return (out_of_xbounds, out_of_ybounds)

