import pylab
import shapely
import numpy as np

DEFAULT_MIN_SIZE = 10
DEFAULT_SAMPLE = 10


def diffxy(line):
    """diff of x,y array in terms of dist."""
    # check it the right way round, otherwise I'll get false results
    if len(line.shape) == 1:
        # return np.sqrt(np.square(line[0])+np.square(line[1]))
        line = line.reshape(1, -1)

    r, c = line.shape
    if r < c:
        line = line.T

    dists = pylab.distances_along_curve(line)
    dists = np.hstack((0, dists))
    return dists

# note this is about 100x as fast as shapely


def resample_polygon(line, n=np.floor(DEFAULT_MIN_SIZE / DEFAULT_SAMPLE), records=[], step=True):
    """
    Interp based on cumdist.

    we loose one point of the length but add it again

    n is samples distance,unless step=False when it's number of points

    good documentation about the errors in this method:
    http://stackoverflow.com/questions/4052225/how-to-equidistant-resample-a-line-or-curve.
    """
    l = np.cumsum(diffxy(line))

    if step:
        if l[-1] < n:
            return line, records
    else:
        if len(line) < n:
            return line, records

    x = line[:, 0]
    y = line[:, 1]
    if step:
        ll = np.arange(min(l), max(l), n)
    else:
        ll = np.linspace(min(l), max(l), n)

    # linear piecewise
    xx = np.interp(ll, l, x)
    yy = np.interp(ll, l, y)

#    from scipy.interpolate import interp1d
#    kind='cubic' # cubic
#    fx=interp1d(l,x,kind=kind)
#    xx=fx(ll)
#    fy=interp1d(l,y,kind=kind)
#    yy=fy(ll)
#    diffxy(ar(zip(xx,yy)))[1:].std()
    # linear: 2016, nearest=2028, slinear=2016, quad: 1E20, cubic

    # from scipy.interpolate import UnivariateSpline

    # now add original end back on so as not too lose data
    xx[-1] = x[-1]
    yy[-1] = y[-1]

    # plot(xx,yy);plot(x,y)
    line2 = np.array(list(zip(xx, yy)))
    if records != []:
        rrecords = []
        for i in xrange(records.shape[1]):
            r = records[:, i]
            rr = np.interp(ll, l, r)
            # rr=np.append(r[0],rr)
            rrecords.append(rr)
        rrecords = np.dstack(rrecords)[0]
        return line2, rrecords
    else:
        return line2
