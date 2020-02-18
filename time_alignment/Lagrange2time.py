
from libsmop import *


    
@function
def Lagrange2time(xdata=None,ydata=None,xi=None,*args,**kwargs):
    varargin = Lagrange2time.varargin
    nargin = Lagrange2time.nargin

    x0=xdata[0]

    y0=ydata[0]

    x1=xdata[1]

    y1=ydata[1]

    x2=xdata[2]

    y2=ydata[2]

    yi=dot(dot(y0,(xi - x1)),(xi - x2)) / (dot((x0 - x1),(x0 - x2))) + dot(dot(y1,(xi - x0)),(xi - x2)) / (dot((x1 - x0),(x1 - x2))) + dot(dot(y2,(xi - x0)),(xi - x1)) / (dot((x2 - x0),(x2 - x1)))

    return yi
    
if __name__ == '__main__':
    pass
    
