
from libsmop import *


    
@function
def Lagrange1time(xdata=None,ydata=None,xi=None,*args,**kwargs):
    varargin = Lagrange1time.varargin
    nargin = Lagrange1time.nargin

    x0=xdata[1]

    y0=ydata[1]

    x1=xdata[2]

    y1=ydata[2]

    yi=dot(y0,(xi - x1)) / (x0 - x1) + dot(y1,(xi - x0)) / (x1 - x0)

    return yi
    
if __name__ == '__main__':
    pass
    
