
from libsmop import *


    
@function
def LSFit2time(xdata=None,ydata=None,xi=None,*args,**kwargs):
    varargin = LSFit2time.varargin
    nargin = LSFit2time.nargin

    sx4=sum(xdata ** 4)

    sx3=sum(xdata ** 3)

    sx2=sum(xdata ** 2)

    sx=sum(xdata ** 1)

    sn=length(xdata ** 0)

    sx2y=sum(dot(xdata ** 2.0,ydata))

    sxy=sum(multiply(xdata,ydata))

    sy=sum(ydata)

    m=concat([[sx4,sx3,sx2],[sx3,sx2,sx],[sx2,sx,sn]])

    n=concat([[sx2y],[sxy],[sy]])

    c=numpy.linalg.solve(m,n)

    yi=dot(c[0],xi ** 2) + dot(c[1],xi) + c[2]

    return yi
    
if __name__ == '__main__':
    pass
    
