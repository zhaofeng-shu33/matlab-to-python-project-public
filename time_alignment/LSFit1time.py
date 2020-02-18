

from libsmop import *



    
@function
def LSFit1time(xdata=None,ydata=None,xi=None,*args,**kwargs):
    varargin = LSFit1time.varargin
    nargin = LSFit1time.nargin

    sx2=sum(xdata ** 2)


    sx=sum(xdata ** 1)


    sn=length(xdata ** 0)


    sxy=sum(multiply(xdata,ydata))


    sy=sum(ydata)


    m=concat([[sx2,sx],[sx,sn]])


    n=concat([[sxy],[sy]])


    c=numpy.linalg.solve(m,n)


    yi=multiply(c[0],xi) + c[1]


    return yi
    
if __name__ == '__main__':
    pass
