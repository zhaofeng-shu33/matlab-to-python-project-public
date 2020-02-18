import numpy as np
zeros = np.zeros
eye = np.eye
mean = np.mean
from libsmop import randn, floor, length
from Lagrange1time import Lagrange1time
from Lagrange2time import Lagrange2time
from LSFit1time import LSFit1time
from LSFit2time import LSFit2time
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
title = plt.title
subplot = plt.subplot
plot = plt.plot
xlabel = plt.xlabel
ylabel = plt.ylabel
ylim = plt.ylim
legend = plt.legend
stem = plt.stem
mpl.rcParams['font.sans-serif'] = ['Songti SC'] + mpl.rcParams['font.sans-serif']
