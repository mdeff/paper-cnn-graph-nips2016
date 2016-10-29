#!/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 18})

fourier = np.array([10,23,68,139.5,231.5,351,495])/400*1000
cheby = np.array([7.5,11.5,20,29.5,38,48.5,59])/400*1000
x = np.array([1,2,4,6,8,10,12]) * 1000

plt.figure(figsize=(15,5))
plt.plot(x, cheby, '.-', markersize=15, label='Chebyshev');
plt.plot(x, fourier, '.--', markersize=15, label='Non-Param / Spline');
plt.legend(loc='upper left')
plt.xlim(1000,12000)
plt.xlabel('number of features (words)')
plt.ylabel('time (ms)')
plt.savefig('runtime.pdf')
