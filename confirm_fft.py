# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:19:54 2019

@author: a-kojima
"""
import numpy as np
import matplotlib.pyplot as pl

SPECTRUM_INFO = r'./test.txt'
FFT_SIZE = 400

f = open(SPECTRUM_INFO, 'r', encoding='shift_jis')
lines = f.readlines()

save_spec = np.zeros((FFT_SIZE, len(lines)))

for ii in range(0, len(lines)):
    t = lines[ii].split(',')
    print(len(t))
    for jj in range(0, FFT_SIZE):
        save_spec[jj, ii] = np.float(t[jj])


pl.figure()
pl.subplot(2,1,1)
pl.imshow(save_spec, aspect='auto')
pl.colorbar()

pl.subplot(2,1,2)
pl.imshow(10 * np.log10(save_spec**2), aspect='auto')
pl.colorbar()
pl.show()
        
        