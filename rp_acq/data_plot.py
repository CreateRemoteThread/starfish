#!/usr/bin/env python3

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys


fig,(ax1,ax2) = plt.subplots(2)

datafile = h5py.File("write.hdf","a")
traces = datafile["traces"]
for i in range(0,len(sys.argv[1:])):
  ax1.plot(traces[int(sys.argv[1+i])][6500+300:7000+300])
  ax2.plot(traces[int(sys.argv[1+i])])
  
plt.show()
