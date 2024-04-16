#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import numpy as np
import struct
from scipy.signal import butter,lfilter,freqz
import scipy.signal

# raw-typing2: this is me typing raw2lllll

def butter_bandpass(lowcut,highcut,fs,order=5):
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b,a = butter(order, [low, high], btype = 'band')
  return b,a   

def butter_bandpass_filter(data,lowcut,highcut,fs,order=5):
  b,a = butter_bandpass(lowcut,highcut,fs,order=order)
  y = lfilter(b,a,data)
  return y

f = open(sys.argv[1],"rb")

data_x = []
data_y = []
data_z = []

while True:
  try:
    data_x.append(struct.unpack("<h",f.read(2))[0])
    data_y.append(struct.unpack("<h",f.read(2))[0])
    data_z.append(struct.unpack("<h",f.read(2))[0])
  except:
    break

f.close()

CFG_SAMPLERATE=916
CFG_VOLTHRESH=800

fig,(ax1,ax2,ax3) = plt.subplots(nrows=3)

def moving_average(a, n=3):
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n

def detect_outlier(data_1):
  outliers = []
  threshold = -3
  mean_1 = np.mean(data_1)
  std_1 = np.std(data_1)
  for idx in range(0,len(data_1)):
    y = data_1[idx]
    z_score = (y - mean_1) / std_1
    if z_score < threshold:
      outliers.append(idx)
  return outliers

data_sum = []
for i in range(0,len(data_x)):
  data_sum.append(np.abs(data_x[i]) + np.abs(data_y[i]) + np.abs(data_z[i]))

data_sum = butter_bandpass_filter(data_sum,50,457,916,order=1)[100:]
ax1.plot(np.abs(np.array(data_sum)))
plt.show()
