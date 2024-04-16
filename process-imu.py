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

# f.read(1)
aud = []

while True:
  try:
    aud.append(struct.unpack("<h",f.read(2))[0])
  except:
    break

f.close()

data = np.array(aud)
data = np.abs(butter_bandpass_filter(data,150,457,916,order=3)[100:])
CFG_SAMPLERATE=916
CFG_VOLTHRESH=800

fig,ax1 = plt.subplots(nrows=1)

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

ax1.plot(data)
plt.show()
