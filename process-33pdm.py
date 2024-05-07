#!/usr/bin/env python3

# converts raw 16-bit samples (from nano 33 ble gather) to wav

import wave
import sys
import matplotlib.pyplot as plt
import numpy as np
import struct
from scipy.signal import butter,lfilter

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

fig,(ax1,ax2) = plt.subplots(nrows=2)
aud_f = np.array([float(a) for a in aud],np.float32)
aud_f_highpass = butter_bandpass_filter(aud_f,400,7999,16000,1)

ax1.specgram(aud_f_highpass,Fs=16000,NFFT=1024)
ax2.plot(aud)

w = wave.open("output.wav","wb")
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(16000)
w.writeframes(b"".join([struct.pack("<h",x) for x in aud]))
w.close()

plt.show()
