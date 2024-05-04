#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import math
import numpy as np

dataArray = []
perm_phase = []
perm_amp = []

def moving_average(a, n=3):
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n

def translate_array(a):
  channel_data_len = len(a[0])
  sample_time_len =  len(a)
  for i in range(1,len(a)):
    if len(a[i]) != channel_data_len:
      print("Sanity: incorrect channel data")
  channel_data = {}
  for i in range(0,channel_data_len):
    channel_data[i] = moving_average([sublist[i] for sublist in a],n=5)
  # for c in channel_data.keys():
  #   plt.plot(moving_average(channel_data[c],n=5),label="Channel %d" % c)
  # plt.legend()
  # plt.show()
  return channel_data

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("usage: ./viz.py [filename]")
    sys.exit(0)
  f = open(sys.argv[1],"rb")
  for d in f.readlines():
    tokens = d.rstrip().split(b",")
    csi_data = tokens[24:]
    csi_data[0] = csi_data[0].replace(b"\"[",b"")
    csi_data[-1] = csi_data[-1].replace(b"]\"",b"")
    csi_data.pop()
    # print(len(csi_data))
    # print(csi_data)
    csi_data_ints = [int(c) for c in csi_data]
    # print(csi_data_ints) 
    # see https://github.com/StevenMHernandez/ESP32-CSI-Tool
    imaginary = []
    real = []
    csi_size = len(csi_data_ints)
    for i, val in enumerate(csi_data_ints):
      if i % 2 == 0:
        imaginary.append(val)
      else:
        real.append(val)
    amplitudes = []
    phases = []
    for j in range(int(csi_size / 2)):
      amplitude_calc = math.sqrt(imaginary[j] ** 2 + real[j] ** 2)
      phase_calc = math.atan2(imaginary[j], real[j])
      amplitudes.append(amplitude_calc)
      phases.append(phase_calc)
    perm_phase.append(phases)
    perm_amp.append(amplitudes)
  plt.plot(perm_amp)
  # perm_amp_sum = [sum(sub_carrier) for sub_carrier in perm_amp]
  # plt.plot(moving_average(perm_amp_sum,n=8))
  plt.show()
  f.close()
