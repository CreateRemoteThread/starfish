#!/usr/bin/env python3

import getopt
import sys
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import butter,lfilter,freqz
import scipy.signal

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

def background_noise(wavdata,n=5,factor=11,output=True):
  first_slice = wavdata[:int(CFG_SAMPLERATE*0.1)]
  return max(max(first_slice),abs(min(first_slice)))

def f_suppress_silence(in_data,sil_thresh):
  if abs(in_data) >= sil_thresh:
    return in_data
  else:
    return 0

def suppress_silence(in_data,sil_thresh):
  return [f_suppress_silence(d,sil_thresh) for d in in_data]

def doSplitPSD(wave_in,CFG_SAMPLERATE,CFG_SPLITSTEP,CFG_SPLITLEN):
  wav_slices = []
  for i in range(0,len(wave_in),CFG_SPLITSTEP):
    if i + CFG_SPLITLEN >= len(wave_in):
      print("doSplitPSD: discarding last sample")
      break
    wav_slices.append(wave_in[i:i + CFG_SPLITLEN])
  psd_s = [scipy.signal.welch(wave_in,fs=CFG_SAMPLERATE)[1] for wave_in in wav_slices]
  psd_sums = [[sum(psd)] * CFG_SPLITSTEP for psd in psd_s]
  # fig,(ax1,ax2) = plt.subplots(nrows=2)
  # ax1.plot(wave_in)
  # ax2.plot(psd_sums)
  # plt.show()
  return psd_sums

class WaveHelper:
  def __init__(self, fn, CFG_SAMPLERATE=48000, CFG_SKIP = [], CFG_VOLTHRESH=500,CFG_SPLITLEN=500,CFG_SPLITSTEP=250,CFG_WORDSPLIT=0.2):
    self.CFG_SAMPLERATE = CFG_SAMPLERATE
    self.CFG_VOLTHRESH = CFG_VOLTHRESH
    self.CFG_SPLITLEN = CFG_SPLITLEN
    self.CFG_SPLITSTEP = CFG_SPLITSTEP
    self.CFG_WORDSPLIT = CFG_WORDSPLIT
    self.CFG_FN = fn
    self.wavData = raw_data = wavfile.read(fn)[1] 

  def extractPeakSamples(self,plotResults = True):
    psd_sums = np.ndarray.flatten(np.array(doSplitPSD(self.wavData,self.CFG_SAMPLERATE,self.CFG_SPLITSTEP,self.CFG_SPLITLEN)))
    peaks,_ = scipy.signal.find_peaks(psd_sums,height=self.CFG_VOLTHRESH,distance=0.030 * self.CFG_SAMPLERATE,prominence=1000)
    retval = []
    for p in peaks:
      sampleData = self.wavData[p-self.CFG_SPLITLEN:p+4*self.CFG_SPLITLEN]
      localPeak = np.argmax(sampleData) - self.CFG_SPLITLEN
      print("Local peak identified at offset %d" % localPeak)
      realSampleData = np.array(self.wavData[p-self.CFG_SPLITLEN+localPeak:p+4*self.CFG_SPLITLEN+localPeak],np.int32)
      retval.append(realSampleData)
      if plotResults is True:
        plt.plot(realSampleData)
    if plotResults is True:
      plt.show()
    return retval
