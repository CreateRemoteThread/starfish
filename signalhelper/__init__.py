#!/usr/bin/env python3

import getopt
import sys
import numpy as np
import copy
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import butter,lfilter,freqz
import scipy.signal

def butter_lowpass(cutoff, fs, order=5):
  nyq = 0.5 * fs
  normal_cutoff = cutoff / nyq
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
  b, a = butter_lowpass(cutoff, fs, order=order)
  y = lfilter(b, a, data)
  return y

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
      print("doSplitPSD: last sample > wave length, discarding")
      break
    wav_slices.append(wave_in[i:i + CFG_SPLITLEN])
  psd_s = [scipy.signal.welch(wave_in,fs=CFG_SAMPLERATE)[1] for wave_in in wav_slices]
  psd_sums = [[sum(psd)] * CFG_SPLITSTEP for psd in psd_s]
  return psd_sums

def simpleFFT(d,FFT_BASEFREQ=48000):
  n = len(d)
  k = np.arange(n)
  T = n / FFT_BASEFREQ
  frq = k / T
  frq = frq[list(range(n // 2))]
  Y = np.fft.fft(d)/n
  Y = Y[list(range(n //  2))]
  return abs(Y)

def normalizeSlice(oneSignal, sigMin=None,sigMax=None):
  if sigMin is None and sigMax is None:
    localMin = min(oneSignal)
    localMax = max(oneSignal)
  else:
    localMin = sigMin
    localMax = sigMax
  signal = np.copy(oneSignal)
  signal -= localMin
  signal /= np.float32(localMax)
  signal -= 0.5
  signal *= 2
  return signal

def getMaxCorrCoeff(sigref,sigxcorr,minslide=-200,maxslide=200):
  maxCf = 0
  maxCfIndex = 0
  for i in range(minslide,maxslide):
    sig_test = np.roll(sigxcorr,i)
    try:
      r = np.corrcoef(sigref,sig_test)
    except:
      print("getMaxCorrCoef: error, passing to next")
    if r[0,1] > maxCf:
      maxCf = r[0,1]
      maxCfIndex = i
  return maxCfIndex

from . import mel
def featureExtract(sig_in):
  retval = []
  for sig in sig_in:
    retval.append(mel.mfcc(y=sig,n_mfcc=16,sr=48000,n_fft=240,hop_length=120))
  # print(retval)
  globalMax = np.max(retval)
  globalMin = np.min(retval)
  retval = [normalizeSlice(r,sigMax=globalMax,sigMin=globalMin) for r in retval]
  return retval
 
def realign(signalData,reftrace=0):
  newSignalData = [signalData[0]]
  for i in range(1,len(signalData)):
    mcf = getMaxCorrCoeff(signalData[0],signalData[i])
    newSignalData.append(np.roll(signalData[i],mcf))
  return newSignalData

CFG_HIGHCUT = 480
CFG_LOWCUT  = 60
def realignFilter(signalData,reftrace=0):
  newOrig = butter_bandpass_filter(signalData[0],CFG_LOWCUT,CFG_HIGHCUT,48000,3)
  newSignalData = [newOrig[500:-500]]
  for i in range(1,len(signalData)):
    tempData = butter_bandpass_filter(signalData[i],CFG_LOWCUT,CFG_HIGHCUT,48000,3)
    mcf = getMaxCorrCoeff(newOrig,tempData)
    newSignalData.append(np.roll(tempData,mcf)[500:-500])
  return newSignalData

 
class WaveHelper:
  def __init__(self, fn, CFG_SAMPLERATE=48000, CFG_SKIP = [], CFG_VOLTHRESH=400,CFG_SPLITLEN=480,CFG_SPLITSTEP=240,CFG_WORDSPLIT=0.2):
    self.CFG_SAMPLERATE = CFG_SAMPLERATE
    self.CFG_VOLTHRESH = CFG_VOLTHRESH
    self.CFG_SPLITLEN = CFG_SPLITLEN
    self.CFG_SPLITSTEP = CFG_SPLITSTEP
    self.CFG_WORDSPLIT = CFG_WORDSPLIT
    self.CFG_BEFORESLICES = 4
    self.CFG_AFTERSLICES = 8
    self.CFG_FN = fn
    self.wavData = [float(sample) for sample in wavfile.read(fn)[1]]
    # self.normData = normalizeSlice(self.wavData,min(self.wavData),max(self.wavData))

  def extractPeakSamples(self,plotResults = True):
    psd_sums = np.ndarray.flatten(np.array(doSplitPSD(self.wavData,self.CFG_SAMPLERATE,self.CFG_SPLITSTEP,self.CFG_SPLITLEN)))
    peaks,_ = scipy.signal.find_peaks(np.abs(psd_sums),height=self.CFG_VOLTHRESH,distance=0.030 * self.CFG_SAMPLERATE,prominence=1000)
    retval = []
    retval_fft = []
    if plotResults is True:
      fig,(ax1,ax2,ax3) = plt.subplots(nrows=3)
      ax1.margins(x=0)
      ax1.plot(self.wavData)
    for p in peaks:
      sampleData = self.wavData[p-self.CFG_BEFORESLICES*self.CFG_SPLITLEN:p+self.CFG_AFTERSLICES*self.CFG_SPLITLEN]
      localPeak = np.argmax(np.abs(sampleData)) - self.CFG_BEFORESLICES*self.CFG_SPLITLEN
      if plotResults is True:
        ax1.vlines(x=[p-self.CFG_BEFORESLICES*self.CFG_SPLITLEN+localPeak],ymin=-1000,ymax=1000,color="red")
        ax1.vlines(x=[p+self.CFG_AFTERSLICES*self.CFG_SPLITLEN+localPeak],ymin=-1000,ymax=1000,color="green")
      realSampleData = np.array(self.wavData[p-self.CFG_BEFORESLICES*self.CFG_SPLITLEN+localPeak:p+self.CFG_AFTERSLICES*self.CFG_SPLITLEN+localPeak],np.float32)
      # retval_fft.append(list(simpleFFT(realSampleData)))
      realSampleData = normalizeSlice(realSampleData)
      retval.append(realSampleData)
    if plotResults is True:
      # print(self.wavData)
      ax1.plot(peaks,np.array(self.wavData)[peaks],"x")
      ax2.specgram(self.wavData,NFFT=1024,Fs=48000,noverlap=900)
    retval = realignFilter(retval)
    retval_fft = featureExtract(retval)
    if len(retval) == 1:
      print("Error: only one sample provided for training key %s" % self.CFG_FN)
      return retval
    retval_backup = copy.copy(retval)
    refHead = 0
    sigref = retval[refHead]
    deleteIndexes = []
    # 50% of samples must remain, otherwise we picked a bad reference.
    for i in range(1,len(retval)):
      r = np.corrcoef(sigref,retval[i])
      if 1.0 - r[0,1] > 0.2:
        deleteIndexes.append(i)
    while len(deleteIndexes) > 0.5 * len(retval):
      refHead += 1
      print("Warning: Internal corrcoef check failed too many samples, changing sigref to %d FIXME REFHEAD CASE" % refHead)
      sigref = retval[refHead]
      deleteIndexes = 0
      for i in range(1,len(retval)):
        r = np.corrcoef(sigref,retval[i])
        if 1.0 - r[0,1] > 0.2:
          deleteIndexes.append(i)
    print("Flushing %d samples" % len(deleteIndexes))
    deleteIndexes.sort()
    deleteIndexes = deleteIndexes[::-1]
    print(deleteIndexes)
    for a in deleteIndexes:
      del(retval[a])
      del(retval_fft[a])
    if plotResults is True:
      for v in retval:
        ax3.plot(v)
    plt.show()
    return (retval,retval_fft)
