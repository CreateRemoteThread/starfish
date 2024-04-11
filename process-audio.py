#!/usr/bin/env python3

# process-audio - convert wav to timing distance

import getopt
import sys
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import butter,lfilter,freqz
import scipy.signal

if len(sys.argv) < 2:
  usage()
  sys.exit(0)

CFG_SAMPLERATE = 48000
CFG_FREQLOW    = 5000
CFG_FREQHIGH   = 23000
CFG_INFILE     = None
CFG_SKIP       = []
CFG_VOLTHRESH  = 50
CFG_WRITEFILE  = "cheeseburger.data"

CFG_SPLITLEN   = 500
CFG_SPLITSTEP  = 250

args, rems = getopt.getopt(sys.argv[1:],"f:s:l:h:t:w:",["file=","samplerate=","lowcut=","highcut=","skip=","thresh=","writefile="])
for arg, val in args:
  if arg in ["-f","--file"]:
    CFG_INFILE = val
  elif arg in ["-s","--samplerate"]:
    CFG_SAMPLERATE = int(val)
  elif arg in ["-l","--lowcut"]:
    CFG_FREQLOW = int(val)
  elif arg in ["-h","--highcut"]:
    CFG_FREQHIGH = int(val)
  elif arg in ["-t","--thresh"]:
    CFG_VOLTHRESH = int(val)
  elif arg in ["-w","--writefile"]:
    CFG_WRITEFILE = val
  elif arg in ["--skip"]:
    print("Parsing --skip option...")
    if "," in val:
      CFG_SKIP = [int(x) for x in val.split(",")]
    else:
      CFG_SKIP.append( int(val) )

if CFG_INFILE is None:
  print("You must specify an input wavfile with -f")
  sys.exit(0)

if CFG_FREQHIGH >= (CFG_SAMPLERATE // 2):
  print("Bandpass highcut (-h) must be below half of sample rate")
  sys.exit(0)

if CFG_FREQLOW >= CFG_FREQHIGH:
  print("Bandpass lowcut (-l) must be lower than highcut (-h)")
  sys.exit(0)

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

# CFG_SPLITSTEP = how far to move cursor forward
# CFG_SPLITLEN  = how many samples to read
import random
def doSplitPSD(wave_in):
  wav_slices = []
  for i in range(0,len(wave_in),CFG_SPLITSTEP):
    if i + CFG_SPLITLEN >= len(wave_in):
      print("doSplitPSD: discarding last sample")
      break
    wav_slices.append(wave_in[i:i + CFG_SPLITLEN])
  psd_s = [scipy.signal.welch(wave_in,fs=CFG_SAMPLERATE)[1] for wave_in in wav_slices]
  psd_sums = [[sum(psd)] * CFG_SPLITSTEP for psd in psd_s]
  fig,(ax1,ax2) = plt.subplots(nrows=2)
  ax1.plot(wave_in)
  ax2.plot(psd_sums)
  plt.show() 

data = wavfile.read(CFG_INFILE)[1]
doSplitPSD(data)

fig,(ax1,ax2) = plt.subplots(2,1)
ax1.specgram(data,Fs=CFG_SAMPLERATE,NFFT=256)
step1 = butter_bandpass_filter(data,CFG_FREQLOW,CFG_FREQHIGH,CFG_SAMPLERATE,3)
step2 = suppress_silence(step1,background_noise(step1))

peaks,_ = scipy.signal.find_peaks(step1,height=CFG_VOLTHRESH,distance=0.075 * CFG_SAMPLERATE)
# peaks = peaks[2:]
print(peaks)

filt_peaks = []
for i in range(0,len(peaks)):
  if i in CFG_SKIP:
    print("Discarding sample %d" % i)
  else:
    filt_peaks.append(peaks[i])

print("Samples remaining: %d" % len(filt_peaks))

peaks = filt_peaks

with open(CFG_OUTFILE,"w") as f:
  sectimes = [float(peak / CFG_SAMPLERATE) for peak in peaks]
  for i in range(0,len(sectimes),6):
    f.write("??????:%s\n" % sectimes[i:i+6])

ax2.margins(x=0)
ax2.plot(step1)
ax2.plot(peaks,step1[peaks],"x")
plt.show()

