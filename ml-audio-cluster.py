#!/usr/bin/env python3

# attempt 2 - statistical clustering

import getopt  
import sys
import signalhelper
import signalhelper.mel
import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import DBSCAN,AgglomerativeClustering,OPTICS

CFG_TRAIN   = []
CFG_FIT     = []
CFG_VERBOSE = False
CFG_PLTACC  = False

args, rems = getopt.getopt(sys.argv[1:],"t:f:va",["train=","fit=","verbose","acc"])
for arg, val in args:
  if arg in ["-t","--train"]:
    CFG_TRAIN.append(val)
  elif arg in ["-f","--fit"]:
    CFG_FIT.append(val)
  elif arg in ["-v","--verbose"]:
    CFG_VERBOSE = True
  elif arg in ["-a","--acc"]:
    CFG_PLTACC = True

import sys

def custom_metric(x,y):
  r = np.corrcoef(x,y)
  if 1.0 - r[0,1] > 0.1:
    print("eps > 0.1")
  return (1.0 - r[0,1])

print("Step 1: Extract sample data")
trainingData = {}

for fn in CFG_TRAIN:
  fs = signalhelper.WaveHelper(fn)
  (peakSamples,peakSamples_fft) = fs.extractPeakSamples(plotResults=CFG_VERBOSE)
  trainingData[fn] = (peakSamples,peakSamples_fft)

trainSamples = []
trainLabels = []
trainDict = {}

print("Step 1 Summary")
for k in trainingData.keys():
  print("Key %s has %d samples" % (k,len(trainingData[k][1])))

if CFG_VERBOSE is True:
  input("Press enter to continue...")

print("Step 2: Reshape into tensors array")
trainHead = 0
for fn in trainingData.keys():
  trainDict[trainHead] = fn
  (peakSamples,peakSamples_fft) = trainingData[fn]
  # print(len(peakSamples))
  for i in range(0,len(peakSamples)):
    trainSamples.append(peakSamples[i])
    trainLabels += [trainHead]
  trainHead += 1

trainLabels = np.array(trainLabels,np.int8)
trainSamples = np.array(trainSamples)

db = DBSCAN(min_samples=25,eps=0.15,metric=custom_metric).fit(trainSamples)
# db = OPTICS(min_samples=25,max_eps=0.3,eps=0.1,metric=custom_metric).fit(trainSamples)
labels= db.labels_
print(labels)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
