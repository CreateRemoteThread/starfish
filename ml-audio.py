#!/usr/bin/env python3

import getopt  
import sys
import signalhelper
import signalhelper.mel
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class PlotLearning(tf.keras.callbacks.Callback):
  def on_train_begin(self,logs={}):
    self.metrics = {}
    for metric in logs:
      self.metrics[metric] = []

  def getLastAccuracy(self):
    return (self.metrics['loss'],self.metrics['accuracy'])

  def on_epoch_end(self,epoch,logs={}):
    for metric in logs:
      if metric in self.metrics:
        self.metrics[metric].append(logs.get(metric))
      else:
        self.metrics[metric] = [logs.get(metric)]

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

print("Step 1: Extract sample data")
trainingData = {}

for fn in CFG_TRAIN:
  fs = signalhelper.WaveHelper(fn)
  (peakSamples,peakSamples_feat) = fs.extractPeakSamples(plotResults=CFG_VERBOSE)
  # (peakSamples,peakSamples_feat) = fs.extractPeakMessy(plotResults=CFG_VERBOSE)
  trainingData[fn] = (peakSamples,peakSamples_feat)

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
  (peakSamples,peakSamples_feat) = trainingData[fn]
  for i in range(0,len(peakSamples)):
    # trainSamples.append(peakSamples_feat[i])
    print(peakSamples[i])
    trainSamples.append(peakSamples[i])
    trainLabels.append(trainHead)
  trainHead += 1

trainLabels = np.array(trainLabels,np.int8)
# trainSamples = np.array([np.array(t).flatten() for t in trainSamples])
trainSamples = np.array(trainSamples)

mdl = tf.keras.models.Sequential()
mdl.add(tf.keras.layers.Flatten())
mdl.add(tf.keras.layers.Dense(128,activation="relu"))
mdl.add(tf.keras.layers.Dense(64,activation="relu"))
mdl.add(tf.keras.layers.Dense(32,activation="relu"))
mdl.add(tf.keras.layers.Dense(8,activation="softmax"))
mdl.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

mdl.summary()

history = mdl.fit(trainSamples,trainLabels,epochs=100,batch_size=24,validation_split=0.10)
if CFG_PLTACC is True:
  plt.plot(history.history['accuracy'])
  plt.show()
