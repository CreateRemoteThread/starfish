#!/usr/bin/env python3

import getopt  
import sys
import signalhelper
import keras
import tensorflow as tf
import numpy as np

CFG_TRAIN   = []
CFG_FIT     = []
CFG_VERBOSE = False

args, rems = getopt.getopt(sys.argv[1:],"t:f:v",["train=","fit=","verbose"])
for arg, val in args:
  if arg in ["-t","--train"]:
    CFG_TRAIN.append(val)
  elif arg in ["-f","--fit"]:
    CFG_FIT.append(val)
  elif arg in ["-v","--verbose"]:
    CFG_VERBOSE = True

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
for k in trainingData.keys():
  trainDict[trainHead] = fn
  (peakSamples,peakSamples_fft) = trainingData[fn]
  for i in range(0,len(peakSamples)):
    trainSamples.append(np.concatenate((peakSamples[i],peakSamples_fft[i])))
    trainLabels += [trainHead]
  trainHead += 1

trainLabels = np.array(trainLabels,np.int8)
trainSamples = np.array(trainSamples)

mdl = tf.keras.models.Sequential()
mdl.add(tf.keras.layers.Flatten())
mdl.add(tf.keras.layers.Dense(128,activation="relu"))
mdl.add(tf.keras.layers.Dense(128,activation="relu"))
mdl.add(tf.keras.layers.Dense(32,activation="softmax"))
mdl.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0025),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
mdl.summary()

mdl.fit(trainSamples,trainLabels,epochs=100,batch_size=24,validation_split=0.10)

