#!/usr/bin/env python3

import getopt  
import sys
import signalhelper
import keras
import tensorflow as tf
import numpy as np

CFG_TRAIN = []
CFG_FIT   = []

args, rems = getopt.getopt(sys.argv[1:],"t:f:",["train=","fit="])
for arg, val in args:
  if arg in ["-t","--train"]:
    CFG_TRAIN.append(val)
  elif arg in ["-f","--fit"]:
    CFG_FIT.append(val)

print("Step 1: Extract sample data")
trainingData = {}

for fn in CFG_TRAIN: 
  fs = signalhelper.WaveHelper(fn)
  trainingData[fn] = fs.extractPeakSamples(plotResults=False)

trainSamples = []
trainLabels = []
trainDict = {}

print("Step 2: Reshape into tensors array")
trainHead = 0
for k in trainingData.keys():
  trainDict[trainHead] = fn
  trainSamples += trainingData[fn]
  trainLabels += [trainHead] * len(trainingData[fn])
  trainHead += 1

trainLabels = np.array(trainLabels,np.int8)
trainSamples = np.array(trainSamples)

def normalizeSlice(oneSignal, sigMin,sigMax):
  signal = np.copy(oneSignal)
  signal -= sigMin
  signal /= np.float32(sigMax)
  signal -= 0.5
  signal *= 2
  return signal

def normalize(allSignals):
  globalMin = allSignals.min()
  globalMax = allSignals.max()
  # normalize all slice volumes
  newSignals = [normalizeSlice([float(os) for os in oneSig],globalMin,globalMax) for oneSig in allSignals]
  return np.array(allSignals)

trainSamples = normalize(trainSamples)

print(trainLabels)

mdl = tf.keras.models.Sequential()
mdl.add(tf.keras.layers.Flatten())
mdl.add(tf.keras.layers.Dense(128,activation="relu"))
mdl.add(tf.keras.layers.Dense(64,activation="relu"))
mdl.add(tf.keras.layers.Dense(32,activation="softmax"))
mdl.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.005),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
mdl.summary()
mdl.fit(trainSamples,trainLabels,epochs=100,batch_size=24,validation_split=0.10)


