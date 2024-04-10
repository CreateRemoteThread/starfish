#!/usr/bin/env python3

import platform
import sys
import keras
import numpy as np
import tensorflow as tf

if len(sys.argv) != 3:
  print("./analysis.py [training.data] [test.data]")
  sys.exit(0)

def loadDataset(fn,labelDict=None):
  f = open(fn,"r")
  l = [l_.rstrip() for l_ in f.readlines()]
  f.close()
  print(len(l))
  trainingLabels = []
  trainingData = np.zeros((len(l),10),np.float32)
  writeHead = 0
  for line in l:
    word,timingData = line.split(":")
    timingData = timingData.replace("[","")
    timingData = timingData.replace("]","")
    timingData = timingData.replace("\'","")
    trainingLabels.append(word)
    # timingRaw = [float(f) for f in timingData.split(",")]
    timingRaw = [round(float(f),3) for f in timingData.split(",")]
    timingDiff = np.zeros(5,np.float32)
    timingNorm = np.zeros(5,np.float32)
    for i in range(0,5):
      timingDiff[i] = timingRaw[i+1] - timingRaw[i]
    timingNorm[0] = 0.5
    for i in range(1,5):
      timingNorm[i] = (timingDiff[i] / timingDiff[0]) * 0.5
    timingTotal = np.zeros(10,np.float32)
    timingTotal[0:5] = timingDiff
    timingTotal[5:10] = timingNorm
    trainingData[writeHead:] = timingTotal
    writeHead += 1
  if labelDict is None:
    trainingDict = {}
  else:
    trainingDict = labelDict
  labelCtr = 0
  for l in trainingLabels:
    if l not in trainingDict.keys():
      trainingDict[l] = labelCtr
      labelCtr += 1
  numericTrainingLabels = np.array([trainingDict[l] for l in trainingLabels],np.uint8)
  return (trainingData,numericTrainingLabels,trainingDict)

print("Loading data...")
(trainData,trainLabels,trainDict) = loadDataset(sys.argv[1])
(testData,testLabels,testDict) = loadDataset(sys.argv[2],trainDict)

print("Training...")
mdl = tf.keras.models.Sequential()
mdl.add(tf.keras.layers.Flatten())
# mdl.add(tf.keras.layers.Dense(128,activation="relu"))
# mdl.add(tf.keras.layers.Dense(32,activation="softmax"))
mdl.add(tf.keras.layers.Dense(128,activation="relu"))
mdl.add(tf.keras.layers.Dense(32,activation="softmax"))
mdl.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.005),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
mdl.summary()
mdl.fit(trainData,trainLabels,epochs=100,batch_size=24,validation_split=0.10)

def flipLabels(inDict):
  outDict = {}
  for i in inDict.keys():
    k = inDict[i]
    outDict[k] = i
  return outDict

recoverDict = flipLabels(testDict)
recoveredWords = []

print("Predicting...")
predictData = mdl.predict(testData)

print("Summarizing...")
for i in range(0,len(predictData)):
  prediction = np.argmax(predictData[i])
  recoveredWords.append(recoverDict[prediction])

f = open(sys.argv[2],"r")
actualWords = [l.split(":")[0] for l in f.readlines()]
f.close()

correctWords = 0

for i in range(0,len(actualWords)):
  print("ACTUAL: %s | GUESS: %s" % (actualWords[i],recoveredWords[i]))
  if actualWords[i] == recoveredWords[i]:
    correctWords += 1

print("Accuracy: %d/%d" % (correctWords,len(actualWords)))
