#!/usr/bin/env python3

import termios
import tty
import sys
import random
import time

wordsDict = []
with open("wordlist.txt","r") as f:
  wordsDict = [l.rstrip() for l in f.readlines()]

try:
  wordsDict.remove("")
except:
  pass

def doTrain(numWords,fn):
  global wordsDict
  random.seed()
  f = open(fn,"w")
  stdin = sys.stdin.fileno()
  tattr = termios.tcgetattr(stdin)
  tty.setcbreak(stdin,termios.TCSANOW)
  for i in range(0,numWords):
    chosenWord = random.choice(wordsDict)
    print("[%d:%d] %s" % (i,numWords,chosenWord))
    timingData = []
    c = ""
    for letter in range(0,6):
      c += sys.stdin.read(1)
      timingData.append(time.time())
    if c != chosenWord:
      print("typo - discarding")
      sys.stdin.flush()
      continue
    else:
      f.write("%s:%s\n" % (c,timingData))
      # print(timingData)
  f.close()
  termios.tcsetattr(stdin,termios.TCSANOW,tattr)

if len(sys.argv) != 3:
  print("./samplegen.py [numWords] [out.data]")
  sys.exit(0)

doTrain(int(sys.argv[1]),sys.argv[2])
