#!/usr/bin/env python3

# arbitrary sample generator

import termios
import tty
import sys
import random
import getopt
import time

def usage(numWords):
  print("type %d lowercase words, hitting space after each one. use letters only, do not use backspace" % numWords)

def doTrain(numWords,fn):
  f = open(fn,"w")
  stdin = sys.stdin.fileno()
  tattr = termios.tcgetattr(stdin)
  tty.setcbreak(stdin,termios.TCSANOW)
  for i in range(0,numWords):
    timingData = []
    c = ""
    while True:
      x = sys.stdin.read(1)
      if x == ' ':
        break
      c += x
      timingData.append(time.time())
    print("%d words remaining" % (numWords - i - 1))
    f.write("%s:%s\n" % (c,timingData))
  f.close()
  termios.tcsetattr(stdin,termios.TCSANOW,tattr)

if len(sys.argv) != 3:
  print("./samplegen.py [numWords] [out.data]")
  sys.exit(0)

usage(int(sys.argv[1]))
doTrain(int(sys.argv[1]),sys.argv[2])
