#!/usr/bin/env python3

import sys
import serial

if len(sys.argv) != 3:
  print("./gather-33pdm.py [inserial] [outfile]")
  sys.exit(0)

print("press ^c to exit...")
f = open(sys.argv[2],"wb")
ser = serial.Serial(sys.argv[1],115200)
loggedData = 1
while True:
  loggedData += 1
  while ser.inWaiting() < 2:
    pass
  data = ser.read(2)
  f.write(data)
  if loggedData % 16000 == 0 and loggedData != 0:
    f.flush()

f.close()
