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


print("synchronising...")
while True:
  while ser.inWaiting() < 10:
    pass
  sync1 = ser.read(2)
  while sync1 != b"\xFF\xFF":
    while ser.inWaiting() < 3:
      pass
    ser.read(1)
    print(sync1.hex())
    sync1 = ser.read(2)
  x = ser.read(2)
  y = ser.read(2)
  z = ser.read(2)
  sync2 = ser.read(2)
  if sync1 == sync2 and sync1 == b"\xff\xff":
    print("synced!")
    break
  else:
    print(sync1.hex()+sync2.hex())
    ser.read(1)
  
while True:
  while ser.inWaiting() < 8:
    pass
  data = ser.read(6)
  sync = ser.read(2)
  if sync != b"\xff\xff":
    print("sync broken")
    f.flush()
    break
  else:
    loggedData += 1
    f.write(data)
  if loggedData % 3000 == 0 and loggedData != 0:
    f.flush()

f.close()
