#!/usr/bin/env python3

import sys
import serial

ser = serial.Serial(port=sys.argv[1],baudrate=921600)
if not ser.isOpen():
  print("Failed serial open")

f = open(sys.argv[2],"w")

while True:
  try:
    strings = str(ser.readline())
    strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
    index = strings.find('CSI_DATA')
    if index == -1:
      continue
    f.write(strings + "\n")
  except KeyboardInterrupt:
    print("bye!")
    ser.close()
    f.flush()
    f.close()
