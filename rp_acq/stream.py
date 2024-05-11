#!/usr/bin/env python3

import signal
import random
import sys
import socket
import struct
import binascii
import h5py
import numpy as np

outfile = h5py.File("write.hdf","w")
traces = np.zeros((20,16416))
dataCount = 0

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

RP_HOST = "192.168.2.2"
RP_PORT = 8900

ID_PACK = [0xFF,0xFF,0xFF,0xFF,0xA0,0xA0,0xA0,0xA0,0xFF,0xFF,0xFF,0xFF,0xA0,0xA0,0xA0,0xA0]
ID_PACK_END = [0xFF,0xFF,0xFF,0xFF,0x50,0x50,0x50,0x50,0xFF,0xFF,0xFF,0xFF,0x50,0x50,0x50,0x50]
ID_BUFFER = [0xFF,0xFF,0xFF,0xFF,0x0A,0x0A,0x0A,0x0A,0xFF,0xFF,0xFF,0xFF,0x0A,0x0A,0x0A,0x0A]

class Scope:
  def __init__(self,ax):
    self.ax = ax
    self.data =[random.randint(0,0xFF) for i in range(0,16416)]
    self.line = Line2D(range(0,16416),self.data)
    self.ax.add_line(self.line)
    self.ax.set_ylim(-0xFFFF,0xFFFF)
    self.ax.set_xlim(0,16416)

  def update(self,data):
    print(len(data))
    self.line.set_data(range(0,16416),data)
    return self.line,

def recvall(sock,lenx):
  databuf = b""
  remaining = lenx
  while remaining != 0:
    newdata = sock.recv(remaining)
    remaining -= len(newdata)
    databuf += newdata
  return databuf 

buf_decimate = 0

notShown = True

plt.ion()
fig,ax = plt.subplots()
scope = Scope(ax)

data_buf = []

def emitter(p=0.1):
  global data_buf
  while True:
    yield(data_buf)

ani = animation.FuncAnimation(fig,scope.update,emitter,interval=500,blit=True,save_count=100)

plt.show()

sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock.connect((RP_HOST,RP_PORT))

while True:
  packet_hdr = sock.recv(16)
  packet_size = struct.unpack("<q",sock.recv(8))[0]
  packet_data = recvall(sock, packet_size - 24)
  if packet_hdr[4:8] == b"\x0a\x0a\x0a\x0a":
    data_buf = []
    data_len = len(packet_data) // 2   # 16-bit samples
    for i in range(0,data_len):
      db = packet_data[2*i:2*i+2]
      # print(db)
      data_buf.append(struct.unpack("<h",db)[0])
      # data_buf.append(packet_data[2*i] + packet_data[2*i] * 0x100)
    if buf_decimate == 100:
      plt.pause(.001)
      if outfile is not None:
        traces[dataCount] = data_buf
        dataCount += 1
        print("+")
        if dataCount == 20:
          print("Closing datafile")
          outfile.create_dataset(data=traces,name="traces")
          outfile.close()
          outfile = None
      buf_decimate = 0
      if notShown:
        notShown = False
    else:
      # print("+")
      buf_decimate += 1
