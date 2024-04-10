#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

xs = []
ys = []
zs = []

with open(sys.argv[1]) as f:
  data = [line.rstrip() for line in f.readlines()]
  for datal in data:
    fx,fy,fz = [x for x in datal.split(",")]
    xs.append(float(fx))
    ys.append(float(fy))
    zs.append(float(fz))

plt.plot(xs)
plt.plot(ys)
plt.plot(zs)
plt.show()
