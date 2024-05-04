#!/usr/bin/python3

import sys
import numpy as np
import redpitaya_scpi as scpi
import matplotlib.pyplot as plt
import time

# & is needed because computers are dumb
# systemctl start redpitaya_scpi &

rp_s = scpi.scpi(sys.argv[1])

def trigger_acq():
  global rp_s
  print("Setting trigger")
  rp_s.tx_txt('ACQ:RST')
  rp_s.tx_txt('ACQ:START')
  rp_s.tx_txt('ACQ:DEC 1')
  rp_s.tx_txt('ACQ:SOUR1:GAIN LV')
  rp_s.tx_txt('ACQ:SOUR2:GAIN LV')
  rp_s.tx_txt('ACQ:TRIG:LEV 10 mV')
  rp_s.tx_txt('ACQ:TRIG:LEV?')
  data = rp_s.rx_txt()
  rp_s.tx_txt('ACQ:TRIG CH1_PE')
  return

trigger_acq()

while True:
  try:
    time.sleep(0.1)
    rp_s.tx_txt('ACQ:TRIG:STAT?')
    data = rp_s.rx_txt()
    if data == 'TD':
      rp_s.tx_txt('ACQ:SOUR1:DATA?')
      data = rp_s.rx_txt()
      data = data.strip('{}\n\r')
      trace = np.fromstring(data, dtype=float, sep=',')
      plt.plot(trace)
      print("TRIGGERED")
      print(len(trace))
      trigger_acq()
  except KeyboardInterrupt:
    rp_s.tx_txt('ACQ:START')
    plt.show()
    sys.exit(0)

