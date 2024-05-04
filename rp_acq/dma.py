#!/usr/bin/env python3

import time
import matplotlib.pyplot as plt
import numpy as np
import redpitaya_scpi as scpi
import sys

# dma read.
rp_s = scpi.scpi(sys.argv[1])
rp_s.tx_txt('ACQ:RST')
start_address = int(rp_s.txrx_txt('ACQ:AXI:START?'))
# size = int(rp_s.txrx_txt('ACQ:AXI:SIZE?'))
# print(size)

rp_s.tx_txt("ACQ:AXI:SOUR1:Trig:Dly 32000")
rp_s.tx_txt(f"ACQ:AXI:SOUR1:SET:Buffer {start_address},{32000*4}")

rp_s.tx_txt("ACQ:AXI:DEC 1")
rp_s.tx_txt('ACQ:AXI:SOUR1:ENable ON')
rp_s.tx_txt("ACQ:TRig:LEV 10 mV")
print("starting!")

rp_s.tx_txt('ACQ:START')
rp_s.tx_txt('ACQ:TRig CH1_PE')

while 1:
    rp_s.tx_txt("ACQ:TRig:STAT?")
    if rp_s.rx_txt() == 'TD':
        print("Triggered")
        time.sleep(0.1)
        break

while 1:
    rp_s.tx_txt('ACQ:AXI:SOUR1:TRIG:FILL?')
    if rp_s.rx_txt() == '1':
        print('DMA buffer full\n')
        break

rp_s.tx_txt('ACQ:STOP')
posChA = int(rp_s.txrx_txt('ACQ:AXI:SOUR1:Trig:Pos?'))
print("Receiving data")
rp_s.tx_txt(f"ACQ:AXI:SOUR1:DATA:Start:N? {posChA},32000")
signal_str = rp_s.rx_txt()

buff1 = list(map(float, signal_str.strip('{}\n\r').replace("  ", "").split(',')))

rp_s.tx_txt('ACQ:AXI:SOUR1:ENable OFF')
rp_s.close()
plt.plot(buff1)
plt.show()
