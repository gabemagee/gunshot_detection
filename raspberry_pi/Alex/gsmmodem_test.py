#!/usr/bin/env python
# coding: utf-8

from gsmmodem.modem import GsmModem


# Configuring the modem connection
modem_port = '/dev/ttyUSB0'
modem_baudrate = 115200
modem_sim_pin = None  # SIM card PIN (if any)
    
# Establishing a connection to the SMS modem
modem = GsmModem(modem_port, modem_baudrate)
modem.smsTextMode = False
modem.connect(modem_sim_pin)

# Waiting for a signal and then dispatching an SMS text
modem.waitForNetworkCoverage(timeout = 60)
modem.sendSms(8163449956, "Mission Successful: SMS was sent.")
