#!/usr/bin/env python
# coding: utf-8

import gammu


sm = gammu.StateMachine()
sm.ReadConfig()
sm.Init()

message = {
    'Text': 'Mission successful: SMS sent',
    'SMSC': {'Location': 1},
    'Number': '8163449956'
}

sm.SendSMS(message)