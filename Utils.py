#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 20:03:26 2017

@author: suresh
"""

import shelve

def saveEnvironment(fileName,globals_=None):
    if globals_ is None:
        globals_ = globals()
    my_shelf = shelve.open(fileName, 'n')
    for key, value in globals_.items():
        if not key.startswith('__'):
            try:
                my_shelf[key] = value
            except Exception:
                print('ERROR shelving: "%s"' % key)
            else:
                print('shelved: "%s"' % key)
    my_shelf.close()
