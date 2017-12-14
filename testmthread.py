#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:05:16 2017

@author: ince
"""

import threading, time
from multiprocessing import Process
 
## Example usage
def someOtherFunc():
    time.sleep(.5)
    print ("someOtherFunc")
    
    

## Example usage
def someOther():
    time.sleep(.1)
    print ("someOther")

# Example usage
def someFun():
    print ("someFun")

# 
#t1 = threading.Thread(target=someOtherFunc)
#t1.start()
#
#t2 = threading.Thread(target=someFun)
#t2.start()
#
#
#t1.join()
#t2.join()



def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()
    
runInParallel(someOther, someOtherFunc, someFun) #call alla funzione che fa partire i processi in parallelo
