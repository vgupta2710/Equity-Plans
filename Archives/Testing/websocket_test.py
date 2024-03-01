#!/usr/bin/env python
# coding: utf-8

# In[3]:

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('/Users/vaigupta/Documents/DS/Equity_Simulator/Production Copy/ShoonyaApi-py'))))
from api_helper import ShoonyaApiPy, get_time
import datetime
import logging
import time
import yaml
import pandas as pd
import warnings
warnings.filterwarnings(action= 'ignore')
#sample
logging.basicConfig(level=logging.DEBUG)


# In[4]:



#start of our program
api = ShoonyaApiPy()

#yaml for parameters
with open('cred.yml') as f:
    cred = yaml.load(f, Loader=yaml.FullLoader)
    print(cred)

ret = api.login(userid = cred['uid'], password = cred['pwd'], twoFA=cred['factor2'], vendor_code=cred['vc'], api_secret=cred['app_key'], imei=cred['imei'])


# In[5]:


#flag to tell us if the websocket is open
socket_opened = False

#application callbacks
def event_handler_order_update(message):
    print("order event: " + str(message))


SYMBOLDICT = {}
def event_handler_quote_update(message):
    global SYMBOLDICT
    #e   Exchange
    #tk  Token
    #lp  LTP
    #pc  Percentage change
    #v   volume
    #o   Open price
    #h   High price
    #l   Low price
    #c   Close price
    #ap  Average trade price

    print("quote event: {0}".format(time.strftime('%d-%m-%Y %H:%M:%S')) + str(message))
    global strr
    strr= message
    
    key = message['e'] + '|' + message['tk']

    if key in SYMBOLDICT:
        symbol_info =  SYMBOLDICT[key]
        symbol_info.update(message)
        SYMBOLDICT[key] = symbol_info
    else:
        SYMBOLDICT[key] = message

    #print(SYMBOLDICT[key])

def open_callback():
    global socket_opened
    socket_opened = True
    print('app is connected')
    
    api.subscribe('NSE|22', feed_type='d')
    #api.subscribe(['NSE|22', 'BSE|522032'])

#end of callbacks

def get_time(time_string):
    data = time.strptime(time_string,'%d-%m-%Y %H:%M:%S')

    return time.mktime(data)

if ret != None:   
    ret = api.start_websocket(order_update_callback=event_handler_order_update, subscribe_callback=event_handler_quote_update, socket_open_callback=open_callback)
    
    while True:
        if socket_opened == True:
            print('q => quit')
            
            print('r => ')
            prompt1=input('what shall we do? ').lower()    
            if prompt1 == ' s':
                print('closing websocket')
                api.close_websocket()
                continue
            if prompt1 == 'r':
                print('closing websocket')
                api.close_websocket()
                sleep(1)
                api.start_websocket(order_update_callback=event_handler_order_update, subscribe_callback=event_handler_quote_update, socket_open_callback=open_callback)            
                continue
            else:
                print('Fin') #an answer that wouldn't be yes or no
                break   

        else:
            continue


# In[61]:


api.close_websocket()


# In[6]:


api.start_websocket(order_update_callback=event_handler_order_update, subscribe_callback=event_handler_quote_update, socket_open_callback=open_callback)


# In[12]:


strr


# In[52]:


time = datetime.fromtimestamp(int(strr['ft'])).strftime('%Y-%m-%d %H:%M:%S')


# In[70]:


json.loads(({'Time':time,'Open':strr['o']}))


# In[8]:


strr


# In[108]:


s = ({'Time':time,'Open':strr['o']})
v = ({'Time':time,'Open':strr['o']})


# In[109]:


token = []
datetime = []
stock = []
low = []
high = []
close = []
open = []
volume = []


# In[110]:


v


# In[78]:


import python_version


# In[79]:


from platform import python_version

print(python_version())


# In[ ]:




