from api_helper import ShoonyaApiPy, get_time
import datetime
import logging
import time
import yaml
import pandas as pd
import warnings
warnings.filterwarnings(action= 'ignore')
base_dir = os.path.dirname(__file__)
sys.path.insert(1,'../ShoonyaApi-Files/')
sys.path.insert(1,'../StreamingData/')
import traceback
#sample
logging.basicConfig(level=logging.DEBUG)

#start of our program
api = ShoonyaApiPy()
tokens = pd.read_csv('..//N500.csv',usecols=['Insert_Token'])

#yaml for parameters
with open('cred.yml') as f:
    cred = yaml.load(f, Loader=yaml.FullLoader)
    print(cred)

ret = api.login(userid = cred['uid'], password = cred['pwd'], twoFA=cred['factor2'], vendor_code=cred['vc'], api_secret=cred['app_key'], imei=cred['imei'])

#flag to tell us if the websocket is open
socket_opened = False
global current_batch,ldr_error
current_batch = []
MAX_BATCH_SIZE = 1000

#application callbacks
def event_handler_order_update(message):
    print("order event: " + str(message))

def event_handler_quote_update(event):
    #print("quote event: {0}".format(time.strftime('%d-%m-%Y %H:%M:%S')) + str(message))
    order_append(event)
    #data = (event['tk'],event['ts'],event['ft'],event['o'],event['l'],event['h'],event['c'],event['v'])
    #current_batch.append(data)  
    
def order_append(event):
    try:
        data = (event['tk'],event['ts'],event['ft'],event['o'],event['l'],event['h'],event['c'],event['v'])
        current_batch.append(data)
        check_current_batch(current_batch)
    except:
        ldr_error = traceback.format_exc()
        print(ldr_error) 
        pass

def check_current_batch(current_batch):
    try:
        if len(current_batch) > MAX_BATCH_SIZE:
            pd.DataFrame(current_batch).to_csv('streaming_data.csv',mode='a',index=False,header=False)
            current_batch = []     
    except:
        ldr_error = traceback.format_exc()
        print(ldr_error) 
        pass

def open_callback():
    global socket_opened
    socket_opened = True
    print('app is connected')
    #api.subscribe('NSE|13')
    api.subscribe(tokens['Insert_Token'].tolist())

def get_time(time_string):
    data = time.strptime(time_string,'%d-%m-%Y %H:%M:%S')
    return time.mktime(data)


api.start_websocket(order_update_callback=event_handler_order_update, 
                    subscribe_callback=event_handler_quote_update, 
                    socket_open_callback=open_callback)

api.close_websocket()


pd.DataFrame(current_batch)

ldr_error

os.getcwd()

os.chdir('/Users/vaigupta/Documents/DS/Equity_Simulator/')