################################# Jai Shree Ganesh Ji ###################################
from operator import index
from turtle import right
import pandas as pd
import sys
from datetime import date
import datetime
import warnings
import yfinance as yf
import boto3
warnings.warn("deprecated", DeprecationWarning)
warnings.filterwarnings(action= 'ignore')
import numpy as np
import yaml
base_dir = os.path.dirname(__file__)
sys.path.insert(1,'../ShoonyaApi-Files/')
sys.path.insert(1,'../Input_Data/')
sys.path.insert(1,'/Users/vaigupta/Downloads/TA-Lib-0.4.24/talib')
sys.path.insert(1,'/Users/vaigupta/Downloads/TA-Lib-0.4.24')
from api_helper import ShoonyaApiPy, get_time
import talib

order_numb_list = []
stat_list = []
stock_ordered=[]

def get_historicals(period,interval):
    stocks = pd.read_csv('../Input_Data/stocks.csv')
    def refresh_data(equity,period1,interval1):
        equity1 = yf.Ticker(equity)
        history = equity1.history(period=period1,interval=interval1,actions=False)   
        return history
    historical_consolidated = pd.DataFrame(columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'equity'])
    for x in stocks['instrumentName']:
        historical = refresh_data(x+'.NS',period,interval).reset_index()
        historical['equity']=x
        historical_consolidated = historical_consolidated.append(historical,ignore_index=True)
    return historical_consolidated , stocks

def api_connection():
    api = ShoonyaApiPy()
    with open('../ShoonyaApi-Files/cred.yml') as f:
        cred = yaml.load(f, Loader=yaml.FullLoader)
    ret = api.login(userid = cred['uid'], password = cred['pwd'], twoFA=cred['factor2'], vendor_code=cred['vc'], api_secret=cred['app_key'], imei=cred['imei'])
    return api

def buy_order(api,stck,buy_or_sell,buy_price,stop_loss,target_price,qty):
    order_details = api.place_order(buy_or_sell=buy_or_sell, product_type='B',
                exchange='NSE', tradingsymbol=str(stck+'-EQ'),quantity=qty, discloseqty=0,price_type='LMT', price=buy_price,trigger_price=None,retention='DAY', remarks='my_order_001', 
                bookloss_price = stop_loss, bookprofit_price = target_price)
    return order_details.get('norenordno') ,order_details.get('stat')  

def RSI_Calc(df_input):
    delta = df_input['Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up/ema_down
    RSI = 100 - (100 / (1 + rs))
    df_output= pd.concat([df_input,RSI.rename('RSI')],axis=1 )
    return df_output

def whole_nbr(input,func):
    output = (func((input*100)/5)*5)/100
    return output    

def sudden_movement(_candle_range,pct_move,_target,_SL,historical_consolidated):
    historical_consolidated['Datetime'] = pd.to_datetime(historical_consolidated['Datetime']).dt.tz_localize(None)
    historical_consolidated['hour']=historical_consolidated['Datetime'].dt.hour
    historical_consolidated['min']= historical_consolidated['Datetime'].dt.minute
    historical_consolidated['day']= historical_consolidated['Datetime'].dt.day

    backtest_after = 16
    _candle_range = 0.80
    _buy_down_price=0.0020

    stck = []
    date = []
    close = []
    buy_price=[]
    target = []
    SL = []
    max = []
    min = []
    move= []
    RSI = []
    for x in historical_consolidated['equity'].unique():
        layer1 = historical_consolidated[historical_consolidated['equity']==x]
        layer1 = RSI_Calc(layer1)
        for y in layer1['day'].unique():
            layer2 = layer1[layer1['day']==y]
            for i in range(0+2,layer2.shape[0]): 
                    current = layer2.iloc[i-1,:]
                    latest = layer2.iloc[i,:]
                    prev = layer2.iloc[i-2,:]
                    upper_stick = abs( current['High'] - current['Close'])
                    lower_stick = abs(current['Open'] - current['Low'])                
                    candle_body_prev = abs(prev['Close'] - prev['Open'])
                    candle_range_prev = abs(prev['High'] - prev['Low'])
                    candle_body = abs(current['Close'] - current['Open'])
                    candle_range = abs(current['High'] - current['Low'])
                    if (candle_body_prev > prev['High'] * pct_move) & (prev['Close'] > prev['Open']) & (current['Open'] < prev['Close']) and candle_body_prev > (candle_range_prev * _candle_range) and prev['RSI']>70 :
                        ###### Bearish trend post sudden move Section #####
                        buy_at = whole_nbr(round(prev['Close'] * (1-_buy_down_price),2),np.ceil)
                        if  ((buy_at > current['Close']) &  (candle_body > (candle_range * 0.50)) & (current['Close'] < current['Open']) & (abs(current['Close'] - current['Low'])  < (candle_range * 0.20)))  or ((candle_body< (candle_range * 0.20)) & (abs(current['Open'] - current['High']) > (candle_range * 0.60) )): ## bearish strong candle for confirmation
                            stck.append(layer2['equity'].iloc[i])
                            date.append(layer2['Datetime'].iloc[i])
                            close.append(layer2['Open'].iloc[i])
                            buy_price.append(buy_at)
                            target.append(whole_nbr( buy_at *_target,np.ceil))
                            SL.append(whole_nbr(buy_at * _SL,np.ceil))
                            RSI.append(prev['RSI'])
                            if i != layer2.shape[0]:
                                max.append(round(layer2.iloc[i:].High.max(),2))
                                min.append(round(layer2.iloc[i:].Low.min(),2))
                                move.append('S')
                    elif (candle_body_prev > prev['High'] * pct_move) & (prev['Close'] < prev['Open']) & (current['Open']>prev['Close']) and candle_body_prev > (candle_range_prev * _candle_range)  and prev['RSI']<30:
                        buy_at = whole_nbr(prev['Close'] * (1+_buy_down_price),np.floor)                        
                        if (current['Close'] > buy_at) and (candle_body > (candle_range * 0.50)) & (current['Close'] > current['Open']) & (upper_stick < (candle_range * 0.20)) or ((candle_body< (candle_range * 0.20)) & (abs(current['Close'] - current['Low']) > (candle_range * 0.60) )) : ## Bearish strong candle for confirmation
                            stck.append(layer2['equity'].iloc[i])
                            date.append(layer2['Datetime'].iloc[i])
                            close.append(layer2['Open'].iloc[i])
                            buy_at = whole_nbr(prev['Close'] * (1+_buy_down_price),np.floor)
                            buy_price.append(buy_at)
                            target.append(whole_nbr( buy_at *_target,np.floor))
                            SL.append(whole_nbr(buy_at * _SL,np.floor))
                            RSI.append(prev['RSI'])               
                            if i != layer2.shape[0]:
                                max.append(round(layer2.iloc[i:].High.max(),2))
                                min.append(round(layer2.iloc[i:].Low.min(),2)) 
                                move.append('B')                       
    trade_frame = pd.DataFrame({'Stock':stck,'date':date,'close':close,'max':max,'min':min,'buy price':buy_price,'target':target,'SL':SL,'move':move,'RSI':RSI}).sort_values('date').reset_index(drop=True)
    return trade_frame

api = api_connection()

history,stocks= get_historicals('2D','5m')
trade_frame = sudden_movement(_candle_range = 0.80,pct_move = 0.025,_target = 0.01,_SL = 0.01,historical_consolidated = history)

for x in trade_frame[~trade_frame.Stock.isin(stock_ordered)].index:
    order_numb,stat  = buy_order(api,trade_frame['Stock'].iloc[x],trade_frame['move'].iloc[x],trade_frame['buy price'].iloc[x],trade_frame['SL'].iloc[x],trade_frame['target'].iloc[x],qty=10)
    order_numb_list.append(order_numb)
    stat_list.append(stat)
    stock_ordered.append(trade_frame['Stock'].iloc[x])

pd.DataFrame({'Order_No:':order_numb_list,'Stat':stat_list,'Stck_Ordered':stock_ordered})



### testing to be delelted later on

history = historical_consolidated[historical_consolidated['equity'] =='M&M']

history = historical_consolidated.copy()
trade_frame = sudden_movement(_candle_range = 0.80,pct_move = 0.015,_target = 0.015,_SL = 0.01,historical_consolidated = history)

trade_frame

trade_frame.to_csv('/Users/vaigupta/Downloads/equity_backtest_14_08_22.csv')

historical_consolidated,stocks= get_historicals('15D','5m')



datetime.date.today()