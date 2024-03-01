import datetime
from operator import index
from turtle import right
import pandas as pd
import sys
from datetime import date
import datetime
from matplotlib import pyplot
import warnings
import yfinance as yf
from ks_api_client import ks_api
import boto3
from ks_api_client import ks_api
warnings.warn("deprecated", DeprecationWarning)
warnings.filterwarnings(action= 'ignore')
import numpy as np
from api_helper import ShoonyaApiPy, get_time
import yaml

start_time = datetime.datetime.now()

def get_historicals(period,interval):
    stocks = pd.read_csv('stocks.csv')
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

def stratergies(historical_consolidated,stocks,return_time):
    historical_consolidated['Datetime'] = pd.to_datetime(historical_consolidated['Datetime']).dt.tz_localize(None)
    historical_consolidated['hour']=historical_consolidated['Datetime'].dt.hour
    historical_consolidated['min']= historical_consolidated['Datetime'].dt.minute
    historical_consolidated['day']= historical_consolidated['Datetime'].dt.day

    historical_consolidated['Open_Max']= historical_consolidated[['Open','Close']].max(axis=1)
    historical_consolidated['Open_Min']= historical_consolidated[['Open','Close']].min(axis=1)

    eq_min_max= historical_consolidated[(historical_consolidated['hour']<10) & (historical_consolidated['min']<35)].groupby(['equity']).agg({'Open_Min':['min'],'Open_Max':['max']})
    pp_tf= historical_consolidated[(historical_consolidated['hour']<11) & (historical_consolidated['min']<35)].groupby(['equity']).agg({'Low':['min'],'High':['max'],'Close':['max']})
    pp_tf.columns = ['Low','High','Close']
    pp_tf['pp'] = (pp_tf['Close'] + pp_tf['High'] + pp_tf['Low'])/3
    pp_tf['R1'] = 2 * pp_tf['pp'] - pp_tf['Low']
    pp_tf['S1'] = 2 * pp_tf['pp'] - pp_tf['High']
    day_min_max= historical_consolidated.groupby(['equity']).agg({'Open':['min'],'Close':['max']})
    eq_min_max.columns=['Low_pp','High_pp']
    day_min_max.columns=['day_Low','day_High']
    historical_consolidated = historical_consolidated.merge(eq_min_max,on='equity')
    historical_consolidated = historical_consolidated.merge(pp_tf[['R1','S1']],on='equity')
    historical_consolidated = historical_consolidated.merge(day_min_max,on='equity')

    df = historical_consolidated[(historical_consolidated['hour']>10) & (historical_consolidated['hour']<15)]
    df['bb']=0
    df['day_breakout']=0
    df['day_breakdown']=0
    df['pivot_breakout']=0
    df['pivot_breakdown']=0

    for i in range(5,df.shape[0]): 
        current = df.iloc[i,:]
        prev = df.iloc[i-1,:]
        prev_2 = df.iloc[i-2,:]
        prev_3 = df.iloc[i-3,:]
        prev_4 = df.iloc[i-4,:]
        prev_5 = df.iloc[i-5,:]
        realbody = abs(current['Open'] - current['Close'])
        realbody_prev = abs(prev['Open'] - prev['Close'])
        realbody_prev2 = abs(prev_2['Open'] - prev_2['Close'])
        upper_stick = abs( current['High'] - current['Close'])
        lower_stick = abs(current['Open'] - current['Low'])
        candle_range = current['High'] - current['Low']
        candle_range_prev = abs(prev['High'] - prev['Low'])
        idx = df.index[i]
        if prev['Open']<prev['High_pp'] and prev['Close']>prev['High_pp'] and current['Open']>prev['Close'] and current['Close']>current['Open']  and current['bb']==0 and prev['Open'] < prev['Close'] and realbody_prev > candle_range_prev *0.80 and current['Volume']>prev['Volume'] and prev['Volume']>(prev_2['Volume']*1.50):
            df.loc[idx,'day_breakout'] =df.loc[idx,'Close']
            df.loc[df['equity']==current['equity'],'bb']=1
        
        elif prev['Open']>=prev['Low_pp'] and prev['Low_pp']>prev['Close'] and current['Open']<prev['Close'] and prev['Open']>prev['Close'] and current['Close']<current['Open'] and current['bb']==0 and current['Close']<current['Open'] and current['bb']==0 and realbody_prev > candle_range_prev *0.80 and current['Volume']>prev['Volume'] and prev['Volume']>(prev_2['Volume']*1.50):
            df.loc[idx,'day_breakdown'] =df.loc[idx,'Close']
            df.loc[df['equity']==current['equity'],'bb']=1
    
        """elif prev['Open']<prev['R1'] and prev['Close']>prev['R1'] and current['Open']>prev['Close'] and current['Close']>current['Open']  and current['bb']==0 and prev['Open'] < prev['Close'] and realbody_prev > candle_range_prev *0.80 and current['Volume']>prev['Volume'] and prev['Volume']>(prev_2['Volume']*1.50):
            df.loc[idx,'day_breakout'] =df.loc[idx,'Close']
            df.loc[df['equity']==current['equity'],'bb']=1  
    
        elif prev['Open']>=prev['S1'] and prev['Low_pp']>prev['S1'] and current['Open']<prev['Close'] and prev['Open']>prev['Close'] and current['Close']<current['Open'] and current['bb']==0 and current['Close']<current['Open'] and current['bb']==0 and realbody_prev > candle_range_prev *0.80 and current['Volume']>prev['Volume'] and prev['Volume']>(prev_2['Volume']*1.50):
            df.loc[idx,'day_breakout'] =df.loc[idx,'Close']
            df.loc[df['equity']==current['equity'],'bb']=1"""

    break_out_set= df[(df['day_breakout']>0) | (df['day_breakdown']>0) | (df['pivot_breakdown']>0) | (df['pivot_breakout']>0)]
    break_out_set['refresh_time'] = datetime.datetime.now()
    break_out_set['refresh_difference'] = (break_out_set['refresh_time'] - break_out_set['Datetime']).dt.total_seconds()
    break_out_set = break_out_set.merge(stocks, left_on=['equity'],right_on=['instrumentName'])
    break_out_set = break_out_set[break_out_set['refresh_difference'] < return_time]
    break_out_set = break_out_set[['InstrumentToken','instrumentName','Datetime','day_breakout','day_breakdown']]
    return break_out_set

def email(table):
    client = boto3.client('ses',region_name='us-west-2') 
    CHARSET = "UTF-8"
    table = break_out_set[break_out_set['refresh_difference']<3000].to_html(justify='center',render_links=True,escape=False,index=False)
    email(table)
    mail_sbj = """Test Table"""
    response = client.send_email(
            Destination={
                'ToAddresses': [
                    'vaigupta@egencia.com',
                ],
            'CcAddresses': [
                'vaigupta@egencia.com',
            ],
        },
        Message={
            'Body': {
                'Html': {
                    'Charset': CHARSET ,
                    'Data': table
                },
            },
            'Subject': {
                'Charset': CHARSET,
                'Data': mail_sbj,
            },
        },
        Source= 'vaigupta@egencia.com')

################ sudden movemonts capture ##############################


def sudden_movement(get_historicals):
    historical_consolidated,stocks= get_historicals('1D','5M')

    historical_consolidated['Datetime'] = pd.to_datetime(historical_consolidated['Datetime']).dt.tz_localize(None)
    historical_consolidated['hour']=historical_consolidated['Datetime'].dt.hour
    historical_consolidated['min']= historical_consolidated['Datetime'].dt.minute
    historical_consolidated['day']= historical_consolidated['Datetime'].dt.day

    pct_move = 0.020
    backtest_after = 16
    _candle_range = 0.80
    _target = 0.01
    _SL = 0.01
    _buy_down_price=0.0045

    stck = []
    date = []
    close = []
    buy_price=[]
    target = []
    SL = []
    max = []
    min = []
    move= []
    for x in historical_consolidated['equity'].unique():
        layer1 = historical_consolidated[historical_consolidated['equity']==x]
        for y in layer1['day'].unique():
            layer2 = layer1[layer1['day']==y]
            for i in range(0,layer2.shape[0]): 
                    current = layer2.iloc[i,:]
                    prev = layer2.iloc[i-1,:]
                    upper_stick = abs( prev['High'] - prev['Close'])
                    lower_stick = abs(prev['Open'] - prev['Low'])                
                    candle_body = abs(prev['Close'] - prev['Open'])
                    candle_range = abs(prev['High'] - prev['Low'])
                    if (candle_body > prev['High'] * pct_move) & (prev['Close'] > prev['Open']) & (current['Open'] < prev['Close']) and candle_body > (candle_range * _candle_range):
                        stck.append(layer2['equity'].iloc[i])
                        date.append(layer2['Datetime'].iloc[i])
                        close.append(layer2['Open'].iloc[i])
                        buy_price.append(round(prev['Close'] * (1-_buy_down_price),2))
                        target.append(round(prev['Close'] * (1-_buy_down_price) * (1-_target),2))
                        SL.append(round(prev['Close'] * (1-_buy_down_price) * (1+_SL),2))
                        if i != layer2.shape[0]:
                            max.append(round(layer2.iloc[i:].High.max(),2))
                            min.append(round(layer2.iloc[i:].Low.min(),2))
                            move.append('Down')
                    elif (candle_body > prev['High'] * pct_move) & (prev['Close'] < prev['Open']) & (current['Open']>prev['Close']) and candle_body > (candle_range * _candle_range):
                        stck.append(layer2['equity'].iloc[i])
                        date.append(layer2['Datetime'].iloc[i])
                        close.append(layer2['Open'].iloc[i])
                        buy_price.append(round(prev['Close'] * (1+_buy_down_price),2))
                        target.append(round(prev['Close'] * (1-_buy_down_price) * (1+_target),2))
                        SL.append(round(prev['Close'] * (1-_buy_down_price) * (1-_SL),2))                   
                        if i != layer2.shape[0]:
                            max.append(round(layer2.iloc[i:].High.max(),2))
                            min.append(round(layer2.iloc[i:].Low.min(),2)) 
                            move.append('Up')                       
    trade_frame = pd.DataFrame({'Stock':stck,'date':date,'close':close,'max':max,'min':min,'buy price':buy_price,'target':target,'SL':SL,'move':move})
    
    return trade_frame

trade_frame = sudden_movement(get_historicals)


trade_frame.to_csv('/Users/vaigupta/Downloads/backtest1.csv',index=False)

stck= 'AARTIDRUGS'
stop_loss = 10000
target_price= 11000
buy_price = '12022'
buy_or_sell= 'B'

order_details = api.place_order(buy_or_sell=buy_or_sell, product_type='B',
                exchange='NSE', tradingsymbol=stck+'-EQ',quantity=1, discloseqty=0,price_type='LMT', price=buy_price,trigger_price=None,retention='DAY', remarks='my_order_001', 
                bookloss_price = stop_loss, bookprofit_price = target_price )

type(buy_price)

order_details

order_details.get('norenordno')


api = ShoonyaApiPy()

#yaml for parameters
with open('cred.yml') as f:
    cred = yaml.load(f, Loader=yaml.FullLoader)
    print(cred)

ret = api.login(userid = cred['uid'], password = cred['pwd'], twoFA=cred['factor2'], vendor_code=cred['vc'], api_secret=cred['app_key'], imei=cred['imei'])



######moving average crossing above/below ####################
stck = []
date = []
close = []
buy_price=[]
target = []
SL = []
max = []
min = []
move= []
for x in historical_consolidated['equity'].unique():
    layer1 = historical_consolidated[historical_consolidated['equity']==x]
    for y in layer1['day'].unique():
        layer2 = layer1[layer1['day']==y]
        layer2['SMA_9']=layer2['Close'].rolling(9).mean()
        layer2['SMA_18']= layer2['Close'].rolling(18).mean()        
        for i in range(0,layer2.shape[0]): 
                current = layer2.iloc[i,:]
                prev = layer2.iloc[i-1,:]
                upper_stick = abs( prev['High'] - prev['Close'])
                lower_stick = abs(prev['Open'] - prev['Low'])                
                candle_body = abs(prev['Close'] - prev['Open'])
                candle_range = abs(prev['High'] - prev['Low'])
                if (current['SMA_18']>current['SMA_9']) & (prev['SMA_18']<prev['SMA_9'])  & (current['Open']>current['Close']) & (candle_body > candle_range *0.90)  :
                    stck.append(layer2['equity'].iloc[i])
                    date.append(layer2['Datetime'].iloc[i])
                    close.append(layer2['Open'].iloc[i])


pd.DataFrame({'Stock':stck,'date':date,'close':close}).to_csv('moving_average_drop.csv')              
