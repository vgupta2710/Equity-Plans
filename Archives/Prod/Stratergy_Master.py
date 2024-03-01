################################# Jai Shree Ganesh Ji ###################################
from audioop import avg
from itertools import dropwhile
from tkinter.tix import COLUMN
import pandas as pd
import sys
from datetime import date, datetime, timedelta
import time
import warnings
import yfinance as yf
import datetime
from scipy.stats import linregress
warnings.warn("deprecated", DeprecationWarning)
warnings.filterwarnings(action= 'ignore')
import numpy as np
import yaml
base_dir = os.path.dirname(__file__)
sys.path.insert(1,'../ShoonyaApi-Files/')
sys.path.insert(1,'../Input_Data/')
from api_helper import ShoonyaApiPy, get_time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

order_numb_list = []
stat_list = []
stock_ordered=[]
order_time = []

current_date = datetime.datetime.today().strftime('%Y_%m_%d')


def get_historicals(period,interval):
    stocks = pd.read_csv('../Input_Data/stocks.csv')
    def refresh_data(equity,period1,interval1):
        equity1 = yf.Ticker(equity)
        history = equity1.history(period=period1,interval=interval1,actions=False)   
        return history
    historical_consolidated = pd.DataFrame(columns=['candleid','Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'equity'])
    for x in stocks['instrumentName']:
        historical = refresh_data(x+'.NS',period,interval).reset_index()
        historical['equity']=x
        historical.reset_index(inplace=True)
        historical.columns = ['candleid','Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'equity']
        historical_consolidated = pd.concat([historical_consolidated,historical])
    return historical_consolidated , stocks

def api_connection():
    api = ShoonyaApiPy()
    with open('../ShoonyaApi-Files/cred.yml') as f:
        cred = yaml.load(f, Loader=yaml.FullLoader)
    ret = api.login(userid = cred['uid'], password = cred['pwd'], twoFA=cred['factor2'], vendor_code=cred['vc'], api_secret=cred['app_key'], imei=cred['imei'])
    return api

def buy_order(api,stck,buy_or_sell,buy_price,stop_loss,target_price,qty):
    order_details = api.place_order(buy_or_sell=buy_or_sell, product_type='B',
                exchange='NSE', tradingsymbol=str(stck+'-EQ'),quantity=qty, discloseqty=0,price_type='LMT', price=buy_price,trigger_price=None,retention='DAY', remarks='Testers', 
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

    _buy_down_price=0.0035
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
        #for y in layer1['day'].unique()[-1]: ##required if it needs to executed for multiple days
        y = (datetime.date.today()).day
        #y=2
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
                if (candle_body_prev > prev['High'] * pct_move) & (prev['Close'] > prev['Open']) and candle_body_prev > (candle_range_prev * _candle_range) and prev['RSI']>80 and (current['Close'] < current['Open']):
                    ###### Bearish trend post sudden move Section #####
                    trend_conf = whole_nbr(round(prev['Close'] * (1-_buy_down_price),2),np.ceil)
                    if  ((trend_conf > current['Close'])  & (current['Close'] < current['Open']) & ((abs(current['Close'] - current['Low'])  < (candle_range * 0.20))))  or ((candle_body< (candle_range * 0.20)) & (abs(current['Open'] - current['High']) > (candle_range * 0.60) ) & (current['Close'] < current['Open']) ): ## bearish strong candle for confirmation
                        stck.append(layer2['equity'].iloc[i])
                        date.append(layer2['Datetime'].iloc[i])
                        close.append(layer2['Open'].iloc[i])
                        buy_at = whole_nbr(latest['Open'] * (1-(0.0001)),np.floor)
                        buy_price.append(buy_at)
                        target.append(whole_nbr( buy_at *_target,np.ceil))
                        SL.append(whole_nbr(buy_at * _SL,np.ceil))
                        RSI.append(prev['RSI'])
                        if i != layer2.shape[0]:
                            max.append(round(layer2.iloc[i:].High.max(),2))
                            min.append(round(layer2.iloc[i:].Low.min(),2))
                            move.append('S')
                elif (candle_body_prev > prev['High'] * pct_move) & (prev['Close'] < prev['Open']) and (candle_body_prev > (candle_range_prev * _candle_range))  and prev['RSI']<20 and (current['Close'] > current['Open']) :
                    trend_conf= whole_nbr(prev['Close'] * (1+_buy_down_price),np.floor)                        
                    if (current['Close'] > trend_conf) & (current['Close'] > current['Open']) & ((abs(current['Close'] - current['High']) < (candle_range * 0.20))) or ((candle_body< (candle_range * 0.20)) & (abs(current['Close'] - current['High']) > (candle_range * 0.60)) & (current['Close'] > current['Open']) ) : ## Bearish strong candle for confirmation
                        stck.append(layer2['equity'].iloc[i])
                        date.append(layer2['Datetime'].iloc[i])
                        close.append(layer2['Open'].iloc[i])
                        buy_at = whole_nbr(latest['Open']  * (1+(0.0001)),np.floor)
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

def get_qty(order_amt,buy_amt):
    qty = np.ceil((order_amt/buy_amt)*4)
    return qty

api = api_connection()


for run in range(0,10):
    history,stocks= get_historicals('2D','15m')
    trade_frame = sudden_movement(_candle_range = 0.80,pct_move = 0.015,_target = 0.015,_SL = 0.0108,historical_consolidated = history)

    for x in trade_frame[~trade_frame.Stock.isin(stock_ordered)].index:
        quantity = get_qty(3000,trade_frame['buy price'].iloc[x])
        order_numb,stat  = buy_order(api,trade_frame['Stock'].iloc[x],trade_frame['move'].iloc[x],trade_frame['buy price'].iloc[x],trade_frame['SL'].iloc[x],trade_frame['target'].iloc[x],qty=quantity)
        order_numb_list.append(order_numb)
        stat_list.append(stat)
        stock_ordered.append(trade_frame['Stock'].iloc[x])
        order_time.append(datetime.datetime.now())
    pd.DataFrame({'Order_No:':order_numb_list,'Stat':stat_list,'Stck_Ordered':stock_ordered,'Order_time':order_time}).to_csv('Placed_Orders/orders_'+current_date+'.csv',index=False)    
    time.sleep(900)    

order_exec = pd.DataFrame({'Order_No:':order_numb_list,'Stat':stat_list,'Stck_Ordered':stock_ordered,'Order_time':order_time})


        equity1 = yf.Ticker('^NSEBANK')
        history = equity1.history(period='60D',interval='5m',actions=False) 







######## testing trend break out #######
history['Datetime'] = pd.to_datetime(history['Datetime']).dt.tz_localize(None)
history['Date'] = history['Datetime'].dt.strftime('%Y/%m/%d')
history['hour']=history['Datetime'].dt.hour
history['week']=history['Datetime'].dt.week
history['min']= history['Datetime'].dt.minute
history['day']= history['Datetime'].dt.day

daily_agg = history.groupby(['equity','day','Date'])['High','Open','Low'].agg({'High':max,'ow':min}).sort_values('Date',ascending=True).reset_index()

backup = backup1.copy()
backup1 = backup.copy()

equity_list = []
slmin_list = []
slmax_list = []
intercmin_list = []
intercmax_list = []
history = backup[backup['index']< 30 ] 
for eq in history['equity'].unique():
    train_set = history[(history['Datetime'] >(history['Datetime'].min() + timedelta(days=0))) & (history['equity'] == eq)].reset_index(drop=True)
    
    train_set['pivot'] = train_set.apply(lambda x: pivotid(train_set, x.name,2,2), axis=1)    
    train_set['pointpos'] = train_set.apply(lambda row: pointpos(row), axis=1)

    backcandles =train_set.shape[0] - (train_set[train_set['Low']==train_set['Low'].max()].index[0])
    candleid = train_set.shape[0] -1

    maxim = np.array([])
    minim = np.array([])
    xxmin = np.array([])
    xxmax = np.array([])
    
    for i in range(candleid-backcandles, candleid+1):
        if train_set.iloc[i].pivot == 1:
            minim = np.append(minim, train_set.iloc[i].Low)
            xxmin = np.append(xxmin, i) #could be i instead df.iloc[i].name
        if train_set.iloc[i].pivot == 2:
            maxim = np.append(maxim, train_set.iloc[i].High)
            xxmax = np.append(xxmax, i) # df.iloc[i].name
    try:
        slmin, intercmin, rmin, pmin, semin = linregress(xxmin, minim)
        slmax, intercmax, rmax, pmax, semax = linregress(xxmax, maxim)   
    except:
        continue

    equity_list.append(eq)
    slmin_list.append(slmin)
    slmax_list.append(slmax) 
    intercmin_list.append(intercmin)
    intercmax_list.append(intercmax)

eq_slope= pd.DataFrame({'Equity':equity_list,'Slope_Min':slmin_list,'Slope_Max':slmax_list,'Intercept_Min':intercmin_list,'Intercept_Max':intercmax_list })

backup = backup.merge(eq_slope, right_on ='Equity',left_on='equity')

backup['upper_point'] = (backup['index'] * backup['Slope_Max'] ) +  backup['Intercept_Max'] 
backup['lower_point'] = (backup['index'] * backup['Slope_Min'] ) + backup['Intercept_Min'] 



df =  backup[backup['equity']=='HDFC']
fig = go.Figure(data=[go.Candlestick(x=df['index'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
fig.add_trace(go.Scatter(x=df['index'], y=df['lower_point'], mode='lines', name='min slope'))
fig.add_trace(go.Scatter(x=df['index'], y=df['upper_point'], mode='lines', name='max slope'))
fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()

eq_slope

test_set = backup
test_set[(test_set['Open'] > test_set['lower_point']) & (test_set['Close'] < test_set['lower_point']) & (test_set['Slope_Min']<0.00)  & (test_set['Slope_Min']>-0.02) ].to_clipboard()


eqq = []
datetime = []
breakdown=[]
breakout =[]
for z in history['equity'].unique():
    test_set = history[history['equity']==z].reset_index(drop=True)
    validaiton_set = test_set[ (test_set['Datetime'] >(test_set['Datetime'].max() - timedelta(days=1)))].reset_index().merge(eq_slope,left_on='equity',right_on = 'Equity')    
    for x in range(2,len(validaiton_set)-1):
        if (validaiton_set['Close'][x-1] < (validaiton_set['index'][x]*validaiton_set['Slope_Min'][x])+validaiton_set['Intercept_Min'][x]) & (validaiton_set['Open'][x-1] > (validaiton_set['index'][x]*validaiton_set['Slope_Min'][x])+validaiton_set['Intercept_Min'][x]) & (validaiton_set['Open'][x] < validaiton_set['Close'][x-1]) & (validaiton_set['Volume'][x-1] > validaiton_set['Volume'].mean() * 1.90):
            eqq.append(z)
            datetime.append(validaiton_set['Datetime'][x])
            breakdown.append(1)
            breakout.append(0)
        if (validaiton_set['Close'][x-1] > (validaiton_set['index'][x]*validaiton_set['Slope_Max'][x])+validaiton_set['Intercept_Max'][x]) & (validaiton_set['Open'][x-1] < (validaiton_set['index'][x]*validaiton_set['Slope_Max'][x])+validaiton_set['Intercept_Max'][x]) & (validaiton_set['Open'][x] > validaiton_set['Close'][x-1]) & (validaiton_set['Volume'][x-1] > validaiton_set['Volume'].mean() * 1.90 ):
            eqq.append(z)
            datetime.append(validaiton_set['Datetime'][x])
            breakdown.append(0)
            breakout.append(1)

pd.DataFrame({'Equity':eqq,'Date':datetime, 'breakout':breakout,'breakdown':breakdown}).merge(eq_slope)

def pivotid(df1, l, n1, n2): #n1 n2 before and after candle l
    if l-n1 < 0 or l+n2 >= len(df1):
        return 0
    
    pividlow=1
    pividhigh=1
    for i in range(l-n1, l+n2+1):
        if(df1.Low[l]>df1.Low[i]):
            pividlow=0
        if(df1.High[l]<df1.High[i]):
            pividhigh=0
    if pividlow and pividhigh:
        return 3
    elif pividlow:
        return 1
    elif pividhigh:
        return 2
    else:
        return 0

def pointpos(x):
    if x['pivot']==1:
        return x['Low']-1e-3
    elif x['pivot']==2:
        return x['High']+1e-3
    else:
        return np.nan


train_set = history[(history['Datetime'] <(history['Datetime'].max() - timedelta(days=0))) & (history['equity'] == 'BSOFT')].reset_index(drop=True)

train_set['pivot'] = train_set.apply(lambda x: pivotid(train_set, x.name,5,5), axis=1)    
train_set['pointpos'] = train_set.apply(lambda row: pointpos(row), axis=1)

backcandles =train_set.shape[0] - (train_set[train_set['Low']==train_set['Low'].min()].index[0])
candleid = train_set.shape[0] -1

maxim = np.array([])
minim = np.array([])
xxmin = np.array([])
xxmax = np.array([])

for i in range(candleid-backcandles, candleid+1):
    if train_set.iloc[i].pivot == 1:
        minim = np.append(minim, train_set.iloc[i].Low)
        xxmin = np.append(xxmin, i) #could be i instead df.iloc[i].name
    if train_set.iloc[i].pivot == 2:
        maxim = np.append(maxim, train_set.iloc[i].High)
        xxmax = np.append(xxmax, i) # df.iloc[i].name

slmin, intercmin, rmin, pmin, semin = linregress(xxmin, minim)
slmax, intercmax, rmax, pmax, semax = linregress(xxmax, maxim)  
