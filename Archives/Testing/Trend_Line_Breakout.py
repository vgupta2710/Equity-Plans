from datetime import date, datetime, timedelta
import pandas as pd
import yfinance as yf
import numpy as np
from matplotlib import pyplot
from scipy.stats import linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

def get_historicals(period,interval):
    stocks = pd.read_csv('../Input_Data/stocks.csv')
    stocks = stocks[stocks['instrumentName']=='ADANIGREEN']
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

def candle_stick_patterns(histoical):
    consolidated = pd.DataFrame(columns = ['candleid', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
    'equity', 'Date', 'hour', 'week', 'min', 'day', 'Equity', 'Slope_Min',
    'Slope_Max', 'Intercept_Min', 'Intercept_Max', 'P_min', 'P_max'
    'upper_point', 'lower_point', 'Bullish swing', 'Bearish swing',
    'Bullish pinbar', 'Bearish pinbar', 'Inside bar', 'Outside bar',
    'Bullish engulfing', 'Bearish engulfing'])
    for e in histoical['equity'].unique():
        df = histoical[(histoical['equity']==e)].iloc[-4000:]
        for i in range(2,df.shape[0]):
            current = df.iloc[i,:]
            prev = df.iloc[i-1,:]
            prev_2 = df.iloc[i-2,:]
            realbody = abs(current['Open'] - current['Close'])
            candle_range = current['High'] - current['Low']
            idx = df.index[i]
            
            # Bullish swing
            df.loc[idx,'Bullish swing'] = current['Low'] > prev['Low'] and prev['Low'] < prev_2['Low']
            # Bearish swing
            df.loc[idx,'Bearish swing'] = current['High'] < prev['High'] and prev['High'] > prev_2['High']
            # Bullish pinbar
            df.loc[idx,'Bullish pinbar'] = realbody <= candle_range/3 and  min(current['Open'], current['Close']) > (current['High'] + current['Low'])/2 and current['Low'] < prev['Low']
            # Bearish pinbar
            df.loc[idx,'Bearish pinbar'] = realbody <= candle_range/3 and max(current['Open'] , current['Close']) < (current['High'] + current['Low'])/2 and current['High'] > prev['High']
                
            # Inside bar
            df.loc[idx,'Inside bar'] = current['High'] < prev['High'] and current['Low'] > prev['Low']
                
            # Outside bar
            df.loc[idx,'Outside bar'] = current['High'] > prev['High'] and current['Low'] < prev['Low']
            
            # Bullish engulfing
            df.loc[idx,'Bullish engulfing'] = current['High'] > prev['High'] and current['Low'] < prev['Low'] and realbody >= 0.8 * candle_range and current['Close'] > current['Open']
            # Bearish engulfing
            df.loc[idx,'Bearish engulfing'] = current['High'] > prev['High'] and current['Low'] < prev['Low'] and realbody >= 0.8 * candle_range and current['Close'] < current['Open']
        consolidated = pd.concat([consolidated,df])
    return consolidated


historical,stocks = get_historicals('5d','15m')

historical['Datetime'] = pd.to_datetime(historical['Datetime']).dt.tz_localize(None)
historical['Date'] = historical['Datetime'].dt.strftime('%Y/%m/%d')
historical['hour']=historical['Datetime'].dt.hour
historical['week']=historical['Datetime'].dt.week
historical['min']= historical['Datetime'].dt.minute
historical['day']= historical['Datetime'].dt.day

equity_list = []
slmin_list = []
slmax_list = []
intercmin_list = []
intercmax_list = []
p_min = []
p_max = []
min_points = []
max_points = []
se_max = []
se_min = []
volume=[]



####################################
def support(df1, l, n1, n2): #n1 n2 before and after candle l
    if l-n1 < 0 or l+n2 >= len(df1):
        return 0
    for i in range(l-n1+1, l+1):
        if(df1.Low[i]>df1.Low[i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df1.Low[i]<df1.Low[i-1]):
            return 0
    return df1.Low[i]

#support(df,46,3,2)
def resistance(df1, l, n1, n2): #n1 n2 before and after candle l
    for i in range(l-n1+1, l+1):
        if(df1.high[i]<df1.High[i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df1.high[i]>df1.high[i-1]):
            return 0
    return 1
#resistance(df, 30, 3, 5)

eq = 'ADANIGREEN'
################################




history = historical.copy()

history = historical[(historical['hour']<13) & (historical['Date'] == '2022/09/30' )]
master = historical[(historical['Date'] == '2022/09/30' )] 


for eq in history['equity'].unique():
    train_set = history[(history['Datetime'] >(history['Datetime'].min() + timedelta(days=0))) & (history['equity'] == eq)].reset_index(drop=True)
    
    train_set['pivot'] = train_set.apply(lambda x: pivotid(train_set, x.name,3,3), axis=1)  
    
    train_set['sr'] = train_set.apply(lambda x: support(train_set, x.name,15,15), axis=1)  

    train_set['pointpos'] = train_set.apply(lambda row: pointpos(row), axis=1)

    backcandles =train_set.shape[0] - (train_set[train_set['High']==train_set['High'].max()].index[0])
    candleid = train_set.shape[0] -1

    maxim = np.array([])
    minim = np.array([])
    xxmin = np.array([])
    xxmax = np.array([])
    
    for i in range(0, candleid+1):
        if train_set.iloc[i].pivot == 1:
            minim = np.append(minim, train_set.iloc[i].Low)
            xxmin = np.append(xxmin, i) 
        if train_set.iloc[i].pivot == 2:
            maxim = np.append(maxim, train_set.iloc[i].High)
            xxmax = np.append(xxmax, i) 
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
    p_min.append(pmin)
    p_max.append(pmax)
    max_points.append(len(maxim))
    min_points.append(len(minim))
    se_max.append(semax)
    se_max.append(semin)
    volume.append(train_set.Volume.max())




eq_slope= pd.DataFrame({'Equity':equity_list,'Slope_Min':slmin_list,'Slope_Max':slmax_list,'Intercept_Min':intercmin_list,'Intercept_Max':intercmax_list,'P_min': p_min,'P_max': p_max,'Points_min': min_points,'Points_max': max_points, 'Max_Vol':volume})

historical,stocks = get_historicals('1d','1m')
historical['Datetime'] = pd.to_datetime(historical['Datetime']).dt.tz_localize(None)
historical['Date'] = historical['Datetime'].dt.strftime('%Y/%m/%d')
historical['hour']=historical['Datetime'].dt.hour
historical['week']=historical['Datetime'].dt.week
historical['min']= historical['Datetime'].dt.minute
historical['day']= historical['Datetime'].dt.day
master = historical[(historical['Date'] == '2022/09/30') & ((historical['hour']>12))]

master = master.merge(eq_slope,left_on = 'equity' ,right_on = 'Equity')
master['upper_point'] = (master['candleid'] * master['Slope_Max'] ) +  master['Intercept_Max'] 
master['lower_point'] = (master['candleid'] * master['Slope_Min'] ) + master['Intercept_Min'] 

#master = candle_stick_patterns(master)

master[(master['Close']< master['lower_point']) & (master['Bearish engulfing'] == True)
& (master['Slope_Min']>0) & (master['Slope_Max']< master['Slope_Min'])
& (master['P_min']< 0.5)
& (master['Volume'] )]

master[((master['Close']*1.006)< (master['lower_point'])) 
& (master['Slope_Min']< -0.10) & (master['Slope_Max']< (master['Slope_Min'])) ] 

master[((master['Close']*1.006)< (master['lower_point'])) 
& (master['Slope_Min'] > 0.10) & (master['Slope_Max']> (master['Slope_Min']))] 
#& (master['P_min']< 0.5)]


train_set = candle_stick_patterns(master[master['equity']=='ADANIGREEN'])
train_set['breakout'] = ''
train_set['breakdown'] = ''
train_set.loc[((train_set['Bearish engulfing']== True) | (train_set['Bearish swing']== True )),'breakdown'] = train_set[(train_set['Bearish engulfing']== True) ]['Close'] - (1e-3)
train_set.loc[((train_set['Bullish engulfing']== True) ),'breakout'] = train_set[(train_set['Bullish engulfing']== True) ]['Close']+(1e-3)
fig = go.Figure(data=[go.Candlestick(x=train_set['candleid'],
                open=train_set['Open'],
                high=train_set['High'],
                low=train_set['Low'],
                close=train_set['Close'])])
fig.add_trace(go.Scatter(x=train_set['candleid'], y=train_set['lower_point'], mode='lines', name='min slope'))
fig.add_trace(go.Scatter(x=train_set['candleid'], y=train_set['upper_point'], mode='lines', name='max slope'))
fig.add_trace(go.Scatter(x=train_set['candleid'], y=train_set['breakout'], mode='markers', name='breakout', marker=dict(size=9)))
fig.add_trace(go.Scatter(x=train_set['candleid'], y=train_set['breakdown'], mode='markers', name='breakdown', marker=dict(size=9)))
for x in train_set['sr'].unique():
    if x > 0:
        fig.add_hline(y=x)
fig.show()



eq_slope[eq_slope['Equity']=='ADANITRANS']
################################################################################################
######## master for the last day to validate the slope break out ######

master_copy = master[master['hour']>12]

master_copy = master_copy.merge(eq_slope,left_on = 'equity' ,right_on = 'Equity')
master_copy['upper_point'] = (master_copy['candleid'] * master_copy['Slope_Max'] ) +  master_copy['Intercept_Max'] 
master_copy['lower_point'] = (master_copy['candleid'] * master_copy['Slope_Min'] ) + master_copy['Intercept_Min'] 
master_copy = candle_stick_patterns(master_copy)

master_copy['breakout'] = ''
master_copy['breakdown'] = ''
for x in master_copy.Equity.unique():
    test = master_copy[master_copy['equity']=='x'].reset_index(drop=True)
    v= 0
    breakdown = 0
    breakout = 0
    for y in range(1,len(test)):
        if ((v<1) & (breakdown==0) &  (test['Open'].iloc[y] < test['lower_point'].iloc[y])  & (test['Close'].iloc[y] > test['upper_point'].iloc[y]) & (test['upper_point'].iloc[y] > test['lower_point'].iloc[y]) & (test['Volume'].iloc[y] >  (test['Max_Vol'].mean()*7))  )  :
            breakdown = 2
            v=2              
        if ((v <1) & (breakout==0) & (test['Open'].iloc[y] < test['upper_point'].iloc[y]) & (test['upper_point'].iloc[y]>test['lower_point'].iloc[y]) & (test['Close'].iloc[y] > test['upper_point'].iloc[y]) & (test['Volume'].iloc[y] >  (test['Max_Vol'].mean()*7)) & (abs(test['Close'].iloc[y]  - test['Open'].iloc[y] ) > abs(test['High'].iloc[y]  - test['Low'].iloc[y] )*0.80)  & (test['Slope_Max'].iloc[y] < 0) ):
            breakout = 2
            v=2  
        if ((v <1) & (breakdown==0) & (test['Open'].iloc[y] > test['lower_point'].iloc[y]) & (test['upper_point'].iloc[y]>test['lower_point'].iloc[y]) & (test['Close'].iloc[y] < test['lower_point'].iloc[y]) & (test['Volume'].iloc[y] >  (test['Max_Vol'].mean()*7)) & (abs(test['Close'].iloc[y]  - test['Open'].iloc[y] ) > abs(test['High'].iloc[y]  - test['Low'].iloc[y] )*0.80)  & (test['Slope_Min'].iloc[y] > 0) ):
            breakdown = 2
            v=2
        if ((v == 2) & (breakdown ==2) & ((test['Bearish pinbar'].iloc[y]== False) | (test['Bearish engulfing'].iloc[y]== True) | (test['Bearish swing'].iloc[y]== False))):
            v=3
            print(test['equity'].iloc[y]+ " "+str(test['candleid'].iloc[y]) +'--drop')
            master_copy.loc[(master_copy['equity']==x) & (master_copy['candleid']==test['candleid'].iloc[y]),'breakdown']=test['Close'].iloc[y]
        if ((v ==2) & (breakout ==2) & ((test['Bullish pinbar'].iloc[y]== True) | (test['Bullish engulfing'].iloc[y]== True) | (test['Bullish swing'].iloc[y]== True))):
            v=3
            print(test['equity'].iloc[y]+ " "+str(test['candleid'].iloc[y]) +'--up')
            master_copy.loc[(master_copy['equity']==x) & (master_copy['candleid']==test['candleid'].iloc[y]),'breakout']=test['Close'].iloc[y]

train_set= master_copy[master_copy['equity']=='GUJALKALI']
fig = go.Figure(data=[go.Candlestick(x=train_set['candleid'],
                open=train_set['Open'],
                high=train_set['High'],
                low=train_set['Low'],
                close=train_set['Close'])])
fig.add_trace(go.Scatter(x=train_set['candleid'], y=train_set['lower_point'], mode='lines', name='min slope'))
fig.add_trace(go.Scatter(x=train_set['candleid'], y=train_set['upper_point'], mode='lines', name='max slope'))
fig.add_trace(go.Scatter(x=train_set['candleid'], y=train_set['breakout'], mode='markers', name='breakout', marker=dict(size=8)))
fig.add_trace(go.Scatter(x=train_set['candleid'], y=train_set['breakdown'], mode='markers', name='breakdown', marker=dict(size=8)))
fig.show()

############################################################################

master[(master['Close']< master['lower_point']) & (master['Bearish engulfing'] == True)
& (master['candleid']>224) & (master['P_min']<0.000001)  & (master['Volume'] >  (master['Max_Vol']*0.5))] 



eq_slope[(eq_slope['Slope_Min']< -0.10) & (eq_slope['Slope_Max']<eq_slope['Slope_Min'])]



eq_slope[(eq_slope['Points_max'] > 10) & (eq_slope['Points_max'] > 20)  & (eq_slope['P_max'] < 0.0000001)]
eq_slope[(eq_slope['Points_min'] > 10) & (eq_slope['Points_min'] > 20)  & (eq_slope['P_min'] < 0.0000001)]


master_copy['breakout'] = ''
master_copy['breakdown'] = ''
for x in master_copy.Equity.unique():
    test = master_copy[master_copy['equity']==x].reset_index(drop=True)
    v= 0
    breakdown = 0
    breakout = 0
    for y in range(1,len(test)):
        if ((v<1) & (breakdown==0) &  (test['Open'].iloc[y] < test['lower_point'].iloc[y])  & (test['Close'].iloc[y] > test['upper_point'].iloc[y]) & (test['upper_point'].iloc[y] > test['lower_point'].iloc[y])  )  :
            breakdown = 2
            v=2              
        if ((v <1) & (breakout==0) & (test['Open'].iloc[y] < test['upper_point'].iloc[y]) & (test['upper_point'].iloc[y]>test['lower_point'].iloc[y]) & (test['Close'].iloc[y] > test['upper_point'].iloc[y])  & (abs(test['Close'].iloc[y]  - test['Open'].iloc[y] ) > abs(test['High'].iloc[y]  - test['Low'].iloc[y] )*0.80)  & (test['Slope_Max'].iloc[y] < 0) ):
            breakout = 2
            v=2  
        if ((v <1) & (breakdown==0) & (test['Open'].iloc[y] > test['lower_point'].iloc[y]) & (test['upper_point'].iloc[y]>test['lower_point'].iloc[y]) & (test['Close'].iloc[y] < test['lower_point'].iloc[y]) & (abs(test['Close'].iloc[y]  - test['Open'].iloc[y] ) > abs(test['High'].iloc[y]  - test['Low'].iloc[y] )*0.80)  & (test['Slope_Min'].iloc[y] > 0) ):
            breakdown = 2
            v=2
        if ((v == 2) & (breakdown ==2) & ((test['Bearish pinbar'].iloc[y]== False) | (test['Bearish engulfing'].iloc[y]== True) | (test['Bearish swing'].iloc[y]== False) | (test['Bullish swing'].iloc[y]== True))  & (test['Volume'].iloc[y] >  (test['Max_Vol'].mean()*7)) ):
            v=3
            print(test['equity'].iloc[y]+ " "+str(test['candleid'].iloc[y]) +'--drop')
            master_copy.loc[(master_copy['equity']==x) & (master_copy['candleid']==test['candleid'].iloc[y]),'breakdown']=test['Close'].iloc[y]
        if ((v ==2) & (breakout ==2) & ((test['Bullish pinbar'].iloc[y]== True) | (test['Bullish engulfing'].iloc[y]== True) | (test['Bullish swing'].iloc[y]== True) | (test['Bearish engulfing'].iloc[y]== True))  & (test['Volume'].iloc[y] >  (test['Max_Vol'].mean()*7))):
            v=3
            print(test['equity'].iloc[y]+ " "+str(test['candleid'].iloc[y]) +'--up')
            master_copy.loc[(master_copy['equity']==x) & (master_copy['candleid']==test['candleid'].iloc[y]),'breakout']=test['Close'].iloc[y]
daily_agg = history.groupby(['equity','day','Date'])['High','Open','Low'].agg({'High':max,'Low':min}).sort_values('Date',ascending=True)