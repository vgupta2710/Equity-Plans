from datetime import date, datetime, timedelta
from pickle import TRUE
from re import X
from turtle import left
import pandas as pd
from sqlalchemy import column
import yfinance as yf
import numpy as np
from matplotlib import pyplot
from scipy.stats import linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn import preprocessing

def get_historicals(period,interval):
    stocks = pd.read_csv('../Input_Data/stocks.csv')
    stocks = stocks[stocks['instrumentName']=='RENUKA']
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

def get_nifty_historicals(period,interval):
    stocks = pd.read_csv('../Input_Data/stocks.csv')
    stocks = stocks[stocks['instrumentName']=='^NSEBANK']
    def refresh_data(equity,period1,interval1):
        equity1 = yf.Ticker(equity)
        history = equity1.history(period=period1,interval=interval1,actions=False)   
        return history
    historical_consolidated = pd.DataFrame(columns=['candleid','Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'equity'])
    for x in stocks['instrumentName']:
        historical = refresh_data(x,period,interval).reset_index()
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

def support_fn(df1, l, n1, n2): #n1 n2 before and after candle l
    if l-n1 < 0 or l+n2 >= len(df1):
        return 0
    for i in range(l-n1+1, l+1):
        if(df1.Low[i]>df1.Low[i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df1.Low[i]<df1.Low[i-1]):
            return 0
    return df1.Low[i-n1]

def resistance_fn(df1, l, n1, n2): #n1 n2 before and after candle l
    if l-n1 < 0 or l+n2 >= len(df1):
        return 0
    for i in range(l-n1+1, l+1):
        if(df1.High[i]<df1.High[i-1]):
            return 0
    for i in range(l+1,l+n2+1):
        if(df1.High[i]>df1.High[i-1]):
            return 0
    return df1.High[i-n1]

def support_resistance_cal_depricated(custom_list):
    custom_list.sort()
    custom_list.pop(0)
    for x in range(0,7):
        
        for i in range(1,len(custom_list)):
            if(i>=len(custom_list)):
                break
            for y in range(i+1,len(custom_list)-i):

                if abs(custom_list[i]-custom_list[i-1])/custom_list[i-1] < 0.05:
                    custom_list.pop(i)
    return custom_list            

def support_resistance_cal(custom_list):
    custom_list.sort()
    points = []
    for i in range(0,len(custom_list)):
        if(i>=len(custom_list)):
            break
        pt =0
        for y in range(i+1,len(custom_list)):
            if abs(custom_list[i]-custom_list[y])/custom_list[i] < 0.05:
                pt = pt+1 
        points.append([custom_list[i],pt])

    point = pd.DataFrame(points,columns=['pts','itr'])
    point['change'] = point['pts'].pct_change()
    point = point[point['change']>0.01]
    return point          


daily_agg,stocks= get_historicals('10d','15m')
historical,stocks = get_historicals('20d','15m')

historical,stocks = get_nifty_historicals('7d','1m')
daily_agg= historical

historical['Datetime'] = pd.to_datetime(historical['Datetime']).dt.tz_localize(None)
historical['Date'] = historical['Datetime'].dt.strftime('%Y/%m/%d')
historical['hour']=historical['Datetime'].dt.hour
historical['week']=historical['Datetime'].dt.week
historical['min']= historical['Datetime'].dt.minute
historical['day']= historical['Datetime'].dt.day

daily_agg['Datetime'] = pd.to_datetime(daily_agg['Datetime']).dt.tz_localize(None)
daily_agg['Date'] = daily_agg['Datetime'].dt.strftime('%Y/%m/%d')
daily_agg['hour']=daily_agg['Datetime'].dt.hour
daily_agg['week']=daily_agg['Datetime'].dt.week
daily_agg['min']= daily_agg['Datetime'].dt.minute
daily_agg['day']= daily_agg['Datetime'].dt.day


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
support_val = []
resistance_val = []
eqquity = []

#history = historical[(historical['Date']<'2022/10/14')]

history = historical[(historical['hour']<14) ]

#historical[historical['equity']=='ADANIPORTS'].head()

for eqq in daily_agg['equity'].unique():
    points = []
    final_points = []
    key_points_df = daily_agg[(daily_agg['Datetime'] >(daily_agg['Datetime'].min() + timedelta(days=0))) & (daily_agg['equity'] == eqq)].reset_index(drop=True)
    
    key_points_df['sr'] = key_points_df.apply(lambda x: support_fn(key_points_df, x.name,2,2), axis=1)  
    key_points_df['resistance'] = key_points_df.apply(lambda x: resistance_fn(key_points_df, x.name,2,2), axis=1)  
    for pt in key_points_df['sr'].unique():
        points.append(pt)
    for pt in key_points_df['resistance'].unique():
        points.append(pt)        
    touchpoints = support_resistance_cal(points)

    for a in range(0,len(touchpoints)):
        final_points.append(list(touchpoints.iloc[a,:2]))


    
    #resistance = support_resistance_cal(list(key_points_df['resistance'].unique()))
    eqquity.append(eqq)
    support_val.append(final_points)

for eq in history['equity'].unique():
    train_set = history[(history['Datetime'] >(history['Datetime'].min() + timedelta(days=0))) & (history['equity'] == eq)].reset_index(drop=True)
    
    train_set['pivot'] = train_set.apply(lambda x: pivotid(train_set, x.name,3,3), axis=1)  
    
    train_set['sr'] = train_set.apply(lambda x: support_fn(train_set, x.name,2,2), axis=1)  
    support = support_resistance_cal(list(train_set['sr'].unique()))

    train_set['resistance'] = train_set.apply(lambda x: resistance_fn(train_set, x.name,2,2), axis=1)  
    resistance = support_resistance_cal(list(train_set['resistance'].unique()))

    train_set['pointpos'] = train_set.apply(lambda row: pointpos(row), axis=1)

    train_set['SMA_5'] = train_set['Close'].rolling(5,min_periods=2).mean()

    if (train_set['SMA_5'][3] > train_set[train_set['candleid']==train_set['candleid'].max()]['SMA_5'].values) == True:
        backcandles =(train_set[train_set['High']==train_set['High'].max()].index[0])
    if (train_set['SMA_5'][3] < train_set[train_set['candleid']==train_set['candleid'].max()]['SMA_5'].values) == True:
        backcandles =(train_set[train_set['Low']==train_set['Low'].min()].index[0])

    candleid = train_set.shape[0] -1

    maxim = np.array([])
    minim = np.array([])
    xxmin = np.array([])
    xxmax = np.array([])
    
    for i in range(backcandles, candleid+1):
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

eq_slope= pd.DataFrame({'Equity':equity_list,'Slope_Min':slmin_list,'Slope_Max':slmax_list,'Intercept_Min':intercmin_list,'Intercept_Max':intercmax_list,'P_min': p_min,'P_max': p_max
,'Points_min': min_points,'Points_max': max_points, 'Max_Vol':volume})

key_points = pd.DataFrame({'Equity_key_point':eqquity,'Support':support_val})
eq_slope = eq_slope.merge(key_points,left_on='Equity',right_on='Equity_key_point')

history = history.merge(eq_slope,left_on = 'equity' ,right_on = 'Equity')

history['upper_point'] = (history['candleid'] * history['Slope_Max'] ) +  history['Intercept_Max'] 
history['lower_point'] = (history['candleid'] * history['Slope_Min'] ) + history['Intercept_Min'] 

master_set = daily_agg.merge(eq_slope,left_on = 'equity' ,right_on = 'Equity')
master_set = pd.merge(master_set, history[['upper_point','lower_point','equity','hour','day','min','Date']],  how='left', on=['equity','hour','day','min','Date'])


viz_set = master_set[master_set['equity']=='^NSEBANK'].reset_index()
#train_set['breakout'] = ''
#train_set['breakdown'] = ''
#train_set.loc[((train_set['Bearish engulfing']== True) | (train_set['Bearish swing']== True )),'breakdown'] = train_set[(train_set['Bearish engulfing']== True) ]['Close'] - (1e-3)
#train_set.loc[((train_set['Bullish engulfing']== True) ),'breakout'] = train_set[(train_set['Bullish engulfing']== True) ]['Close']+(1e-3)
fig = go.Figure(data=[go.Candlestick(x=viz_set['candleid'],
                open=viz_set['Open'],
                high=viz_set['High'],
                low=viz_set['Low'],
                close=viz_set['Close'])])
fig.add_trace(go.Scatter(x=viz_set['candleid'], y=viz_set['lower_point'], mode='lines', name='min slope'))
fig.add_trace(go.Scatter(x=viz_set['candleid'], y=viz_set['upper_point'], mode='lines', name='max slope'))
#fig.add_trace(go.Scatter(x=train_set['candleid'], y=train_set['breakout'], mode='markers', name='breakout', marker=dict(size=9)))
#fig.add_trace(go.Scatter(x=train_set['candleid'], y=train_set['breakdown'], mode='markers', name='breakdown', marker=dict(size=9)))
for supp in viz_set['Support'][0]:
    if supp[1] == 0:
        fig.add_hline(y=supp[0],line_width=1, line_color="dark green")
    else:
        fig.add_hline(y=supp[0],line_width=supp[1]/3,line_dash="dash", line_color="dark green",annotation_text=str(supp[1]) + "-" +  str(round(supp[0],0)), annotation_position="top left")   

#fig.add_vline(viz_set[viz_set['Datetime']==history.Datetime.max()].candleid.values[0])
fig.update_layout(
    height=500,
    title_text=str(viz_set['equity'].unique()[0]) + " - 1HR, P_Min:" + str(round(viz_set['P_min'].unique()[0],4)) + " P_Max"+ str(round(viz_set['P_max'].unique()[0],4)), 
    showlegend=False,
    xaxis_rangeslider_visible=False
)        
fig.show()

####################################################################
###### Ascending Triangle
history[
    (history['Slope_Max'] < 0) 
    &
    (history['Slope_Min'].between(-0.1,0.05,inclusive=True))
    &
    (history['Points_min'].between(3,8,inclusive=True))
    &
    (history['P_max']< 0.005)
    &
    (history['P_min']< 0.005) 
    &
    (history['hour'] == 15)
    &
    (history['day'] == history['Datetime'].max().day)
    &
    (history['Close']< history['upper_point'])
    &
    (history['Close']> history['lower_point'])    
]

###downward trend chart
history [
    (history['Slope_Max'] < 0) 
    &
    (history['Slope_Min']< 0) 
    &
    (history['Slope_Min']>history['Slope_Max']) 
    &
    (history['Points_min'].between(5,8,inclusive=True)) 
    &
    (history['P_max']< 0.005)
    &
    (history['P_min']< 0.005)
    &
    (history['Close']< history['upper_point'])
    &
    (history['Close']> history['lower_point'])
    &
    (history['hour'] == 15)
    &
    (history['day'] ==history['Datetime'].max().day)
]

eq_slope[(eq_slope['P_min'] < 0.00005) & (eq_slope['P_max'] < 0.00005) ]

eq_slope[(eq_slope['P_min'] < 0.000005) & (eq_slope['Points_min'].between(3,8,inclusive=True))]

eq_slope[(eq_slope['P_max'] < 0.000005) & (eq_slope['Points_max'].between(3,8,inclusive=True))]

for pp in 
eq_slope[(eq_slope['Slope_Min'] >0)  & (eq_slope['P_min'] < 0.00005) ]


historical[historical['Date']=='2022/10/04']


daily_aggregate = historical.groupby(['equity','day','Date'])['High','Open','Low'].agg({'High':max,'Low':min,'Open':min}).sort_values('Date',ascending=True).reset_index()

daily_aggregate['move'] = (daily_aggregate['High']-daily_aggregate['Low'])/daily_aggregate['Open']


daily_aggregate[daily_aggregate['Date']=='2022/10/04']['move']

daily_aggregate.to_clipboard(index=False)



import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import plot_importance, plot_tree

x = model_set[model_set.columns.difference(['Move','Equity','Resistance','Support','Equity_key_point','Intercept_Max','Intercept_Min','Max_Vol','Points_max','Points_min'])]
y = model_set['Move']


X_train, X_test,y_train, y_test = train_test_split(x,y ,
                                   random_state=104, 
                                   test_size=0.10, 
                                   shuffle=True)
  

reg = xgb.XGBRegressor(n_estimators=1000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=False) # Change verbose to True if you want to see it train


plot_importance(reg, height=0.9)

y_pred = reg.predict(X_test)



mean_absolute_percentage_error(y_true=y_test,
                   y_pred=y_pred)

def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


