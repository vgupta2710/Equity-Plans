from datetime import date, datetime, timedelta
from pickle import TRUE
from re import X
from turtle import left
import pandas as pd
from sqlalchemy import column
import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn import preprocessing
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.optimizers import SGD

import sklearn.metrics
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error,confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import plot_importance, plot_tree
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier

def get_historicals(period,interval):
    stocks = pd.read_csv('../Input_Data/stocks.csv')
    #stocks = stocks[stocks['instrumentName']=='RENUKA']

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
    x='^NSEBANK'
    def refresh_data(equity,period1,interval1):
        equity1 = yf.Ticker(equity)
        history = equity1.history(period=period1,interval=interval1,actions=False)   
        return history
    historical_consolidated = pd.DataFrame(columns=['candleid','Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'equity'])

    historical = refresh_data(x,period,interval).reset_index()
    historical['equity']=x
    historical.reset_index(inplace=True)
    historical.columns = ['candleid','Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'equity']
    historical_consolidated = pd.concat([historical_consolidated,historical])
    return historical_consolidated

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


historical = get_nifty_historicals('3d','1m')

consolidated_extract = pd.read_csv('../Testing/data/recent/Consolidated_2022.txt',names=('index','date','time','Open','High','Low','Close','0','02'))
consolidated_extract = consolidated_extract.sort_values(by=['date','time'])
historical = consolidated_extract
historical.reset_index(drop=True,inplace=True)
historical = historical.reset_index().rename(columns={'level_0':'candleid','date':'Datetime'})

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
support_val = []
resistance_val = []
eqquity = []
row = []


Y=[]
transformed = []
backcandles = 60
future_candles = 60
for i in range(backcandles,len(historical)-future_candles):
    ls = historical.iloc[i-backcandles:i]['Open'].tolist()
    transformed.append([x/ ls[0] for x in ls])
    
    Y_var_high = historical.iloc[i:i+future_candles]['High'].tolist()
    Y_var_low = historical.iloc[i:i+future_candles]['Low'].tolist()
    y_var = 0
    if (max(Y_var_high) /  Y_var_high[0])> 1.003:
        y_var = 1
    if  (Y_var_low[0] - min(Y_var_low)) /Y_var_low[0] > 0.003:   
        y_var = 2
    Y.append(y_var)


working_data_set = pd.DataFrame(transformed).reset_index(drop=True)
working_data_set = pd.concat([working_data_set,pd.DataFrame({'Y':Y})],axis=1)

sns.countplot(data=working_data_set,x="Y")

zero_sampled_working_data_set = working_data_set[working_data_set["Y"] == 0].sample(n=int(round(len(working_data_set[working_data_set["Y"] == 0])*.60,0)), random_state=14)

sampled_working_data_set = pd.concat([zero_sampled_working_data_set,working_data_set[working_data_set["Y"] > 0]],axis= 0, ignore_index=True)
sns.countplot(data=sampled_working_data_set,x="Y")

x = pd.DataFrame(sampled_working_data_set.drop(["Y"],axis=1)).reset_index(drop=True)
y = pd.DataFrame(sampled_working_data_set[['Y']])

X_train, X_test,y_train, y_test = train_test_split(x,y ,
                                   random_state=15, 
                                   test_size=0.2) 
  

ada_boost = AdaBoostClassifier(n_estimators=100,
                         learning_rate=1)

model = ada_boost.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print("Recall:", metrics.recall_score(y_test, y_pred,average='macro'))
print("Precision:", metrics.precision_score(y_test, y_pred,average='macro'))

cm = confusion_matrix(y_pred, y_test)
f = sns.heatmap(cm, annot=True, fmt='d')


y_pred = model.predict(X_train)

cm = confusion_matrix(y_pred, y_train)
f = sns.heatmap(cm, annot=True, fmt='d')




pd.concat([y_test.reset_index(drop=True),pd.DataFrame(y_pred)],axis= 1).to_clipboard()

reg = xgb.XGBRegressor(n_estimators=100000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=150,
       verbose=False) # Change verbose to True if you want to see it train


plot_importance(reg, height=0.6)
y_pred = reg.predict(X_test).astype(int)
print(metrics.classification_report(y_test, y_pred, digits=3))

cm = confusion_matrix(y_pred, y_test)
f = sns.heatmap(cm, annot=True, fmt='d')


######################################################################################################################################################

working_data_set = pd.RangeIndex(0,60).to_frame().transpose().drop([0])
for i in range(backcandles,len(historical)-future_candles):

    working_data_set.append(historical.iloc[i-backcandles:i].reset_index(drop=True)[['Open']].transpose().reset_index(drop=True))
    y_var = 0
    if (historical.iloc[i:i+future_candles].High.max() /  historical.iloc[i].High)> 1.005:
        y_var = 1
    if  (historical.iloc[i].Low - historical.iloc[i:i+future_candles].Low.min()) /historical.iloc[i].Low > 0.005:   
        y_var = 2





for i in range(backcandles,len(historical)-future_candles):
    train_set = historical.iloc[i-backcandles:i].reset_index(drop=True)

    maxim = np.array([])
    minim = np.array([])
    xxmin = np.array([])
    xxmax = np.array([])

    train_set['pivot'] = train_set.apply(lambda x: pivotid(train_set, x.name,5,5), axis=1)          
    train_set['pointpos'] = train_set.apply(lambda row: pointpos(row), axis=1)


    minim = np.append(minim, train_set[train_set['pivot'] == 1].Low).astype(int)
    xxmin = np.append(xxmin, train_set[train_set['pivot'] == 1].candleid).astype(int) 
    maxim = np.append(maxim, train_set[train_set['pivot'] == 2].High).astype(int)
    xxmax = np.append(xxmax, train_set[train_set['pivot'] == 2].candleid).astype(int) 

    try:    
        slmin, intercmin, rmin, pmin, semin = linregress(xxmin, minim)
        slmax, intercmax, rmax, pmax, semax = linregress(xxmax, maxim)   
    except:
        continue

    y_var = 0
    if (historical.iloc[i:i+future_candles].High.max() /  historical.iloc[i].High)> 1.005:
        y_var = 1
    if  (historical.iloc[i].Low - historical.iloc[i:i+future_candles].Low.min()) /historical.iloc[i].Low > 0.005:   
        y_var = 2

    row.append(i)
    slmin_list.append(slmin)
    slmax_list.append(slmax) 
    intercmin_list.append(intercmin)
    intercmax_list.append(intercmax)
    p_min.append(pmin)
    p_max.append(pmax)
    Y.append(y_var)


eq_slope= pd.DataFrame({'Candle':row,'Slope_Min':slmin_list,'Slope_Max':slmax_list,'Intercept_Min':intercmin_list,'Intercept_Max':intercmax_list,'P_min': p_min,'P_max': p_max,'Y':Y})
eq_slope.groupby("Y").count()
eq_slope_backup= eq_slope.copy()

eq_slope =eq_slope_backup

eq_slope = eq_slope[(eq_slope['Y']< 2) ]

eq_slope.loc(eq_slope[eq_slope['Y']==2].index)

eq_slope.loc[eq_slope['Y']==2,'Y']=1

eq_slope['Slope_Max'] = eq_slope['Slope_Max'].round(3)
eq_slope['Slope_Min'] = eq_slope['Slope_Min'].round(3)
eq_slope['P_min'] = eq_slope['P_min'].round(3)
eq_slope['P_max'] = eq_slope['P_max'].round(3)



x = eq_slope[eq_slope.columns.difference(['Y','Candle','Intercept_Min','Intercept_Max'])]
y = eq_slope['Y']

X_train, X_test,y_train, y_test = train_test_split(x,y ,
                                   random_state=15, 
                                   test_size=0.2) 
  

reg = xgb.XGBRegressor(n_estimators=100000)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=150,
       verbose=False) # Change verbose to True if you want to see it train


plot_importance(reg, height=0.6)
y_pred = reg.predict(X_test).astype(int)
print(metrics.classification_report(y_test, y_pred, digits=3))

cm = confusion_matrix(y_test, y_pred)
f = sns.heatmap(cm, annot=True, fmt='d')

###################### LSTM Implementation #######################################
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(LSTM(units=50,activation = 'relu'))
model.add(Dense(1))


opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model_history  = model.fit(X_train, y_train, epochs=50, batch_size=50, verbose=1,validation_data=[X_test,y_test] )

y_pred = model.predict(X_test, verbose=1).astype(int)
print(metrics.classification_report(y_test, y_pred, digits=3))

cm = confusion_matrix(y_test, y_pred)
f = sns.heatmap(cm, annot=True, fmt='d')

# evaluate the model
train_acc = model.evaluate(X_train, y_train, verbose=0)
test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
model.get_weights()

model.predict_classes(X_test, verbose=0)

from sklearn.metrics import precision_score
precision_score(y_test, y_pred)

type(y)
pd.DataFrame(y_test).groupby("Y").count()



model = Sequential()
model.add(Dense(60, input_shape=(X_train.shape[1],1), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# evaluate model with standardized dataset
estimator = KerasClassifier(model=model, epochs=10, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, x, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

from sklearn.model_selection import StratifiedKFold
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

############################Latest data validaion set#################################################

historical = get_nifty_historicals('7d','1m')


eq_slope1= pd.DataFrame({'Candle':row,'Slope_Min':slmin_list,'Slope_Max':slmax_list,'Intercept_Min':intercmin_list,'Intercept_Max':intercmax_list,'P_min': p_min,'P_max': p_max,'Y':Y})
eq_slope1.groupby("Y").count()

eq_slope1['Slope_Max'] = eq_slope1['Slope_Max'].round(3)
eq_slope1['Slope_Min'] = eq_slope1['Slope_Min'].round(3)
eq_slope1['P_min'] = eq_slope1['P_min'].round(3)
eq_slope1['P_max'] = eq_slope1['P_max'].round(3)

eq_slope1 = eq_slope1[(eq_slope1['Y'] != 1)].reset_index(drop=True)

eq_slope1.loc[eq_slope1['Y']==2,'Y']=1
x = eq_slope1[eq_slope1.columns.difference(['Y','Candle','Intercept_Min','Intercept_Max'])]
y = eq_slope1['Y']

y_pred = reg.predict(x).astype(int)
print(metrics.classification_report(y, y_pred, digits=3))

cm = confusion_matrix(y, y_pred)
f = sns.heatmap(cm, annot=True, fmt='d')



################################################################################################################################################################################

sns.distplot(eq_slope[eq_slope['Y']==0]['P_max'],label ='0')
sns.distplot(eq_slope[eq_slope['Y']==1]['P_max'],label ='1')
sns.distplot(eq_slope[eq_slope['Y']==2]['P_max'],label ='2')
plt.legend(loc=1, prop={'size': 8})
plt.figure(figsize=(10,10))


files = os.listdir('/Users/vaigupta/Downloads/Intraday 1 Min Data/Consolidated')

files1 = list(filter(lambda x: 'BNF' in x, files))

os.chdir('/Users/vaigupta/Downloads/Intraday 1 Min Data/Consolidated/')

for x in files1:
    temp = pd.read_csv(x,names=('index','date','time','Open','High','Low','Close','0','02'))
    consolidated = pd.concat((consolidated,temp),axis=0)

consolidated=temp.head(0)

consolidated = consolidated[consolidated['date'] > 20200101]

consolidated.tail()
consolidated = consolidated.sort_values(by=['date','time'])
historical = consolidated
historical.reset_index(drop=True,inplace=True)
historical = historical.reset_index().rename(columns={'level_0':'candleid'})