from datetime import date, datetime, timedelta
from pickle import TRUE
from re import X
from turtle import left
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn import preprocessing
from sklearn import metrics
import pickle
import yfinance as yf
import yaml
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error,confusion_matrix
import seaborn as sns
import sklearn.metrics
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from collections import Counter
from scipy import stats

####Read historical data ######
historical = pd.read_csv('Backtest Data/Historical_combined_latest.csv')
historical = historical.drop(axis=1,columns={'Unnamed: 0'})
historical = historical[historical['candleid'] > 855555]
historical = pd.concat([historical[(historical['hour']==9) & (historical['min'] > 15)],historical[(historical['hour']>9)]], axis =0)

##Data Transformatipns: 
Y=[]
transformed = []
tl=[]
backcandles = 45
future_candles = 30
future_candles2 = 60
historical['diff'] = historical['Close'].astype('float')  - historical['Open'].astype('float')
historical['lower_wick'] = historical['Open'].astype('float')  - historical['Low'].astype('float')
historical['upper_wick'] = historical['Close'].astype('float')  - historical['High'].astype('float')
historical['High'] = historical['High'].astype('float')
historical['Low'] = historical['Low'].astype('float')
historical['Open'] = historical['Open'].astype('float')

for a in historical['date'].unique():
    day_df = historical[historical['date']==a]

    for i in range(backcandles,len(day_df)-future_candles):
        ls = day_df.iloc[i-backcandles:i]['diff'].tolist()
        candle = historical.iloc[i,0]
        date_cl = historical.iloc[i,1]
        #ls.append(str(historical.iloc[i,1] ) + " " + str(historical.iloc[i,2]))
        #ls.append(str(historical.iloc[i+backcandles,1] ) + " " + str(historical.iloc[i+backcandles,2]))    
        #same_day = historical.iloc[i-backcandles:i]['day'].nunique()
        same_day = 1
        Y_var_high = day_df.iloc[i:i+future_candles]['High'].tolist()
        Y_var_low = day_df.iloc[i:i+future_candles]['Low'].tolist()
        Y_var_high2 = max(day_df.iloc[i:i+future_candles2]['High'].tolist())
        Y_var_low2 = min(day_df.iloc[i:i+future_candles2]['Low'].tolist())

        if ls[backcandles-1] > 1.0:
            tl = [x/ ls[backcandles-1] for x in ls]
            tl.append(same_day)
            tl.append(candle)
            tl.append(date_cl)
            tl.append(Y_var_high[0])
            tl.append(max(Y_var_high))
            tl.append(Y_var_high2)
            tl.append(Y_var_low[0])
            tl.append(min(Y_var_low))
            tl.append(Y_var_low2)
            transformed.append(tl)
        elif ls[backcandles-2]>  1.0:    
            tl = [x/ ls[backcandles-2] for x in ls]
            tl.append(same_day)
            tl.append(candle)
            tl.append(date_cl)    
            tl.append(Y_var_high[0])
            tl.append(max(Y_var_high))
            tl.append(Y_var_high2)
            tl.append(Y_var_low[0])
            tl.append(min(Y_var_low))
            tl.append(Y_var_low2)       
            transformed.append(tl)
        elif ls[backcandles-3] > 1.0:    
            tl = [x/ ls[backcandles-3] for x in ls]
            tl.append(same_day)
            tl.append(candle) 
            tl.append(date_cl)
            tl.append(Y_var_high[0])
            tl.append(max(Y_var_high))
            tl.append(Y_var_high2)
            tl.append(Y_var_low[0])
            tl.append(min(Y_var_low))
            tl.append(Y_var_low2)
            transformed.append(tl)         

working_data_set = pd.DataFrame(transformed).reset_index(drop=True).round(3)

##working_data_set[backcandles+3] 30 period High
##working_data_set[backcandles+2] Current High
##working_data_set[backcandles+4] 60 period high
##working_data_set[backcandles+5] Current Period Low
##working_data_set[backcandles+6] 30 period low
##working_data_set[backcandles+7] 60 period low

working_data_set['Up_Move_Actual'] = working_data_set[backcandles+4] - working_data_set[backcandles+3] 
working_data_set['Up_Move_Actual_'] = working_data_set[backcandles+5] - working_data_set[backcandles+3]

working_data_set['Down_Move_Actual'] = working_data_set[backcandles+6] - working_data_set[backcandles+7]
working_data_set['Down_Move_Actual_'] = working_data_set[backcandles+6] - working_data_set[backcandles+8]

working_data_set['bull1']= working_data_set.apply(lambda x: 1 if x['Up_Move_Actual']>100 else 0,axis=1 )
working_data_set['bull1_']= working_data_set.apply(lambda x: 1 if x['Up_Move_Actual_']>100 else 0,axis=1 )
working_data_set['bear1']= working_data_set.apply(lambda x: 1 if x['Down_Move_Actual']>100 else 0,axis=1 ) 
working_data_set['bear1_']= working_data_set.apply(lambda x: 1 if x['Down_Move_Actual_']>100 else 0,axis=1 ) 

######################################  Model Building ################################################
X_train = working_data_set.iloc[:-24000,:backcandles]
y_train = working_data_set.iloc[:-24000]["bull1"]
X_test = working_data_set.iloc[-24000:,:backcandles]
y_test = working_data_set.iloc[-24000:]["bull1"]


counter = Counter(y_train)
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)

bullish_model = xgb.XGBClassifier(n_estimators=2100,max_depth = 9,learning_rate=0.01,min_child_weight=9,reg_lambda= 18,scale_pos_weight =4.5,subsample = 0.15)
bullish_model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric = 'error@0.75', 
        #early_stopping_rounds=2,
       verbose=False) # Change verbose to True if you want to see it train


X_train = working_data_set.iloc[:-30000,:backcandles]
y_train_bear = working_data_set.iloc[:-30000]["bear1"]
X_test = working_data_set.iloc[-30000:,:backcandles]
y_test_bear = working_data_set.iloc[-30000:]["bear1_"]

counter = Counter(y_train_bear)
# estimate scale_pos_weight value
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)

bearish_model = xgb.XGBClassifier(n_estimators=2100,max_depth = 9,learning_rate=0.01,min_child_weight=9,reg_lambda= 9,scale_pos_weight =1.8,subsample = 0.5)
bearish_model.fit(X_train, y_train_bear,
        eval_set=[(X_train, y_train), (X_test, y_test_bear)],
        eval_metric = 'error',
        #early_stopping_rounds=2,
       verbose=False) # Change verbose to True if you want to see it train


################### Validating the above created Model #################################
y_pred = bullish_model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
f = sns.heatmap(cm, annot=True, fmt='d')
f.set(xlabel='Y_Pred', ylabel='Y_True')
print(metrics.classification_report(y_test,y_pred,digits=3 ))

y_pred = bearish_model.predict(X_test)
cm = confusion_matrix(y_test_bear,y_pred)
f = sns.heatmap(cm, annot=True, fmt='d')
f.set(xlabel='Y_Pred', ylabel='Y_True')
print(metrics.classification_report(y_test_bear,y_pred,digits=3 ))

################### Model Save #################################
filename = 'Models\bullish_model_4_10.sav'
pickle.dump(bullish_model, open(filename, 'wb'))

filename = 'Models\bearish_model_4_10.sav'
pickle.dump(bearish_model, open(filename, 'wb'))




################### Model Consolidate Check #################################
def get_nifty_historicals(period1,interval1):
    equity='^NSEBANK'
    #equity='BHP.AX'
    equity1 = yf.Ticker(equity)
    bank_nifty_data = equity1.history(period=period1,interval=interval1,actions=False).reset_index()
    bank_nifty_data.drop(columns=['Volume'],inplace=True)
    bank_nifty_data.rename(columns={'into':'Open','inth':'High','intl':'Low','intc':'Close'},inplace=True)
    bank_nifty_data['Datetime'] = pd.to_datetime(bank_nifty_data['Datetime'])    
    return bank_nifty_data

historical = get_nifty_historicals('1d','1m')


Y=[]
transformed = []
tl=[]
backcandles = 18
future_candles = 30
future_candles2 = 60
historical['diff'] = historical['Close'].astype('float')  - historical['Open'].astype('float')
historical['lower_wick'] = historical['Open'].astype('float')  - historical['Low'].astype('float')
historical['upper_wick'] = historical['Close'].astype('float')  - historical['High'].astype('float')
historical['High'] = historical['High'].astype('float')
historical['Low'] = historical['Low'].astype('float')
historical['Open'] = historical['Open'].astype('float')
historical['diff2'] = historical['diff'].apply(lambda x:1 if x>0 else 0)

historical['date'] = historical['Datetime'].dt.date
historical['candleid']  = historical.index

for a in historical['date'].unique():
    day_df = historical[historical['date']==a]

    for i in range(backcandles,len(day_df)-future_candles):
        ls = day_df.iloc[i-backcandles:i]['Open'].tolist()
        x = day_df.iloc[i-backcandles:i]['candleid'].tolist()

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, ls)
        candle = day_df.iloc[i,0]
        date_cl = day_df.iloc[i,1]
        Y_var_high = day_df.iloc[i:i+future_candles]['High'].tolist()
        Y_var_low = day_df.iloc[i:i+future_candles]['Low'].tolist()
        Y_var_high2 = max(day_df.iloc[i:i+future_candles2]['High'].tolist())
        Y_var_low2 = min(day_df.iloc[i:i+future_candles2]['Low'].tolist())
        tl = []
        intercept = 0
        tl.append(slope)
        tl.append(p_value)
        ema = day_df.iloc[i-backcandles:i]['Close'].ewm(com=0.9).mean().iloc[-1]
        if (ema > day_df['Open'].iloc[i] and ema < day_df['Close'].iloc[i]) or  (ema < day_df['Close'].iloc[i] and  ema > day_df['Open'].iloc[i])  :
            intercept = 1
        else:    
            intercept = 0
        tl.append(intercept)
        tl.append((max(day_df.iloc[i-backcandles:i]['High'].tolist()) - min(day_df.iloc[i-backcandles:i]['Low'].tolist())) / day_df['Open'].iloc[i] *100)
        tl.append(np.abs(day_df['Close'].iloc[i] - day_df['Open'].iloc[i]) / day_df['Open'].iloc[i] * 100)
        tl.append(np.abs(day_df['Close'].iloc[i] - day_df['Open'].iloc[i]) / np.abs(day_df['High'].iloc[i] - day_df['Low'].iloc[i]))
        tl.append(day_df['diff2'].iloc[i])
        tl.append(candle) 
        tl.append(date_cl)
        tl.append(Y_var_high[0])
        tl.append(max(Y_var_high))
        tl.append(Y_var_high2)
        tl.append(Y_var_low[0])
        tl.append(min(Y_var_low))
        tl.append(Y_var_low2)
        transformed.append(tl)   



working_data_set = pd.DataFrame(transformed).reset_index(drop=True).round(2)
backcandles=7
##working_data_set[backcandles+3] 30 period High
##working_data_set[backcandles+2] Current High
##working_data_set[backcandles+4] 60 period high
##working_data_set[backcandles+5] Current Period Low
##working_data_set[backcandles+6] 30 period low
##working_data_set[backcandles+7] 60 period low

working_data_set['Up_Move_Actual'] = working_data_set[backcandles+3] - working_data_set[backcandles+2] 
working_data_set['Up_Move_Actual_'] = working_data_set[backcandles+4] - working_data_set[backcandles+2]

working_data_set['Down_Move_Actual'] = np.abs(working_data_set[backcandles+5] - working_data_set[backcandles+6])
working_data_set['Down_Move_Actual_'] = np.abs(working_data_set[backcandles+5] - working_data_set[backcandles+7])

working_data_set['bull1']= working_data_set.apply(lambda x: 1 if x['Up_Move_Actual']>75 else 0,axis=1 )
working_data_set['bull1_']= working_data_set.apply(lambda x: 1 if x['Up_Move_Actual_']>75 else 0,axis=1 )
working_data_set['bear1']= working_data_set.apply(lambda x: 1 if x['Down_Move_Actual']>75 else 0,axis=1 ) 
working_data_set['bear1_']= working_data_set.apply(lambda x: 1 if x['Down_Move_Actual_']>75 else 0,axis=1 ) 


X_train = working_data_set.iloc[:-24000,:backcandles]
y_train = working_data_set.iloc[:-24000]["bull1"]
X_test = working_data_set.iloc[-24000:,:backcandles]
y_test = working_data_set.iloc[-24000:]["bull1_"]

counter = Counter(y_train)
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)

bullish_model = xgb.XGBClassifier(n_estimators=2100,max_depth = 6,learning_rate=0.01,min_child_weight=9,reg_lambda= 9,scale_pos_weight =counter[0] / counter[1],subsample = 0.5)
bullish_model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric = 'auc', 
        #early_stopping_rounds=2,
       verbose=False) # Change verbose to True if you want to see it train


X_train = working_data_set.iloc[:-24000,:backcandles]
y_train_bear = working_data_set.iloc[:-24000]["bear1"]
X_test = working_data_set.iloc[-24000:,:backcandles]
y_test_bear = working_data_set.iloc[-24000:]["bear1_"]

counter = Counter(y_train_bear)
# estimate scale_pos_weight value
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)

bearish_model = xgb.XGBClassifier(n_estimators=2100,max_depth = 9,learning_rate=0.01,min_child_weight=9,reg_lambda= 9,scale_pos_weight =counter[0] / counter[1],subsample = 0.5)
bearish_model.fit(X_train, y_train_bear,
        eval_set=[(X_train, y_train), (X_test, y_test_bear)],
        eval_metric = 'error',
        #early_stopping_rounds=2,
       verbose=False) # Change verbose to True if you want to see it train

y_pred = bullish_model.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
f = sns.heatmap(cm, annot=True, fmt='d')
f.set(xlabel='Y_Pred', ylabel='Y_True')
print(metrics.classification_report(y_test,y_pred,digits=3 ))

y_pred = bearish_model.predict(X_test)
cm = confusion_matrix(y_test_bear,y_pred)
f = sns.heatmap(cm, annot=True, fmt='d')
f.set(xlabel='Y_Pred', ylabel='Y_True')
print(metrics.classification_report(y_test_bear,y_pred,digits=3 ))

working_data_set.to_clipboard()

y_pred = bullish_model.predict_proba(X_test)
test_output = pd.merge(working_data_set.iloc[-24000:,].reset_index(drop=True),pd.DataFrame(y_pred,columns=['bull_0','bull_1']),left_index=True,right_index=True)
y_pred = bearish_model.predict_proba(X_test)

working_data_set['Agg'] = working_data_set.iloc[:-3000,:12].sum(axis=1)

test_output = pd.merge(test_output,pd.DataFrame(y_pred,columns=['bear_0','bear_1']),left_index=True,right_index=True)
test_output.to_clipboard()

working_data_set[(working_data_set[5]>0.80) & (working_data_set[4]>0.05)].to_clipboard()


import shap
explainer = shap.Explainer(bullish_model)
shap_values = explainer(X_test)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])

historical.iloc[121105]

historical[historical['candleid']==994152]


working_data_set[(working_data_set[5]>0.80) & (working_data_set[4]>0.05) & (working_data_set[6] == 0) & (working_data_set[0] < 0)]['bear1_'].sum()

working_data_set[(working_data_set[5]>0.80) & (working_data_set[4]>0.03) & (working_data_set[6] == 0) & (working_data_set[0] < 0)]['bull1_'].sum()

/
working_data_set[(working_data_set[5]>0.80) & (working_data_set[4]>0.05) & (working_data_set[6] < 0) & (working_data_set[0] < 0)]['bear1_'].count()

working_data_set[(working_data_set[5]>0.80) & (working_data_set[4]>0.05) & (working_data_set[6] < 0) & (working_data_set[0] < 0)]['bull1_'].sum()
/
working_data_set[(working_data_set[5]>0.80) & (working_data_set[4]>0.05) & (working_data_set[6] < 0) & (working_data_set[0] < 0)]['bear1_'].count()

working_data_set[(working_data_set[5]>0.80)  & (working_data_set[6] == 0) & (working_data_set[0] < 0)].to_clipboard()

working_data_set[(working_data_set[2]==1) & (working_data_set[5]>0.80)  & (working_data_set[6] == 1) & (working_data_set[0] > 0) ].to_clipboard()

