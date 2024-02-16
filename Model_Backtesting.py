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
import smtplib
import time

working_data_set = pd.read_csv('/Users/vaigupta/Documents/Equity Master Folder/Equity_Simulator/Development/Backtest Data/BNF_1m_04_08__02_10.csv')

os.listdir('Models')

bullish_model = pickle.load(open('Models/bullish_model_16_10.sav','rb'))
bearish_model = pickle.load(open('Models/bearish_model_16_10.sav','rb'))

def mail(y,y1):
    mail=smtplib.SMTP('smtp.office365.com', 587)
    mail.ehlo()
    mail.starttls()
    sender='gupta123vaibhav@live.com'
    recipient=['vaibhav.gupta1027@gmail.com']
    mail.login(sender,'suraj9560863332')
    content= '\n BNF Predicted \n Buulish P_Value @ {0} \n \n P_value for Bearish @ {1}'.format(y,y1)
    mail.sendmail(sender, recipient, content)
    mail.close()

def get_nifty_historicals(period1,interval1):
    equity='^NSEBANK'
    #equity='BHP.AX'
    equity1 = yf.Ticker(equity)
    bank_nifty_data = equity1.history(period=period1,interval=interval1,actions=False).reset_index()
    bank_nifty_data.drop(columns=['Volume'],inplace=True)
    bank_nifty_data.rename(columns={'into':'Open','inth':'High','intl':'Low','intc':'Close'},inplace=True)
    bank_nifty_data['Datetime'] = pd.to_datetime(bank_nifty_data['Datetime'])    
    return bank_nifty_data


#### new code start ####
for x in range(0,300):
    backcandles = 90
    tt = []
    historical = get_nifty_historicals('2d','1m')
    historical['diff'] = historical['Close'].astype('float')  - historical['Open'].astype('float')
    ls = historical.iloc[-90:]['diff'].tolist()
    if ls[backcandles-1]>1.0:
        tl = [x/ ls[backcandles-1] for x in ls]
    elif ls[backcandles-2]>1.0: 
        tl = [x/ ls[backcandles-2] for x in ls]  
    elif ls[backcandles-3]>1.0: 
        tl = [x/ ls[backcandles-3] for x in ls]  
    tt.append(tl)
    y_pred = bullish_model.predict_proba(pd.DataFrame(tt))[0][1]
    y_pred_bearish = bearish_model.predict_proba(pd.DataFrame(tt))[0][1]

    if (y_pred>0.54 or y_pred_bearish > 0.55):
        mail(y_pred,y_pred_bearish)
        time.sleep(60)
    else:
        print(x)
        time.sleep(60)
#### new code end ####


historical = get_nifty_historicals('2d','1m')
Y=[]
transformed = []
tl=[]
backcandles = 90
future_candles = 30
future_candles2 = 60
historical['diff'] = historical['Close'].astype('float')  - historical['Open'].astype('float')
historical['lower_wick'] = historical['Open'].astype('float')  - historical['Low'].astype('float')
historical['upper_wick'] = historical['Close'].astype('float')  - historical['High'].astype('float')
historical['High'] = historical['High'].astype('float')
historical['Low'] = historical['Low'].astype('float')
historical['Open'] = historical['Open'].astype('float')

for i in range(backcandles,len(historical)-future_candles):
    ls = historical.iloc[i-backcandles:i]['diff'].tolist()
    candle = historical.iloc[i,0]
    date_cl = historical.iloc[i,1]
    #ls.append(str(historical.iloc[i,1] ) + " " + str(historical.iloc[i,2]))
    #ls.append(str(historical.iloc[i+backcandles,1] ) + " " + str(historical.iloc[i+backcandles,2]))    
    #same_day = historical.iloc[i-backcandles:i]['day'].nunique()
    same_day = 1
    Y_var_high = historical.iloc[i:i+future_candles]['High'].tolist()
    Y_var_low = historical.iloc[i:i+future_candles]['Low'].tolist()
    Y_var_high2 = max(historical.iloc[i:i+future_candles2]['High'].tolist())
    Y_var_low2 = min(historical.iloc[i:i+future_candles2]['Low'].tolist())

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

working_data_set['bull1']= working_data_set.apply(lambda x: 1 if x['Up_Move_Actual']>75 else 0,axis=1 )
working_data_set['bull1_']= working_data_set.apply(lambda x: 1 if x['Up_Move_Actual_']>75 else 0,axis=1 )
working_data_set['bear1']= working_data_set.apply(lambda x: 1 if x['Down_Move_Actual']>75 else 0,axis=1 ) 
working_data_set['bear1_']= working_data_set.apply(lambda x: 1 if x['Down_Move_Actual_']>75 else 0,axis=1 ) 



X_test = working_data_set.iloc[:,0:backcandles]
y_test = working_data_set.iloc[:]["bull1"]

y_pred = bullish_model.predict_proba(X_test)
test_output = pd.merge(working_data_set.reset_index(drop=True),pd.DataFrame(y_pred,columns=['pred_bull_0','pred_bull_1']),left_index=True,right_index=True)
y_pred = bearish_model.predict_proba(X_test)
test_output = pd.merge(test_output,pd.DataFrame(y_pred,columns=['pred_bear_0','pred_bear_1']),left_index=True,right_index=True)

test_output['Y_pred_bull'] = test_output.apply(lambda x: 1 if x['pred_bull_1']>0.55 else 0,axis=1)
test_output['Y_pred_bear'] = test_output.apply(lambda x: 1 if x['pred_bear_1']>0.55 else 0,axis=1)
test_output['date'] = pd.to_datetime(test_output.iloc[:,91]).dt.date
#test_output['hour'] = test_output['date'].dt.hour
#test_output['min'] = test_output['date'].dt.minute
#test_output = test_output[(test_output['hour']<15)]

Agg_output = test_output[test_output['Y_pred_bull']==1].groupby('date').sum()[['Y_pred_bull','bull1','bull1_','bear1','bear1_']].reset_index()
Agg_output


Agg_output = test_output[test_output['Y_pred_bear']==1].groupby('date').sum()[['Y_pred_bear','bull1','bull1_','bear1','bear1_']].reset_index()
Agg_output

test_output.iloc[:,90:].to_clipboard()