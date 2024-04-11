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
from keras.optimizers import SGD
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping

####Read historical data ######
historical = pd.read_csv('Backtest Data/Historical_combined_latest.csv')
historical = historical.drop(axis=1,columns={'Unnamed: 0'})
historical = historical[historical['candleid'] > 655555]
historical = pd.concat([historical[(historical['hour']==9) & (historical['min'] > 15)],historical[(historical['hour']>9)]], axis =0)


##Data Transformatipns: 
Y=[]
transformed = []
tl=[]
backcandles = 30
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

working_data_set = pd.DataFrame(transformed).reset_index(drop=True).round(2)

working_data_set['Up_Move_Actual'] = working_data_set[backcandles+4] - working_data_set[backcandles+3] 
working_data_set['Up_Move_Actual_'] = working_data_set[backcandles+5] - working_data_set[backcandles+3]

working_data_set['Down_Move_Actual'] = working_data_set[backcandles+6] - working_data_set[backcandles+7]
working_data_set['Down_Move_Actual_'] = working_data_set[backcandles+6] - working_data_set[backcandles+8]

working_data_set['bull1']= working_data_set.apply(lambda x: 1 if x['Up_Move_Actual']>200 else 0,axis=1 )
working_data_set['bull1_']= working_data_set.apply(lambda x: 1 if x['Up_Move_Actual_']>200 else 0,axis=1 )
working_data_set['bear1']= working_data_set.apply(lambda x: 1 if x['Down_Move_Actual']>200 else 0,axis=1 ) 
working_data_set['bear1_']= working_data_set.apply(lambda x: 1 if x['Down_Move_Actual_']>200 else 0,axis=1 ) 

######################################  Model Building ################################################
X_train = working_data_set.iloc[:-48000,:backcandles]
y_train = working_data_set.iloc[:-48000]["bull1"]
X_train_val = working_data_set.iloc[-48000:-24000,:backcandles]
y_train_val = working_data_set.iloc[-48000:-24000]["bull1"]
X_test = working_data_set.iloc[-24000:,:backcandles]
y_test = working_data_set.iloc[-24000:]["bull1"]


counter = Counter(y_train)
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)

oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)
X_train_val, y_train_val = oversample.fit_resample(X_train_val, y_train_val)
X_test, y_test = oversample.fit_resample(X_test, y_test)

counter = Counter(y_train)
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)

model = Sequential()
model.add(LSTM(units=13, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(LSTM(12,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience=10)
epoch = 250
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
model_history  = model.fit(X_train, y_train, epochs=epoch, batch_size=10000, verbose=1,validation_data=[X_train_val,y_train_val],callbacks = [es] )

history_dict = model_history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, epoch + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title   ('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


y_pred=pd.DataFrame(model.predict(X_test))
y_pred = y_pred[0].apply(lambda x: 0 if x < 0.6 else 1)
cm = confusion_matrix(y_test,y_pred)
f = sns.heatmap(cm, annot=True, fmt='d')
f.set(xlabel='Y_Pred', ylabel='Y_True')
print(metrics.classification_report(y_test,y_pred,digits=3 ))


y_pred=pd.DataFrame(model.predict(X_train))
y_pred = y_pred[0].apply(lambda x: 0 if x < 0.6 else 1)
cm = confusion_matrix(y_train,y_pred)
f = sns.heatmap(cm, annot=True, fmt='d')
f.set(xlabel='Y_Pred', ylabel='Y_True')
print(metrics.classification_report(y_test,y_pred,digits=3 ))


######################################  Model Building - Multi Label ################################################

working_data_set['bull1']= working_data_set.apply(lambda x: 1 if x['Up_Move_Actual']>200 else 0,axis=1 )
working_data_set['bull1_']= working_data_set.apply(lambda x: 1 if x['Up_Move_Actual_']>200 else 0,axis=1 )
working_data_set['bear1']= working_data_set.apply(lambda x: 2 if x['Down_Move_Actual']>200 else 0,axis=1 ) 
working_data_set['bear1_']= working_data_set.apply(lambda x: 2 if x['Down_Move_Actual_']>200 else 0,axis=1 ) 

working_data_set['Move'] = (working_data_set['bull1']+working_data_set['bear1'])

X_train = working_data_set.iloc[:-48000,:backcandles]
y_train = working_data_set.iloc[:-48000]["Move"].shape[1]
X_train_val = working_data_set.iloc[-48000:-24000,:backcandles]
y_train_val = working_data_set.iloc[-48000:-24000]["Move"]
X_test = working_data_set.iloc[-24000:,:backcandles]
y_test = working_data_set.iloc[-24000:]["Move"]

counter = Counter(y_train)
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)



sns.barplot(working_data_set['Move'].unique(),working_data_set['Move'].groupby(by=working_data_set['Move']).count())
plt.plot(working_data_set['Move'].groupby(by=working_data_set['Move']).count(),type='bar')

sns.barplot(y_train.unique(),y_train.groupby(by=y_train).count())

oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)
X_train_val, y_train_val = oversample.fit_resample(X_train_val, y_train_val)
X_test, y_test = oversample.fit_resample(X_test, y_test)

counter = Counter(y_train)
estimate = counter[0] / counter[1]
print('Estimate: %.3f' % estimate)

model = Sequential()
model.add(LSTM(units=20, return_sequences=True, input_shape=(30,)))

model.add(Dense(4, activation='softmax'))
model.summary()

from keras import models

model = models.Sequential()
model.add(Dense(15, activation='relu', input_shape=(30,)))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))

es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience=10)
epoch = 50
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model_history  = model.fit(X_train, y_train, epochs=epoch, batch_size=10000, verbose=1,validation_data=[X_train_val,y_train_val],callbacks = [es] )

history_dict = model_history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, epoch + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

plt.title   ('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


y_pred=pd.DataFrame(model.predict(X_test))

for i in y_pred.columns:
    y_pred[i]= y_pred[i].apply(lambda x: 0 if x < 0.5 else i)
y_pred['agg'] = y_pred.sum(axis=1)
cm = confusion_matrix(working_data_set.iloc[-24000:]["Move"],y_pred['agg'])
f = sns.heatmap(cm, annot=True, fmt='d')
f.set(xlabel='Y_Pred', ylabel='Y_True')
print(metrics.classification_report(working_data_set.iloc[-24000:]["Move"],y_pred['agg'],digits=3 ))



from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
y_train_val = to_categorical(y_train_val)
y_test = to_categorical(y_test)


pd.DataFrame(to_categorical(y_train))
y_train