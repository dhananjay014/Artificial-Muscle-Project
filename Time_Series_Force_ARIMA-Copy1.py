
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.models import Sequential
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# In[3]:


# df = pd.read_excel('130N_Cycles_1-47.xlsx',sheet_name='Specimen_RawData_1',skiprows=[0])
# df.columns = ['time','load']


# In[4]:


# df.plot(y='load',x='time',figsize=(20,10))


# In[5]:


# from pandas.tools.plotting import autocorrelation_plot


# In[6]:


# autocorrelation_plot(df['load'])


# In[7]:


# series = df['load']
# print(series)


# In[8]:


# series.index = pd.DatetimeIndex(series.index)
# print(series)
# series.index+=1
# print(series)


# In[9]:


# series.index = pd.to_datetime(df['time'],unit = 's')


# In[10]:


# model = ARIMA(series,order=(5,1,0))


# In[11]:


# model_fit = model.fit(disp=0)


# In[12]:


# print(model_fit.summary())


# In[13]:


# residuals = pd.DataFrame(model_fit.resid)
# residuals.plot(figsize=(20,10))a
# plt.figure(figsize=(20,10))
# plt.show()
# residuals.plot(kind='kde',figsize=(20,10))
# plt.show()
# print(residuals.describe())


# ### Training and Forecasting

# In[14]:


# df = pd.read_excel('130N_Cycles_1-47.xlsx',sheetname='Specimen_RawData_1',skiprows=[0])
# df.columns = ['time','load']


# In[15]:


# series = df['load']
# series.index = pd.to_datetime(df['time'],unit='s')


# In[16]:


# X = series.values


# In[17]:


# size = int(len(X) * 0.999)


# In[18]:


# train, test = X[0:size], X[size:len(X)]
# history = [x  for x in train]
# predictions = list()


# In[19]:


# for t in range(len(test)):
#     model = ARIMA(history,order=(5,1,0))
#     model_fit = model.fit(disp=0)
#     output = model_fit.forecast()
#     yhat = output[0]
#     predictions.append(yhat)
#     obs = test[t]
#     history.append(obs)
#     print('t = {}, predicted={}, expected = {}'.format(t, yhat, obs))


# In[21]:


# error = mean_squared_error(test,predictions)
# print('Test MSE:{}'.format(error))
# plt.plot(test)
# plt.plot(predictions, color='red')
# plt.show()


# ### Forecast using predictions

# In[20]:


df = pd.read_excel('130N_Cycles_1-47.xlsx',sheetname='Specimen_RawData_1',skiprows=[0])
df.columns = ['time','load']


# In[11]:


series = df['load']
series.index = pd.to_datetime(df['time'],unit='s')


# In[12]:


X = series.values
size = int(len(X) * 0.999)
train, test = X[0:size], X[size:len(X)]
history = [x  for x in train]
predictions = list()


# In[14]:


for t in range(len(test)):
    model = ARIMA(history,order=(5,2,5))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = yhat
    history.append(obs)
    print('t = {}, predicted={}, expected = {}'.format(t, yhat, test[t]))


# In[ ]:


error = mean_squared_error(test,predictions)
print('Test MSE:{}'.format(error))
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


# ### Forecasting for unseen data

# In[2]:


df = pd.read_excel('130N_Cycles_1-47.xlsx',sheetname='Specimen_RawData_1',skiprows=[0])
df.columns = ['time','load']


# In[3]:


series = df['load']
series.index = pd.to_datetime(df['time'],unit='s')


# In[4]:


X = series.values
size = int(len(X) * 1.0)
train, test = X[0:size], X[size:len(X)]
history = [x  for x in train]
predictions = list()


# In[ ]:


for t in range(10000):
    model = ARIMA(history,order=(5,2,5))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = yhat
    history.append(obs)
    print('t = {}, predicted={}'.format(t, yhat))


# In[ ]:


plt.plot(test)
plt.show()

