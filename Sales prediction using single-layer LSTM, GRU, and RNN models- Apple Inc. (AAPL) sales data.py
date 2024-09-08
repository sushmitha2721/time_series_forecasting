#!/usr/bin/env python
# coding: utf-8

# In[133]:


""" libraries """
import numpy as AAPL__N
import seaborn as S__BORN
import pandas as AAPL__P
import matplotlib.pyplot as AAPL__M


# In[134]:


# Importing dataset

time_data_APPL = AAPL__P.read_csv('AAPL.csv')


# In[135]:


time_data_APPL


# In[136]:


# Checking for null values

time_data_APPL.isnull().sum()


# In[137]:


time_data_APPL['Date'] = AAPL__P.to_datetime(time_data_APPL['Date'])
# Set 'Date' column as the index
time_data_APPL.set_index('Date', inplace=True)


# In[138]:


# Vislualizing the data

time_data_APPL['Close'].plot(figsize=(6,6), color= 'red')
AAPL__M.xlabel('Year')
AAPL__M.ylabel('Close-value')
AAPL__M.title('Yearwise close value- line plot')
AAPL__M.show()


# In[139]:


# Generate a complete range of business dates
all_business_dates = AAPL__P.date_range(start=time_data_APPL.index.min(), end=time_data_APPL.index.max(), freq='B')

# Identify missing dates
missing_dates = all_business_dates.difference(time_data_APPL.index)


# In[140]:


# Create a DataFrame to store the missing dates and their days
missing_dates_df = AAPL__P.DataFrame(missing_dates, columns=['Date'])
missing_dates_df['Day'] = missing_dates_df['Date'].dt.day_name()
missing_dates_df['Month'] = missing_dates_df['Date'].dt.month_name()
missing_dates_df['Year'] = missing_dates_df['Date'].dt.year


# In[141]:


# Print the missing dates along with days
print(missing_dates_df)


# In[142]:


# 1. Count of missing dates by day of the week
AAPL__M.figure(figsize=(10, 6))
S__BORN.countplot(x='Day', data=missing_dates_df, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])
AAPL__M.title('Count of Missing Dates by Day of the Week')
AAPL__M.xlabel('Day of the Week')
AAPL__M.ylabel('Count of Missing Dates')
AAPL__M.show()


# In[143]:


import pandas_market_calendars as mcal


# In[144]:


# Get NYSE market calendar
nyse = mcal.get_calendar('NYSE')

# Get NYSE holidays within the date range
market_holidays = nyse.holidays().holidays

# Categorize missing dates
missing_dates_df['Category'] = 'Random Gap'
missing_dates_df.loc[missing_dates_df['Date'].isin(market_holidays), 'Category'] = 'Holiday'
missing_dates_df.loc[missing_dates_df['Date'].dt.dayofweek >= 5, 'Category'] = 'Weekend'

# Impute missing values using forward fill
time_data_APPL = time_data_APPL.asfreq('B', method='ffill')

# Optionally, add an indicator column for missing days
time_data_APPL['Missing'] = 0
time_data_APPL.loc[missing_dates_df['Date'], 'Missing'] = 1


# In[145]:


time_data_APPL


# In[146]:


from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd


# In[147]:


from statsmodels.tsa.seasonal import seasonal_decompose

# Filter data for the last 10 years
last_10_years = time_data_APPL[time_data_APPL.index >= pd.Timestamp('2014-06-03')]

# Ensure the index is a DatetimeIndex with a frequency
last_10_years = last_10_years.asfreq('B')  # 'B' stands for business day frequency

# Perform seasonal decomposition
results = seasonal_decompose(last_10_years['Close'], model='additive', period=252)

# Plot the results
results.plot()
AAPL__M.show()


# `* In recen years, sale value of AAPL is incresed extremely.`

# ##  Neural_Network_Models
# ```
# 
# 

# In[148]:


time_data_APPL.reset_index(inplace=True)
time_data_APPL = time_data_APPL[['Close', 'Date']]

time_data_APPL


# In[149]:


## normalization
from sklearn.preprocessing import MinMaxScaler as AAPL__G

val_Close = time_data_APPL['Close'].values
val_Close = val_Close.reshape((len(val_Close), 1))

scale_Close = AAPL__G(feature_range=(0, 1))
scale_Close = scale_Close.fit(val_Close)
nor_close= scale_Close.transform(val_Close)
nor_close


# In[150]:


time_data_APPL['Close']= AAPL__P.DataFrame(nor_close)
time_data_APPL # normalized dataframe


# In[151]:


def defnition_of_function_Model(AAPL__sequ, AAPL__n_ste):
  variable_x_AAPL, variable_y_AAPL = [], []

  for FOR in range(len(AAPL__sequ)):
    Addition = FOR + AAPL__n_ste
    if Addition > len(AAPL__sequ)-1:
      break
    x_AAPL, y_AAPL = AAPL__sequ[FOR:Addition], AAPL__sequ[Addition]
    variable_x_AAPL.append(x_AAPL)
    variable_y_AAPL.append(y_AAPL)
  return AAPL__N.array(variable_x_AAPL), AAPL__N.array(variable_y_AAPL)


# In[152]:


## training & testing data separation using n_step=30 ##
AAPL__n_ste = 30

variable_x_AAPL_TN, variable_y_AAPL_TN = defnition_of_function_Model(time_data_APPL['Close'][:-500].tolist(), AAPL__n_ste)
variable_x_AAPL_TS, variable_y_AAPL_TS = defnition_of_function_Model(time_data_APPL['Close'][-500:].tolist(), AAPL__n_ste)
for total in range(len(variable_x_AAPL_TS)):
  print(variable_x_AAPL_TS[total], variable_y_AAPL_TS[total])


# In[180]:


print("X_train: ", variable_x_AAPL_TN.shape)
print("X_test: ", variable_x_AAPL_TS.shape)
time_data_APPL_DT=(time_data_APPL['Date'][-500:-30].tolist())


# ##  Single-layer LSTM
# ```
# 
# 

# In[170]:


from tensorflow.keras.models import Sequential as AAPL__NS
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler as AAPL__NM
from tensorflow.keras.layers import LSTM as AAPL__NLS
from tensorflow.keras.layers import Dense as AAPL__NDE
from sklearn import metrics as AAPL__NMT
from keras.callbacks import EarlyStopping


# In[171]:


AAPL__NS_MD =AAPL__NS()

AAPL__NS_MD.add(AAPL__NLS(500,input_shape=(variable_x_AAPL_TN.shape[1],1),activation="relu"))
AAPL__NS_MD.add(Dropout(0.2))
AAPL__NS_MD.add(AAPL__NDE(1))

AAPL__NS_MD.compile(loss="mean_absolute_error",optimizer="adamax")

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
#mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

# fit model
HY = AAPL__NS_MD.fit(variable_x_AAPL_TN, variable_y_AAPL_TN, validation_split=0.3, epochs=50, batch_size=100, callbacks=[es])



# In[172]:


AAPL__NS_MD.summary()


# In[181]:


#""""""""""""" Evaluating the model """"""""""""
# Plot of Loss data
AAPL__M.figure()
AAPL__M.xlabel('Epoch')
AAPL__M.ylabel('Loss')
AAPL__M.plot(HY.history['loss'], color='tan')
AAPL__M.legend(['Train__val', 'Test__val'], loc='upper right')
AAPL__M.title('loss')
AAPL__M.show()


# In[182]:


# Evaluating the Model
variable_y_AAPL_PRD = AAPL__NS_MD.predict(variable_x_AAPL_TS)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("Metric - R2 Score:", r2_score(variable_y_AAPL_TS, variable_y_AAPL_PRD) * 100)
print("Metric - Mean Absolute Error:", mean_absolute_error(variable_y_AAPL_TS, variable_y_AAPL_PRD))
print("Metric - Mean Squared Error:", mean_squared_error(variable_y_AAPL_TS, variable_y_AAPL_PRD))


# In[183]:


# Creating DataFrame with Actual and Predicted Outputs
import pandas as pd

Predc_APPL = pd.DataFrame()


Predc_APPL['test_val'] = variable_y_AAPL_TS.tolist()
Predc_APPL['pred_val'] = variable_y_AAPL_PRD.tolist()
Predc_APPL['Date_entry'] = time_data_APPL_DT
Predc_APPL = Predc_APPL.set_index("Date_entry")

# Last 7 Days Prediction vs Actual Value
print('Last 7 Days Actual Values:\n', Predc_APPL["test_val"][-7:])
print('Last 7 Days Predicted Values:\n', [x[0] for x in Predc_APPL["pred_val"][-7:]])

# Plotting the Last 7 Days' Predictions vs Actual Values
plt.figure(figsize=(12, 6))
plt.xlabel("Date")
plt.ylabel("Values")
plt.title("Last 7 Days Prediction vs Actual Output")

# Extracting actual and predicted values
actual_values = Predc_APPL["test_val"][-7:]
predicted_values = [x[0] for x in Predc_APPL["pred_val"][-7:]]  # Extracting the float value from the list

# Dates for the last 7 days
last_7_dates = Predc_APPL.index[-7:]

# Plot actual values
plt.plot(last_7_dates, actual_values, label="Actual Value", color='grey', marker='o')

# Plot predicted values
plt.plot(last_7_dates, predicted_values, label="Prediction", color='violet', marker='o')

# Improve x-axis labels
plt.xticks(rotation=45, ha='right')

# Add gridlines for easier reading
plt.grid(True)

# Add legend
plt.legend(loc="lower right")


# In[191]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def build_model(learning_rate, n_neurons, n_timesteps):
    model = Sequential()
    model.add(LSTM(n_neurons, input_shape=(n_timesteps, 1), return_sequences=False))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model



# In[192]:


import itertools

learning_rate = [0.01, 0.001, 0.005]
n_neurons = [200, 300, 500]
n_timesteps = [30, 60, 90]
n_epochs = [50, 20, 30]

hyper_para = (learning_rate, n_neurons, n_timesteps, n_epochs)
hyper_para_grid = list(itertools.product(*hyper_para))


# In[ ]:


from sklearn.model_selection import train_test_split
import numpy as np

def fit_lstm_grid(batch_size):
    best_params = None
    min_val_loss = np.inf
    
    for lr, neurons, timesteps, epochs in hyper_para_grid:
        print(f'Training with lr={lr}, neurons={neurons}, timesteps={timesteps}, epochs={epochs}')
        
        # Build and compile the model
        model = build_model(lr, neurons, timesteps)
        
        # Prepare training data
        X_train, y_train = defnition_of_function_Model(time_data_APPL['Close'][:-500].tolist(), timesteps)
        X_test, y_test = defnition_of_function_Model(time_data_APPL['Close'][-500:].tolist(), timesteps)
        
        # Fit the model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
        
        # Evaluate the model
        val_loss = history.history['val_loss'][-1]
        print(f'Validation Loss: {val_loss}')
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_params = (lr, neurons, timesteps, epochs)
            print(f'New best parameters: {best_params} with loss {min_val_loss}')
    
    return best_params

def run_lstm_grid():
    batch_size = 32  # You can adjust this as needed
    best_params = fit_lstm_grid(batch_size)
    return best_params

# Execute hyperparameter tuning
best_params = run_lstm_grid()
print(f'Best parameters found: {best_params}')


# In[ ]:


early_stopping_rounds = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=200)


# In[ ]:


# Objective function
def run(trial):
    space = {        
        'optimizer':  trial.suggest_categorical('optimizer', [optimizers.Adam, optimizers.Adagrad, optimizers.Adamax, optimizers.Nadam, optimizers.RMSprop, optimizers.SGD]),
        'units': trial.suggest_int('units', 50, 4000, 50),
        'epochs': trial.suggest_int('epochs', 100, 5000, 50),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256, 512]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.6),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 8e-1)
    }

    model = Sequential()
    # Adding the LSTM layer and some Dropout regularization
    model.add(LSTM(units=space['units'], input_shape=(variable_x_AAPL_TN.shape[0], variable_x_AAPL_TN.shape[1])))
    model.add(Dropout(rate=space['dropout_rate']))
    # Adding the output layer
    model.add(Dense(units=1))

    # Compiling the RNN
    model.compile(loss='mean_squared_error', optimizer=space['optimizer'](learning_rate=space['learning_rate']), metrics=['accuracy'])

    model.fit(variable_x_AAPL_TN, variable_y_AAPL_TN, epochs=space['epochs'], validation_data=(x_val, y_val), batch_size=space['batch_size'], callbacks=[early_stopping_rounds], verbose=0)

    yhat_train = np.squeeze(model.predict(x_train))
    yhat_val = np.squeeze(model.predict(x_val))

    yhat_train = scaler.inverse_transform(prep_data_transform(yhat_train, X_train.shape[1]))[:,0]
    yhat_val = scaler.inverse_transform(prep_data_transform(yhat_val, X_val.shape[1]))[:,0]

    nan_check = [np.isnan(yhat_train).any(), np.isnan(yhat_val).any()]
    print(nan_check)
    if True in nan_check:
        rmse = 99999
    else:
        rmse = mean_squared_error(Y_val, yhat_val, squared=False)
    return rmse


# In[ ]:





# In[ ]:





# In[ ]:





# ##  Single-layer GRU
# ```
# 
# 

# In[16]:


from tensorflow.keras.layers import GRU as AAPL__NGU


# In[17]:


AAPL__NS_MD =AAPL__NS()

AAPL__NS_MD.add(AAPL__NGU(500,input_shape=(None,1),activation="relu"))
AAPL__NS_MD.add(AAPL__NDE(1))

AAPL__NS_MD.compile(loss="mean_absolute_error",optimizer="adamax")

HY = AAPL__NS_MD.fit(variable_x_AAPL_TN, variable_y_AAPL_TN,
                            epochs=10, batch_size=100)


# In[18]:


#""""""""""""" Evaluating the model """"""""""""
# Plot of Loss data
AAPL__M.figure()
AAPL__M.xlabel('Epoch')
AAPL__M.ylabel('Loss')
AAPL__M.plot(HY.history['loss'], color='tan')
AAPL__M.legend(['Train__val', 'Test__val'], loc='upper right')
AAPL__M.title('loss')
AAPL__M.show()


variable_y_AAPL_PRD = AAPL__NS_MD.predict(variable_x_AAPL_TS)
print(" Metric-R2 score:", AAPL__NMT.r2_score(variable_y_AAPL_PRD, variable_y_AAPL_TS)*100)

print("Metric-mean_absolute_error:", AAPL__NMT.mean_absolute_error(variable_y_AAPL_PRD, variable_y_AAPL_TS))

print("Metric mean_squared_error:", AAPL__NMT.mean_squared_error(variable_y_AAPL_PRD, variable_y_AAPL_TS))

#Creating dataframe With actual output and predicted output
Predc_APPL= AAPL__P.DataFrame()
Predc_APPL['test_val']=variable_y_AAPL_TS.tolist()
Predc_APPL['pred_val']=variable_y_AAPL_PRD.tolist()
Predc_APPL['Date_entry']=time_data_APPL_DT
Predc_APPL=Predc_APPL.set_index("Date_entry")

##############"""last 7 Days prediction vs Actual value"""
print('Predicted value for next 7 days,\n', variable_y_AAPL_PRD[-7:])
AAPL__M.figure(figsize=(10, 5))
AAPL__M.xlabel("Date")
AAPL__M.ylabel("Values")
AAPL__M.title("last 7 Days prediction vs Actual output")
AAPL__M.xticks(rotation=90)
# plotting the line graph for time series
AAPL__M.plot(Predc_APPL["test_val"][-7:], label="Actual value", color= 'grey')
AAPL__M.plot(Predc_APPL["pred_val"][-7:].tolist(), label=" prediction", color='violet')
AAPL__M.legend(loc="lower right")
AAPL__M.show()


# ##  Single-layer RNN
# ```
# 
# 

# In[19]:


from tensorflow.keras.layers import SimpleRNN as AAPL__NSR


# In[20]:


AAPL__NS_MD =AAPL__NS()

AAPL__NS_MD.add(AAPL__NSR(500,input_shape=(None,1),activation="relu"))
AAPL__NS_MD.add(AAPL__NDE(1))

AAPL__NS_MD.compile(loss="mean_absolute_error",optimizer="adamax")

HY = AAPL__NS_MD.fit(variable_x_AAPL_TN, variable_y_AAPL_TN,
                            epochs=10, batch_size=100)


# In[21]:


#""""""""""""" Evaluating the model """"""""""""
# Plot of Loss data
AAPL__M.figure()
AAPL__M.xlabel('Epoch')
AAPL__M.ylabel('Loss')
AAPL__M.plot(HY.history['loss'], color='tan')
AAPL__M.legend(['Train__val', 'Test__val'], loc='upper right')
AAPL__M.title('loss')
AAPL__M.show()


variable_y_AAPL_PRD = AAPL__NS_MD.predict(variable_x_AAPL_TS)
print(" Metric-R2 score:", AAPL__NMT.r2_score(variable_y_AAPL_PRD, variable_y_AAPL_TS)*100)

print("Metric-mean_absolute_error:", AAPL__NMT.mean_absolute_error(variable_y_AAPL_PRD, variable_y_AAPL_TS))

print("Metric mean_squared_error:", AAPL__NMT.mean_squared_error(variable_y_AAPL_PRD, variable_y_AAPL_TS))

#Creating dataframe With actual output and predicted output
Predc_APPL= AAPL__P.DataFrame()
Predc_APPL['test_val']=variable_y_AAPL_TS.tolist()
Predc_APPL['pred_val']=variable_y_AAPL_PRD.tolist()
Predc_APPL['Date_entry']=time_data_APPL_DT
Predc_APPL=Predc_APPL.set_index("Date_entry")

##############"""last 7 Days prediction vs Actual value"""
print('Predicted value for next 7 days,\n', variable_y_AAPL_PRD[-7:])
AAPL__M.figure(figsize=(10, 5))
AAPL__M.xlabel("Date")
AAPL__M.ylabel("Values")
AAPL__M.title("last 7 Days prediction vs Actual output")
AAPL__M.xticks(rotation=90)
# plotting the line graph for time series
AAPL__M.plot(Predc_APPL["test_val"][-7:], label="Actual value", color= 'grey')
AAPL__M.plot(Predc_APPL["pred_val"][-7:].tolist(), label=" prediction", color='violet')
AAPL__M.legend(loc="lower right")
AAPL__M.show()


# ##  stacked LSTM
# ```
# 
# 

# In[22]:


AAPL__NS_MD =AAPL__NS()

AAPL__NS_MD.add(AAPL__NLS(500,input_shape=(None,1),activation="relu", return_sequences = True))
AAPL__NS_MD.add(AAPL__NLS(100,return_sequences = False))
AAPL__NS_MD.add(AAPL__NDE(1))

AAPL__NS_MD.compile(loss="mean_absolute_error",optimizer="adamax")

HY = AAPL__NS_MD.fit(variable_x_AAPL_TN, variable_y_AAPL_TN,
                            epochs=10, batch_size=100)


# In[23]:


#""""""""""""" Evaluating the model """"""""""""
# Plot of Loss data
AAPL__M.figure()
AAPL__M.xlabel('Epoch')
AAPL__M.ylabel('Loss')
AAPL__M.plot(HY.history['loss'], color='tan')
AAPL__M.legend(['Train__val', 'Test__val'], loc='upper right')
AAPL__M.title('loss')
AAPL__M.show()


variable_y_AAPL_PRD = AAPL__NS_MD.predict(variable_x_AAPL_TS)
print(" Metric-R2 score:", AAPL__NMT.r2_score(variable_y_AAPL_PRD, variable_y_AAPL_TS)*100)

print("Metric-mean_absolute_error:", AAPL__NMT.mean_absolute_error(variable_y_AAPL_PRD, variable_y_AAPL_TS))

print("Metric mean_squared_error:", AAPL__NMT.mean_squared_error(variable_y_AAPL_PRD, variable_y_AAPL_TS))

#Creating dataframe With actual output and predicted output
Predc_APPL= AAPL__P.DataFrame()
Predc_APPL['test_val']=variable_y_AAPL_TS.tolist()
Predc_APPL['pred_val']=variable_y_AAPL_PRD.tolist()
Predc_APPL['Date_entry']=time_data_APPL_DT
Predc_APPL=Predc_APPL.set_index("Date_entry")

##############"""last 7 Days prediction vs Actual value"""
print('Predicted value for next 7 days,\n', variable_y_AAPL_PRD[-7:])
AAPL__M.figure(figsize=(10, 5))
AAPL__M.xlabel("Date")
AAPL__M.ylabel("Values")
AAPL__M.title("last 7 Days prediction vs Actual output")
AAPL__M.xticks(rotation=90)
# plotting the line graph for time series
AAPL__M.plot(Predc_APPL["test_val"][-7:], label="Actual value", color= 'grey')
AAPL__M.plot(Predc_APPL["pred_val"][-7:].tolist(), label=" prediction", color='violet')
AAPL__M.legend(loc="lower right")
AAPL__M.show()

