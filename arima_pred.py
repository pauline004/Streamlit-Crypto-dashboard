import pandas as pd
import numpy as np
import plotly.graph_objects as go 
import plotly.express as px
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
from sklearn.preprocessing import MinMaxScaler
#importing packages for the prediction of time-series data
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats
from itertools import product
import warnings
warnings.filterwarnings('ignore')


def predict_arima(df):
    new = df.reset_index()
    ts = new['close']      
    
    ts_log = np.log(ts)
    
    ts_log_diff = ts_log - ts_log.shift()
    ts_log_diff.dropna(inplace=True)

    qs = range(0, 3)
    ps = range(0, 3)
    d=1
    parameters = product(ps, qs)
    parameters_list = list(parameters)
    len(parameters_list)

    # Model Selection
    results = []
    best_aic = float("inf")
    warnings.filterwarnings('ignore')
    for param in parameters_list:
        try:
            model = SARIMAX(ts_log, order=(param[0], d, param[1])).fit()
        except ValueError:
            print('bad parameter combination:', param)
            continue
        aic = model.aic
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])


    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']

    size = int(len(ts_log)-100)
    # Divide into train and test
    train_arima, test_arima = ts_log[0:size], ts_log[size:len(ts_log)]
    history = [x for x in train_arima]
    predictions = list()
    originals = list()
    error_list = list()
   
    print('Printing Predicted vs Expected Values...')
    print('\n')

    for t in range(len(test_arima)):
        model = ARIMA(history, order=(0, 1, 0))
        model_fit = model.fit()

        output = model_fit.forecast()

        pred_value = output[0]


        original_value = test_arima[t+size]

        history.append(original_value)

        pred_value = np.exp(pred_value)

        original_value = np.exp(original_value)

        # Calculating the error
        error = ((abs(pred_value - original_value)) / original_value) * 100
        error_list.append(error)

        predictions.append(float(pred_value))
        originals.append(float(original_value))

    

    plt.figure(figsize=(10, 6))
    test_day = [t for t in range(len(test_arima))]
    labels={'Orginal','Predicted'}
    plt.plot(test_day, predictions, color= 'firebrick')
    plt.plot(test_day, originals, color = 'midnightblue')
    plt.title('Expected Vs Predicted Views Forecasting')
    plt.xlabel('Day')
    plt.ylabel('Closing Price')
    plt.legend(labels)
    plt.show()
    predict = pd.DataFrame(predictions,index = test_arima.index,columns=['Prediction'])
    new['Prediction']=predict

    pred= model_fit.forecast(steps=1)
    next_value=list(np.exp(pred))

    return predictions, originals , plt, next_value 

