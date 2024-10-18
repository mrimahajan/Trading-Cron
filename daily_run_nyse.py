# import logging
# logging.disable(logging.CRITICAL)
import  scipy.signal.signaltools

def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

scipy.signal.signaltools._centered = _centered
import os
import pandas as pd
from django.contrib.staticfiles.storage import staticfiles_storage
import django
import numpy as np 
import yfinance as yf
from datetime import date
import datetime
from scipy.stats import norm
import pickle
import pytz
# from decimal import Decimal

import sys
import warnings
# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')
from keras.models import load_model
from keras.optimizers import Adam

Name="NYSE"
Extension=""
Currency="USD"
Country="USA"
Frac=0

def get_data(symbol,start,end):
    try:
        df = yf.download(symbol+Extension,start=str(start),end=str(end))
        df['Symbol'] = symbol
        df = df[['Symbol','Open','High','Low','Close','Adj Close','Volume']]
        if(df.shape[0] > 0):
            df.to_csv(staticfiles_storage.path(f'{Name}/Data/{symbol}.csv'))
    except:
        print(f'Data not found for {symbol}')



def create_variables(symbol):
    df = pd.read_csv(staticfiles_storage.path(f'{Name}/Data/{symbol}.csv'),header=0)
    df = df.sort_values(by='Date')
    df = df[(df['Volume']!=0) & (df['Volume'].notnull())]
    df['future_return'] = np.log(df['Adj Close'].shift(-1)/df['Adj Close'])
    df['lift'] = df['future_return'].apply(lambda x: 1 if x >0 else 0)
    df['returns'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
    # Calculate True Range (TR)
    df['TR1'] = abs(df['High'] - df['Low'])
    df['TR2'] = abs(df['High'] - df['Adj Close'].shift())
    df['TR3'] = abs(df['Low'] - df['Adj Close'].shift())
    df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
    df.drop(['TR1', 'TR2', 'TR3'],axis=1, inplace=True)
    # Calculate Directional Movement (DM)
    df['PDM'] = df['High'] - df['High'].shift()
    df['NDM'] = df['Low'].shift() - df['Low']
    df.loc[df['PDM'] < 0, 'PDM'] = 0
    df.loc[df['NDM'] < 0, 'NDM'] = 0
    df['Price Change'] = df['Adj Close'].diff()
    df['Gain'] = df['Price Change'].apply(lambda x: x if x > 0 else 0)
    df['Loss'] = df['Price Change'].apply(lambda x: abs(x) if x < 0 else 0)
    df['Volume Change'] = df['Volume'].diff()
    df['Volume Gain'] = df['Volume Change'].apply(lambda x: x if x > 0 else 0)
    df['Volume Loss'] = df['Volume Change'].apply(lambda x: abs(x) if x < 0 else 0)

    for i in range(1,26):
        df[f'log_return_{i}'] = np.log(df['Adj Close'].shift(i-1)/df['Adj Close'].shift(i))
        df[f'Open_Chg{i}'] = np.log(df['Open'].shift(i-1)/df['Open'].shift(i))
        df[f'High_Chg{i}'] = np.log(df['High'].shift(i-1)/df['High'].shift(i))
        df[f'Low_Chg{i}'] = np.log(df['Low'].shift(i-1)/df['Low'].shift(i))
        df[f'Range_Chg{i}'] = np.log((df['High'].shift(i-1)-df['Low'].shift(i-1))/(df['High'].shift(i)-df['Low'].shift(i)))
        df[f'Volume_Chg{i}'] = np.log(df['Volume'].shift(i-1)/df['Volume'].shift(i))
    for j in range(5,30,5):
        df[f'Voltality{j}'] = df['returns'].rolling(window=j).std()
        df[f'Volume_chg_Voltality{j}'] = df['Volume_Chg1'].rolling(window=j).std()
        df[f'MA{j}'] = df['Adj Close'].rolling(window=j).mean()
        df[f'MAV{j}'] = df['Volume'].rolling(window=j).mean()
        df[f'High{j}'] = df['Adj Close'].rolling(window=j).max()
        df[f'Low{j}'] = df['Adj Close'].rolling(window=j).min()
        df[f'MA{j}_ratio'] = np.log(df['Adj Close']/df[f'MA{j}'])
        df[f'MAV{j}_ratio'] = np.log(df['Volume']/df[f'MAV{j}'])
        df[f'High{j}_ratio'] = np.log(df['Adj Close']/df[f'High{j}'])
        df[f'Low{j}_ratio'] = np.log(df[f'Low{j}']/df['Adj Close'])
        df[f'Price_Voltality{j}'] = df['Adj Close'].rolling(window=j).std()
        df[f'Volume_Voltality{j}'] = df['Volume'].rolling(window=j).std()
        df[f'Bollinger{j}_ratio'] = (df['Adj Close']-df[f'MA{j}']+2*df[f'Price_Voltality{j}'])/(4*df[f'Price_Voltality{j}'])
        df[f'Bollinger{j}_volume_ratio'] = (df['Volume']-df[f'MAV{j}']+2*df[f'Volume_Voltality{j}'])/(4*df[f'Volume_Voltality{j}'])
        df['TR_sum'] = df['TR'].rolling(j).sum()
        df['PDM_sum'] = df['PDM'].rolling(j).sum()
        df['NDM_sum'] = df['NDM'].rolling(j).sum()
        df['PDI'] = (df['PDM_sum'] / df['TR_sum'])
        df['NDI'] = (df['NDM_sum'] / df['TR_sum'])
        df['DX'] = abs(df['PDI'] - df['NDI']) / (df['PDI'] + df['NDI'])
        df[f'ADX{j}'] = df['DX'].rolling(j).mean()
        df['Average Gain'] = df['Gain'].rolling(j).mean()
        df['Average Loss'] = df['Loss'].rolling(j).mean()
        df['RS'] = df['Average Gain'] / df['Average Loss']
        df[f'RSI{j}'] = 1 - (1 / (1 + df['RS']))
        df['Average Volume Gain'] = df['Volume Gain'].rolling(j).mean()
        df['Average Volume Loss'] = df['Volume Loss'].rolling(j).mean()
        df['RSV'] = df['Average Volume Gain']/df['Average Volume Loss']
        df[f'RSIV{j}'] = 1 - (1 / (1 + df['RSV']))
    df.drop(['TR','PDM','NDM','Price Change','Gain','Loss','Volume Change','Volume Gain','Volume Loss',
             'TR_sum','PDM_sum','NDM_sum','PDI','NDI','DX','Average Gain',
             'Average Loss','RS','Average Volume Gain','Average Volume Loss','RSV'],axis=1,inplace=True)
    df.dropna(inplace=True)
    df.to_csv(staticfiles_storage.path(f'{Name}/Train/{symbol}.csv'),index=False)


def pred_variables(symbol,last_trade_date):
    df = pd.read_csv(staticfiles_storage.path(f'{Name}/Data/{symbol}.csv'),header=0)
    df = df.sort_values(by='Date')
    df = df[(df['Volume']!=0) & (df['Volume'].notnull())]
    df['future_return'] = np.log(df['Adj Close'].shift(-1)/df['Adj Close'])
    df['lift'] = df['future_return'].apply(lambda x: 1 if x >0 else 0)
    df['returns'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))
    # Calculate True Range (TR)
    df['TR1'] = abs(df['High'] - df['Low'])
    df['TR2'] = abs(df['High'] - df['Adj Close'].shift())
    df['TR3'] = abs(df['Low'] - df['Adj Close'].shift())
    df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
    df.drop(['TR1', 'TR2', 'TR3'],axis=1, inplace=True)
    # Calculate Directional Movement (DM)
    df['PDM'] = df['High'] - df['High'].shift()
    df['NDM'] = df['Low'].shift() - df['Low']
    df.loc[df['PDM'] < 0, 'PDM'] = 0
    df.loc[df['NDM'] < 0, 'NDM'] = 0
    df['Price Change'] = df['Adj Close'].diff()
    df['Gain'] = df['Price Change'].apply(lambda x: x if x > 0 else 0)
    df['Loss'] = df['Price Change'].apply(lambda x: abs(x) if x < 0 else 0)
    df['Volume Change'] = df['Volume'].diff()
    df['Volume Gain'] = df['Volume Change'].apply(lambda x: x if x > 0 else 0)
    df['Volume Loss'] = df['Volume Change'].apply(lambda x: abs(x) if x < 0 else 0)

    for i in range(1,26):
        df[f'log_return_{i}'] = np.log(df['Adj Close'].shift(i-1)/df['Adj Close'].shift(i))
        df[f'Open_Chg{i}'] = np.log(df['Open'].shift(i-1)/df['Open'].shift(i))
        df[f'High_Chg{i}'] = np.log(df['High'].shift(i-1)/df['High'].shift(i))
        df[f'Low_Chg{i}'] = np.log(df['Low'].shift(i-1)/df['Low'].shift(i))
        df[f'Range_Chg{i}'] = np.log((df['High'].shift(i-1)-df['Low'].shift(i-1))/(df['High'].shift(i)-df['Low'].shift(i)))
        df[f'Volume_Chg{i}'] = np.log(df['Volume'].shift(i-1)/df['Volume'].shift(i))
    for j in range(5,30,5):
        df[f'Voltality{j}'] = df['returns'].rolling(window=j).std()
        df[f'Volume_chg_Voltality{j}'] = df['Volume_Chg1'].rolling(window=j).std()
        df[f'MA{j}'] = df['Adj Close'].rolling(window=j).mean()
        df[f'MAV{j}'] = df['Volume'].rolling(window=j).mean()
        df[f'High{j}'] = df['Adj Close'].rolling(window=j).max()
        df[f'Low{j}'] = df['Adj Close'].rolling(window=j).min()
        df[f'MA{j}_ratio'] = np.log(df['Adj Close']/df[f'MA{j}'])
        df[f'MAV{j}_ratio'] = np.log(df['Volume']/df[f'MAV{j}'])
        df[f'High{j}_ratio'] = np.log(df['Adj Close']/df[f'High{j}'])
        df[f'Low{j}_ratio'] = np.log(df[f'Low{j}']/df['Adj Close'])
        df[f'Price_Voltality{j}'] = df['Adj Close'].rolling(window=j).std()
        df[f'Volume_Voltality{j}'] = df['Volume'].rolling(window=j).std()
        df[f'Bollinger{j}_ratio'] = (df['Adj Close']-df[f'MA{j}']+2*df[f'Price_Voltality{j}'])/(4*df[f'Price_Voltality{j}'])
        df[f'Bollinger{j}_volume_ratio'] = (df['Volume']-df[f'MAV{j}']+2*df[f'Volume_Voltality{j}'])/(4*df[f'Volume_Voltality{j}'])
        df['TR_sum'] = df['TR'].rolling(j).sum()
        df['PDM_sum'] = df['PDM'].rolling(j).sum()
        df['NDM_sum'] = df['NDM'].rolling(j).sum()
        df['PDI'] = (df['PDM_sum'] / df['TR_sum'])
        df['NDI'] = (df['NDM_sum'] / df['TR_sum'])
        df['DX'] = abs(df['PDI'] - df['NDI']) / (df['PDI'] + df['NDI'])
        df[f'ADX{j}'] = df['DX'].rolling(j).mean()
        df['Average Gain'] = df['Gain'].rolling(j).mean()
        df['Average Loss'] = df['Loss'].rolling(j).mean()
        df['RS'] = df['Average Gain'] / df['Average Loss']
        df[f'RSI{j}'] = 1 - (1 / (1 + df['RS']))
        df['Average Volume Gain'] = df['Volume Gain'].rolling(j).mean()
        df['Average Volume Loss'] = df['Volume Loss'].rolling(j).mean()
        df['RSV'] = df['Average Volume Gain']/df['Average Volume Loss']
        df[f'RSIV{j}'] = 1 - (1 / (1 + df['RSV']))
    df.drop(['TR','PDM','NDM','Price Change','Gain','Loss','Volume Change','Volume Gain','Volume Loss',
             'TR_sum','PDM_sum','NDM_sum','PDI','NDI','DX','Average Gain',
             'Average Loss','RS','Average Volume Gain','Average Volume Loss','RSV'],axis=1,inplace=True)
    df = df[df['Date']==last_trade_date]
    df.to_csv(staticfiles_storage.path(f'{Name}/Test/{symbol}.csv'),index=False)



def get_equation(model):
    n_layers = len(model.layers)
    mat1 = np.matrix(model.layers[1].get_weights()[0])
    inputs = mat1.shape[0]
    mat2 = np.matrix(model.layers[1].get_weights()[1])
    mat3 = np.concatenate((mat1,mat2),axis=0)
    if n_layers == 2:
        return [float(x) for x in mat3[:,0]]
    for i in range(2,n_layers):
        mat4 = np.matrix(model.layers[i].get_weights()[0])
        mat5 = np.matmul(mat3,mat4)
        mat5[inputs,:]+=np.matrix(model.layers[i].get_weights()[1])
        
        mat3 = mat5
    return [float(x) for x in mat5[:,0]]

def run_daily():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Trading.settings')
    django.setup() 
    # Redirect stdout and stderr to a null device to discard any output
    # sys.stdout = open(os.devnull, 'w')
    # sys.stderr = open(os.devnull, 'w')

    # Filter out specific warnings
    from Broker.models import Stock,Exchange
    try:
        if Exchange.objects.filter(Name=Name).exists():
            pass
        else:
            market = Exchange(Name=Name,Extension=Extension,Currency=Currency,Country=Country,Frac=Frac)
            market.save()
    except Exception as e:
        print(f"Error creating Exchange instance: {e}")
        exchange = Exchange(Name=Name,Extension=Extension,Currency=Currency,Country=Country,Frac=Frac)
        exchange.save()
    market = Exchange.objects.get(Name=Name)
    warnings.filterwarnings("ignore")
    from Broker.models import Stock
    portfolio_shares = pd.read_excel(staticfiles_storage.path(f'{Name}/portfolio shares.xlsx'),header=0)
    evaluation = pd.read_excel(staticfiles_storage.path(f'{Name}/Evaluation.xlsx'),header=0)
    sectors = list(portfolio_shares['Sector'].unique())
    shares = list(portfolio_shares['Symbol'])
    sector_symbol_mapping = {sector : [share for share in list(portfolio_shares.loc[portfolio_shares['Sector']==sector,'Symbol'])]
                        for sector in sectors}
    timezone = pytz.timezone('America/New_York')
    end = datetime.datetime.now(timezone).date()
    last_run_date = pickle.load(open(staticfiles_storage.path(f'{Name}/last_run_date.pkl'),'rb'))
    if (end > last_run_date):
        for sector in sectors:
            reg_eqn = pd.read_excel(staticfiles_storage.path(f'{Name}/Models/Regression/Equation/{sector}.xlsx'),header=0)
            class_eqn = pd.read_excel(staticfiles_storage.path(f'{Name}/Models/Classification/Equation/{sector}.xlsx'),header=0)
            reg_eqn.sort_values(by='Date')
            class_eqn.sort_values(by='Date')
            reg_last_date = list(reg_eqn['Date'])[-1]
            class_last_date = list(class_eqn['Date'])[-1]
            last_train_date = min(reg_last_date,class_last_date)
            parsed_date = datetime.datetime.strptime(str(last_train_date), '%Y-%m-%d').date()
            start = parsed_date-datetime.timedelta(days=100)
            for share in sector_symbol_mapping[sector]:
                get_data(share,start=start,end=end)
        for sector in sectors:
            for share in sector_symbol_mapping[sector]:
                try:
                    create_variables(share)
                    data = pd.read_csv(staticfiles_storage.path(staticfiles_storage.path(f'{Name}/Data/{share}.csv')),header=0)
                    data = data.sort_values(by='Date')
                    last_trade_date = list(data['Date'])[-1]
                    last_price = list(data['Close'])[-1]
                    pred_variables(share,last_trade_date)
                    try:
                        if (end-pd.to_datetime(last_trade_date).date()).days == 1:
                            evaluation.loc[evaluation['Symbol']==share,'Trade Days'] += 1
                            if Stock.objects.filter(Symbol=share+Extension).exists():
                                stock = Stock.objects.get(Symbol=share+Extension)
                                if np.sign(stock.Expected_Price[0] - stock.CLS_Price) == np.sign(last_price - stock.CLS_Price):
                                    evaluation.loc[evaluation['Symbol']==share,'Correct Regression Prediction'] += 1
                                if np.sign(last_price - stock.CLS_Price) == np.sign(stock.probability - 50):
                                    evaluation.loc[evaluation['Symbol']==share,'Correct Classification Prediction'] += 1

                    except:
                        pass 
                except:
                    pass
        evaluation.to_excel(staticfiles_storage.path(f'{Name}/Evaluation.xlsx'),index=False)

        input_cols_df = pd.read_excel(staticfiles_storage.path(f'{Name}/input_cols.xlsx'),header=0)
        input_cols = list(input_cols_df['Inputs'])
        for sector in sectors:
            equation = pd.read_excel(staticfiles_storage.path(f'{Name}/Models/Regression/Equation/{sector}.xlsx'),header=0)
            equation.sort_values(by='Date',inplace=True)
            last_train_date = list(equation['Date'])[-1]
            train_df = pd.concat([pd.read_csv(staticfiles_storage.path(f'{Name}/Train/{share}.csv'),header=0) for share in
                            sector_symbol_mapping[sector]],axis=0)
            train_df = train_df[train_df['Date']>str(last_train_date)]
            for col in input_cols:
                train_df[col] = train_df[col].apply(lambda x: max(-1,x) if x < 0 else min(x,1))
            train_df = train_df.fillna(0)
            train_dates = sorted(train_df['Date'].unique())
            model = load_model(staticfiles_storage.path(f'{Name}/Models/Regression/{sector}_model.h5'))
            model.compile(loss='mean_squared_error',optimizer=Adam(learning_rate=0.001))
            for train_date in train_dates:
                x = train_df.loc[(train_df['Date']==train_date),input_cols]
                y = train_df.loc[(train_df['Date']==train_date),'future_return']
                if x.shape[0] >0 :
                    try:
                        model.fit(x,y,epochs=1000,verbose=0)
                    except:
                        pass
            if len(train_dates) > 0:
                model.save(staticfiles_storage.path(f'{Name}/Models/Regression/{sector}_model.h5'))
                equation_new = get_equation(model)
                equation_df = pd.DataFrame([[str(train_dates[-1])]+equation_new],columns=['Date']+input_cols+['intercept'])
                old_data = pd.read_excel(staticfiles_storage.path(f'{Name}/Models/Regression/Equation/{sector}.xlsx'),header=0)
                equation_df = pd.concat([old_data,equation_df],axis=0)
                equation_df.to_excel(staticfiles_storage.path(f'{Name}/Models/Regression/Equation/{sector}.xlsx'),index=False)
        
        for sector in sectors:
            equation = pd.read_excel(staticfiles_storage.path(f'{Name}/Models/Classification/Equation/{sector}.xlsx'),header=0)
            equation.sort_values(by='Date',inplace=True)
            last_train_date = list(equation['Date'])[-1]
            train_df = pd.concat([pd.read_csv(staticfiles_storage.path(f'{Name}/Train/{share}.csv'),header=0) for share in
                            sector_symbol_mapping[sector]],axis=0)
            train_df = train_df[train_df['Date']>str(last_train_date)]
            for col in input_cols:
                train_df[col] = train_df[col].apply(lambda x : max(-1,x) if x < 0 else min(x,1))
            train_df = train_df.fillna(0)
            train_dates = sorted(train_df['Date'].unique())
            model = load_model(staticfiles_storage.path(f'{Name}/Models/Classification/{sector}_model.h5'))
            model.compile(loss='binary_crossentropy',optimizer=Adam(learning_rate=0.001))
            for train_date in train_dates:
                x = train_df.loc[(train_df['Date']==train_date),input_cols]
                y = train_df.loc[(train_df['Date']==train_date),'lift']
                if x.shape[0] >0 :
                    try:
                        model.fit(x,y,epochs=1000,verbose=0)
                    except:
                        pass
            
            if len(train_dates) > 0:
                model.save(staticfiles_storage.path(f'{Name}/Models/Classification/{sector}_model.h5'))
                equation_new = get_equation(model)
                equation_df = pd.DataFrame([[str(train_dates[-1])]+equation_new],columns=['Date']+input_cols+['intercept'])
                old_data = pd.read_excel(staticfiles_storage.path(f'{Name}/Models/Classification/Equation/{sector}.xlsx'),header=0)
                equation_df = pd.concat([old_data,equation_df],axis=0)
                equation_df.to_excel(staticfiles_storage.path(f'{Name}/Models/Classification/Equation/{sector}.xlsx'),index=False)
        
        
        for sector in sectors:
            model_reg = load_model(staticfiles_storage.path(f'{Name}/Models/Regression/{sector}_model.h5'))
            model_class = load_model(staticfiles_storage.path(f'{Name}/Models/Classification/{sector}_model.h5'))
            reg_eqn = pd.read_excel(staticfiles_storage.path(f'{Name}/Models/Regression/Equation/{sector}.xlsx'),header=0)
            class_eqn = pd.read_excel(staticfiles_storage.path(f'{Name}/Models/Classification/Equation/{sector}.xlsx'),header=0)
            reg_eqn.sort_values(by='Date',inplace=True)
            class_eqn.sort_values(by='Date',inplace=True)
            last_reg_date = list(reg_eqn['Date'])[-1]
            last_class_date = list(class_eqn['Date'])[-1]
            reg_eqn = reg_eqn[reg_eqn['Date']==last_reg_date]
            class_eqn = class_eqn[class_eqn['Date']==last_class_date]
            reg_eqn.reset_index(inplace=True)
            class_eqn.reset_index(inplace=True)
            momentum_cols = list(input_cols_df.loc[input_cols_df['Type']=='Momentum','Inputs'])
            reversion_cols = list(input_cols_df.loc[input_cols_df['Type']=='Reversion','Inputs'])
            volume_cols = list(input_cols_df.loc[input_cols_df['Type']=='Volume','Inputs'])
            voltality_cols = list(input_cols_df.loc[input_cols_df['Type']=='Voltality','Inputs'])
            for share in sector_symbol_mapping[sector]:
                try:
                    pred = pd.read_csv(staticfiles_storage.path(f'{Name}/Test/{share}.csv'),header=0)
                    for col in input_cols:
                        pred[col] = pred[col].apply(lambda x: max(-1,x) if x < 0 else min(x,1))
                    pred = pred.fillna(0)
                    company = portfolio_shares.loc[portfolio_shares['Symbol']==share,'Company'].values[0]
                    display = portfolio_shares.loc[portfolio_shares['Symbol']==share,'Display'].values[0]
                    cap = portfolio_shares.loc[portfolio_shares['Symbol']==share,'CAP'].values[0]
                    eod_price = pred.loc[0,'Close']
                    risk = pred.loc[0,'Voltality5']*100
                    probability = np.round(model_class.predict(pred[input_cols])[0]*100,2)
                    market_reg = np.exp(reg_eqn.loc[0,'intercept'])
                    momentum_reg = np.exp(np.sum(np.array(reg_eqn.loc[0,momentum_cols])*np.array(pred.loc[0,momentum_cols])))
                    reversion_reg = np.exp(np.sum(np.array(reg_eqn.loc[0,reversion_cols])*np.array(pred.loc[0,reversion_cols])))
                    volume_reg = np.exp(np.sum(np.array(reg_eqn.loc[0,volume_cols])*np.array(pred.loc[0,volume_cols])))
                    voltality_reg = np.exp(np.sum(np.array(reg_eqn.loc[0,voltality_cols])*np.array(pred.loc[0,voltality_cols])))
                    market_class = class_eqn.loc[0,'intercept']
                    momentum_class = np.sum(np.array(class_eqn.loc[0,momentum_cols])*np.array(pred.loc[0,momentum_cols]))
                    reversion_class = np.sum(np.array(class_eqn.loc[0,reversion_cols])*np.array(pred.loc[0,reversion_cols]))
                    volume_class = np.sum(np.array(class_eqn.loc[0,volume_cols])*np.array(pred.loc[0,volume_cols]))
                    voltality_class = np.sum(np.array(class_eqn.loc[0,voltality_cols])*np.array(pred.loc[0,voltality_cols]))
                    expected_prices = []
                    reg_prediction = model_reg.predict(pred[input_cols])[0]
                    expected_price = np.exp(reg_prediction)*eod_price
                    expected_prices.append(expected_price)
                    latest_volume = pred.loc[0,'Volume']
                    for i in range(1,25):
                        pred.loc[0,'Close'] = expected_prices[-1]
                        pred.loc[0,'Volume'] = latest_volume 
                        for j in range(25,1,-1):
                            pred.loc[0,f'log_return_{j}'] = pred.loc[0,f'log_return_{j-1}']
                            pred.loc[0,f'Volume_Chg{j}'] = pred.loc[0,f'Volume_Chg{j-1}']
                        pred.loc[0,'log_return_1'] = reg_prediction
                        pred.loc[0,'Volume_Chg1'] = 0
                        pred['Prev_Price0'] = pred['Close']
                        pred['Prev_Volume0'] = pred['Volume']
                        for j in range(1,26):
                            pred[f'Prev_Price{j}'] = pred[f'Prev_Price{j-1}']/np.exp(pred[f'log_return_{j}'])
                            pred[f'Gain{j}'] = pred[[f'Prev_Price{j-1}',f'Prev_Price{j}']].apply(lambda x: max(x[0]-x[1],0),axis=1)
                            pred[f'Loss{j}'] = -pred[[f'Prev_Price{j-1}',f'Prev_Price{j}']].apply(lambda x: min(x[0]-x[1],0),axis=1)
                            pred[f'Prev_Volume{j}'] = pred[f'Prev_Volume{j-1}']/np.exp(pred[f'Volume_Chg{j}'])
                            pred[f'Vol_Gain{j}'] = pred[[f'Prev_Volume{j-1}',f'Prev_Volume{j}']].apply(lambda x: max(x[0]-x[1],0),axis=1)
                            pred[f'Vol_Loss{j}'] = -pred[[f'Prev_Volume{j-1}',f'Prev_Volume{j}']].apply(lambda x: min(x[0]-x[1],0),axis=1)
                        for k in range(5,30,5):
                            pred.loc[0,f'Avg_Gain{k}'] = np.mean(np.array([pred.loc[0,f'Gain{l}'] for l in range(1,k+1)]))
                            pred.loc[0,f'Avg_Loss{k}'] = np.mean(np.array([pred.loc[0,f'Loss{l}'] for l in range(1,k+1)]))
                            pred.loc[0,f'Avg_Vol_Gain{k}'] = np.mean(np.array([pred.loc[0,f'Vol_Gain{l}'] for l in range(1,k+1)]))
                            pred.loc[0,f'Avg_Vol_Loss{k}'] = np.mean(np.array([pred.loc[0,f'Vol_Loss{l}'] for l in range(1,k+1)]))
                            pred.loc[0,f'RSI{k}'] = pred.loc[0,f'Avg_Gain{k}']/(pred.loc[0,f'Avg_Gain{k}']+pred.loc[0,f'Avg_Loss{k}'])
                            pred.loc[0,f'RSIV{k}'] = pred.loc[0,f'Avg_Vol_Gain{k}']/(pred.loc[0,f'Avg_Vol_Gain{k}']+pred.loc[0,f'Avg_Vol_Loss{k}'])
                            pred.loc[0,f'Price_Voltality{k}'] = np.std(np.array([pred.loc[0,f'Prev_Price{l}'] for l in range(0,k)]))
                            pred.loc[0,f'Volume_Voltality{k}'] = np.std(np.array([pred.loc[0,f'Prev_Volume{l}'] for l in range(0,k)]))
                        for k in range(5,30,5):
                            pred.loc[0,f'MA{k}'] = (pred.loc[0,'Close'] + pred.loc[0,f'MA{k}']*k - pred.loc[0,'Close']/np.exp(pred.loc[0,'log_return_1']))/k
                            pred.loc[0,f'MAV{k}'] = (pred.loc[0,'Volume'] + pred.loc[0,f'MAV{k}']*k - pred.loc[0,'Volume']/np.exp(pred.loc[0,'Volume_Chg1']))/k
                            pred.loc[0,f'MA{k}_ratio'] = np.log(pred.loc[0,'Close']/pred.loc[0,f'MA{k}'])
                            pred.loc[0,f'MAV{k}_ratio'] = np.log(pred.loc[0,'Volume']/pred.loc[0,f'MAV{k}'])
                            pred.loc[0, f'Bollinger{k}_ratio'] = np.nan if pred.loc[0, f'Price_Voltality{k}'] == 0 else (pred.loc[0, 'Close'] - pred.loc[0, f'MA{k}'] + 2 * pred.loc[0, f'Price_Voltality{k}']) / (4 * pred.loc[0, f'Price_Voltality{k}'])
                            pred.loc[0, f'Bollinger{k}_volume_ratio'] = np.nan if pred.loc[0, f'Volume_Voltality{k}'] == 0 else (pred.loc[0, 'Volume'] - pred.loc[0, f'MAV{k}'] + 2 * pred.loc[0, f'Volume_Voltality{k}']) / (4 * pred.loc[0, f'Volume_Voltality{k}'])
                        
                        for k in range(5,30,5):
                            pred.loc[0,f'Voltality{k}'] = np.std(np.array([pred.loc[0,f'log_return_{l}'] for l in range(1,k+1)]))
                            pred.loc[0,f'Volume_chg_Voltality{k}'] = np.std(np.array([pred.loc[0,f'Volume_Chg{l}'] for l in range(1,k+1)])) 

                        for col in input_cols:
                            pred[col] = pred[col].apply(lambda x: max(-1,x) if x < 0 else min(x,1))
                        reg_prediction = model_reg.predict(pred[input_cols].fillna(0))[0]
                        expected_price = np.exp(reg_prediction)*expected_prices[i-1]
                        expected_prices.append(expected_price)
                    #net_return = np.round((np.exp(model_reg.predict(pred[input_cols])[0]) - 1)*100,6)
                    prev_returns = [pred.loc[0,'log_return_1'],pred.loc[0,'log_return_2'],pred.loc[0,'log_return_3'],pred.loc[0,'log_return_4'],pred.loc[0,'log_return_5']]
                    net_return = np.round((expected_prices[0]/eod_price-1)*100,6) 
                    trade_days = evaluation.loc[evaluation['Symbol']==share,'Trade Days'].values[0]
                    correct_reg = evaluation.loc[evaluation['Symbol']==share,'Correct Regression Prediction'].values[0]
                    correct_class = evaluation.loc[evaluation['Symbol']==share,'Correct Classification Prediction'].values[0]
                    reg_acc = np.round(correct_reg/trade_days*100,2)
                    class_acc = np.round(correct_class/trade_days*100,2)
                    if Stock.objects.filter(Symbol=share+Extension).exists():
                        # print(f'starting process from {share}')
                        stock = Stock.objects.get(Symbol=share+Extension)
                        stock.Exchange = market
                        stock.Display = display
                        stock.Sector = sector 
                        stock.Cap = cap
                        stock.Company = company
                        stock.CLS_Price = eod_price
                        stock.EOD_Price = eod_price
                        stock.Expected_Price = expected_prices
                        stock.Prev_Returns = prev_returns
                        stock.net_return = net_return 
                        stock.risk = risk
                        stock.probability = probability
                        stock.market_contri_reg = market_reg
                        stock.momentum_contri_reg = momentum_reg
                        stock.mean_reversion_contri_reg = reversion_reg
                        stock.voltality_contri_reg = voltality_reg
                        stock.volume_contri_reg = volume_reg
                        stock.market_contri_class = market_class
                        stock.momentum_contri_class = momentum_class
                        stock.mean_reversion_contri_class = reversion_class
                        stock.voltality_contri_class = voltality_class
                        stock.volume_contri_class = volume_class
                        stock.trade_days = trade_days
                        stock.correct_reg = correct_reg
                        stock.correct_class = correct_class
                        stock.reg_acc = reg_acc
                        stock.class_acc = class_acc
                        # print(f'{share} has closed price {stock.CLS_Price}')
                        stock.save()
                        # print(f'{share} is being saved at {eod_price} with actual value {stock.CLS_Price}')
                    else:
                        # print(f'starting process from {share}')
                        stock = Stock(Exchange=market,Sector=sector,Company=company,Cap=cap,Symbol=share+Extension,Display=display,CLS_Price=eod_price,EOD_Price=eod_price,Expected_Price=expected_prices,
                                        Prev_Returns=prev_returns,net_return=net_return,risk=risk,probability=probability,market_contri_reg=market_reg,momentum_contri_reg=momentum_reg,
                                        mean_reversion_contri_reg=reversion_reg,voltality_contri_reg=voltality_reg,
                                        volume_contri_reg=volume_reg,market_contri_class=market_class,
                                        momentum_contri_class=momentum_class,mean_reversion_contri_class=reversion_class,
                                        voltality_contri_class=voltality_class,volume_contri_class=volume_class,trade_days=trade_days,
                                    correct_reg=correct_reg,correct_class=correct_class,reg_acc=reg_acc,class_acc=class_acc)
                        # print(f'{share} has closed price {stock.CLS_Price}')
                        stock.save()
                        # print(f'{share} is being created at {eod_price} with actual value {stock.CLS_Price}')
                except:
                    pass
        pickle.dump(end,open(staticfiles_storage.path(f'{Name}/last_run_date.pkl'),'wb'))

run_daily()
