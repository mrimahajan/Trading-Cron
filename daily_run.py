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
from keras.models import load_model
import numpy as np 
import yfinance as yf
from datetime import date
import datetime
from scipy.stats import norm
import pickle
import pytz
# from decimal import Decimal
import logging
import sys
import warnings
import tensorflow as tf
logging.disable(logging.CRITICAL)
tf.get_logger().setLevel('ERROR')

def get_data(symbol,start,end):
    try:
        df = yf.download(symbol+'.NS',start=str(start),end=str(end))
        df['Symbol'] = symbol
        df = df[['Symbol','Open','High','Low','Close','Adj Close','Volume']]
        if(df.shape[0] > 0):
            df.to_csv(staticfiles_storage.path(f'Data/{symbol}.csv'))
    except:
        print(f'Data not found for {symbol}')



def create_variables(symbol):
    df = pd.read_csv(staticfiles_storage.path(f'Data/{symbol}.csv'),header=0)
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
    df.to_csv(staticfiles_storage.path(f'Train/{symbol}.csv'),index=False)


def pred_variables(symbol,last_trade_date):
    df = pd.read_csv(staticfiles_storage.path(f'Data/{symbol}.csv'),header=0)
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
    df.to_csv(staticfiles_storage.path(f'Test/{symbol}.csv'),index=False)



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
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    # Filter out specific warnings
    warnings.filterwarnings("ignore", message="DataFrame is highly fragmented", category=UserWarning)
    from Broker.models import Stock
    portfolio_shares = pd.read_excel(staticfiles_storage.path('portfolio shares.xlsx'),header=0)
    sectors = list(portfolio_shares['Sector'].unique())
    shares = list(portfolio_shares['Symbol'])
    sector_symbol_mapping = {sector : [share for share in list(portfolio_shares.loc[portfolio_shares['Sector']==sector,'Symbol'])]
                        for sector in sectors}
    timezone = pytz.timezone('Asia/Kolkata')
    end = datetime.datetime.now(timezone).date()
    last_run_date = pickle.load(open(staticfiles_storage.path('last_run_date.pkl'),'rb'))
    if (end > last_run_date):
        for sector in sectors:
            reg_eqn = pd.read_excel(staticfiles_storage.path(f'Models/Regression/Equation/{sector}.xlsx'),header=0)
            class_eqn = pd.read_excel(staticfiles_storage.path(f'Models/Classification/Equation/{sector}.xlsx'),header=0)
            reg_eqn.sort_values(by='Date')
            class_eqn.sort_values(by='Date')
            reg_last_date = list(reg_eqn['Date'])[-1]
            class_last_date = list(class_eqn['Date'])[-1]
            last_train_date = min(reg_last_date,class_last_date)
            parsed_date = datetime.datetime.strptime(str(last_train_date), '%Y-%m-%d').date()
            start = parsed_date-datetime.timedelta(days=100)
            for share in sector_symbol_mapping[sector]:
                get_data(share,start=start,end=end)
        for share in shares:
            try:
                create_variables(share)
                data = pd.read_csv(staticfiles_storage.path(staticfiles_storage.path(f'Data/{share}.csv')),header=0)
                data = data.sort_values(by='Date')
                last_trade_date = list(data['Date'])[-1]
                pred_variables(share,last_trade_date)
            except:
                pass

        input_cols_df = pd.read_excel(staticfiles_storage.path('input_cols.xlsx'),header=0)
        input_cols = list(input_cols_df['Inputs'])
        for sector in sectors:
            equation = pd.read_excel(staticfiles_storage.path(f'Models/Regression/Equation/{sector}.xlsx'),header=0)
            equation.sort_values(by='Date',inplace=True)
            last_train_date = list(equation['Date'])[-1]
            train_df = pd.concat([pd.read_csv(staticfiles_storage.path(f'Train/{share}.csv'),header=0) for share in
                            sector_symbol_mapping[sector]],axis=0)
            train_df = train_df[train_df['Date']>str(last_train_date)]
            for col in input_cols:
                train_df[col] = train_df[col].apply(lambda x: max(-1,x) if x < 0 else min(x,1))
            train_dates = sorted(train_df['Date'].unique())
            model = load_model(staticfiles_storage.path(f'Models/Regression/{sector}_model.h5'))
            for train_date in train_dates:
                x = train_df.loc[(train_df['Date']==train_date),input_cols]
                y = train_df.loc[(train_df['Date']==train_date),'future_return']
                if x.shape[0] >0 :
                    model.fit(x,y,epochs=1000,verbose=0)
            if len(train_dates) > 0:
                model.save(staticfiles_storage.path(f'Models/Regression/{sector}_model.h5'))
                equation_new = get_equation(model)
                equation_df = pd.DataFrame([[str(train_dates[-1])]+equation_new],columns=['Date']+input_cols+['intercept'])
                equation_df.to_excel(staticfiles_storage.path(f'Models/Regression/Equation/{sector}.xlsx'),index=False)
        
        for sector in sectors:
            equation = pd.read_excel(staticfiles_storage.path(f'Models/Classification/Equation/{sector}.xlsx'),header=0)
            equation.sort_values(by='Date',inplace=True)
            last_train_date = list(equation['Date'])[-1]
            train_df = pd.concat([pd.read_csv(staticfiles_storage.path(f'Train/{share}.csv'),header=0) for share in
                            sector_symbol_mapping[sector]],axis=0)
            train_df = train_df[train_df['Date']>str(last_train_date)]
            for col in input_cols:
                train_df[col] = train_df[col].apply(lambda x : max(-1,x) if x < 0 else min(x,1))
            train_dates = sorted(train_df['Date'].unique())
            model = load_model(staticfiles_storage.path(f'Models/Classification/{sector}_model.h5'))
            for train_date in train_dates:
                x = train_df.loc[(train_df['Date']==train_date),input_cols]
                y = train_df.loc[(train_df['Date']==train_date),'lift']
                if x.shape[0] >0 :
                    model.fit(x,y,epochs=1000,verbose=0)
            
            if len(train_dates) > 0:
                model.save(staticfiles_storage.path(f'Models/Classification/{sector}_model.h5'))
                equation_new = get_equation(model)
                equation_df = pd.DataFrame([[str(train_dates[-1])]+equation_new],columns=['Date']+input_cols+['intercept'])
                equation_df.to_excel(staticfiles_storage.path(f'Models/Classification/Equation/{sector}.xlsx'),index=False)
        
        
        for sector in sectors:
            model_reg = load_model(staticfiles_storage.path(f'Models/Regression/{sector}_model.h5'))
            model_class = load_model(staticfiles_storage.path(f'Models/Classification/{sector}_model.h5'))
            reg_eqn = pd.read_excel(staticfiles_storage.path(f'Models/Regression/Equation/{sector}.xlsx'),header=0)
            class_eqn = pd.read_excel(staticfiles_storage.path(f'Models/Classification/Equation/{sector}.xlsx'),header=0)
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
                    pred = pd.read_csv(staticfiles_storage.path(f'Test/{share}.csv'),header=0)
                    for col in input_cols:
                        pred[col] = pred[col].apply(lambda x: max(-1,x) if x < 0 else min(x,1))
                    company = portfolio_shares.loc[portfolio_shares['Symbol']==share,'Company'].values[0]
                    cap = portfolio_shares.loc[portfolio_shares['Symbol']==share,'CAP'].values[0]
                    eod_price = pred.loc[0,'Close']
                    expected_price = np.exp(model_reg.predict(pred[input_cols])[0])*eod_price
                    net_return = np.round((np.exp(model_reg.predict(pred[input_cols])[0]) - 1)*100,6)
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
                    if Stock.objects.filter(Symbol=share).exists():
                        stock = Stock.objects.get(Symbol=share)
                        stock.Sector = sector 
                        stock.Cap = cap
                        stock.Company = company
                        stock.CLS_Price = eod_price
                        stock.EOD_Price = eod_price
                        stock.Expected_Price = expected_price
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
                        stock.save()
                    else:
                        stock = Stock(Sector=sector,Company=company,Cap=cap,Symbol=share,CLS_Price=eod_price,EOD_Price=eod_price,Expected_Price=expected_price,
                                        net_return=net_return,risk=risk,probability=probability,market_contri_reg=market_reg,momentum_contri_reg=momentum_reg,
                                        mean_reversion_contri_reg=reversion_reg,voltality_contri_reg=voltality_reg,
                                        volume_contri_reg=volume_reg,market_contri_class=market_class,
                                        momentum_contri_class=momentum_class,mean_reversion_contri_class=reversion_class,
                                        voltality_contri_class=voltality_class,volume_contri_class=volume_class)
                        stock.save()
                except:
                    pass
        pickle.dump(end,open(staticfiles_storage.path('last_run_date.pkl'),'wb'))

run_daily()