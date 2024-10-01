from Broker.models import *
import datetime
from datetime import date
import numpy as np 
import yfinance as yf
import pandas as pd
import gc

Name="NSE"
Extension=".NS"

for stock in Stock.objects.filter(Exchange__Name=exchange):
  symbol = stock.Symbol
  df = yf.download(tickers=symbol, period='1d', interval='1m')
  last_price = Decimal(np.round(list(df['Close'])[-1], 2))
  del df
  gc.collect()
  stock.EOD_Price = last_price
  stock.save()


