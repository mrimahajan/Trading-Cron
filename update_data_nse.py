import os
import django
import datetime
from datetime import date
import numpy as np 
import yfinance as yf
import pandas as pd
import gc

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Trading.settings')
django.setup() 

from Broker.models import Stock

Name="NSE"

for stock in Stock.objects.filter(Exchange__Name=Name):
  symbol = stock.Symbol
  df = yf.download(tickers=symbol, period='1d', interval='1m')
  last_price = Decimal(np.round(list(df['Close'])[-1], 2))
  del df
  gc.collect()
  stock.EOD_Price = last_price
  stock.save()


