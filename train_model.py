import yfinance as yf 
import pandas as pd
import json 

with open('US-Stock-Symbols/nasdaq/nasdaq_tickers.json', 'r') as file:
    data = json.load(file)

def fetch_stock_data(stock):
    
    try: 
        df = yf.Ticker(stock).history(period="max")
    except:
        return None




