import yfinance as yf 
import pandas as pd
import time 
import json 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from joblib import dump

def fetch_stock_data(stock):
    try: 
        df = yf.Ticker(stock).history(period="max")
        del df["Dividends"]
        del df["Stock Splits"]
        df["Tomorrow"] = df["Close"].shift(-1)
        df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
        df = df.loc["1990-01-01":].copy()
        
        
        # Feature engineering
        horizons = [2, 5, 60, 250, 1000]
        for horizon in horizons:
            rolling_averages = df["Close"].rolling(horizon).mean()
            df[f"Close_Ratio_{horizon}"] = df["Close"] / rolling_averages
            df[f"Trend_{horizon}"] = df["Target"].shift(1).rolling(horizon).sum()

        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Failed to fetch {stock}: {e}")
        return None

# def predict(train, test, predictors, model):
#     model.fit(train[predictors], train["Target"])
#     preds = model.predict_proba(test[predictors])[:,1]
#     preds[preds >= .6] = 1
#     preds[preds < .6] = 0
#     preds = pd.Series(preds, index=test.index, name="predictions")
#     combined = pd.concat([test["Target"], preds], axis=1)
#     return combined

# def backtest(data, model, predictors, start=2500, step=250):
#     all_predictions = []

#     for i in range(start, data.shape[0], step):
#         train = data.iloc[0:i].copy()
#         test = data.iloc[i:(i+step)].copy()
#         predictions = predict(train, test, predictors, model)
#         all_predictions.append(predictions)
#     return pd.concat(all_predictions)

if __name__ == "__main__":

    print("Fetching tickers from json")
    with open('US-Stock-Symbols/nasdaq/nasdaq_tickers.json', 'r') as file:
        data = json.load(file)

    all_data = []
    print("Fetching stock data")
    for stock in data:
        df = fetch_stock_data(stock)
        if df is not None: 
            print(f"Pushed {stock}")
            all_data.append(df)
        else:
            print("Not pushed")
        time.sleep(0.2)
    
    if not all_data:
        print("No data collected.")
        exit()

    print("Stock data collected")
    # Combine all into one dataset
    combined_df = pd.concat(all_data)
    combined_df.dropna(inplace=True)

    predictors = ["Close", "Volume", "Open", "High", "Low"] + \
                 [f"Close_Ratio_{h}" for h in [2, 5, 60, 250, 1000]] + \
                 [f"Trend_{h}" for h in [2, 5, 60, 250, 1000]]
    
    X = combined_df[predictors]
    y = combined_df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    print("Start training")
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    dump(model, "stock_model.joblib")
    print("Pushed model")
    print(classification_report(y_test, preds))


