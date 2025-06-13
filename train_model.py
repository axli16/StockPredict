import yfinance as yf 
import pandas as pd
import time 
import json 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import os


def fetch_stock_data(stock):
    try: 
        df = yf.Ticker(stock).history(period="max")
        del df["Dividends"]
        del df["Stock Splits"]
        df["Tomorrow"] = df["Close"].shift(-1)
        df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
        df = df.loc["1990-01-01":].copy()
        return df
    except:
        return None

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

if __name__ == "__main__":

    # with open('US-Stock-Symbols/nasdaq/nasdaq_tickers.json', 'r') as file:
    #     data = json.load(file)
    

    # for stock in data:
    #     print(stock)
    #     df = fetch_stock_data(stock)
    #     if df is not None and not df.empty:
    #         df.to_csv("combined_data.csv", mode="a", header=not os.path.exists("combined_data.csv"))
    #     time.sleep(1)
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    chunksize = 100_000
    for chunk in pd.read_csv("combined_data.csv", chunksize=chunksize):

        train = chunk.iloc[:-10000]
        test = chunk.iloc[-10000:]

        predictors = ["Close", "Volume","Open", "High", "Low"]
        model.fit(train[predictors], train["Target"])

        preds = model.predict(test[predictors])

        preds = pd.Series(preds, index=test.index)

        score = precision_score(test["Target"], preds)
        print(f"Chunk precision: {score:.4f}")

        combined = pd.concat([test["Target"], preds], axis=1)

        # horizons = [2, 5, 60, 250, 1000]
        # new_predictors = []

        # for horizon in horizons:
        #     rolling_averages = chunk.rolling(horizon).mean()

        #     ratio_column = f"Close_Ratio_{horizon}"
        #     chunk[ratio_column] = chunk["Close"] / rolling_averages["Close"]

        #     trend_column = f"Trend_{horizon}"
        #     chunk[trend_column] = chunk.shift(1).rolling(horizon).sum()["Target"]

        #     new_predictors += [ratio_column, trend_column]

        chunk.dropna(inplace=True)

        predictions = backtest(chunk, model, predictors)

    predictions["predictions"].value_counts()

    precision_score(predictions["Target"], predictions["predictions"])

