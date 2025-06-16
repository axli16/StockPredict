from flask import Flask, request, render_template
from joblib import load
import yfinance as yf
import os

app = Flask(__name__)
model = load("stock_model.joblib")

predictors = ["Close", "Volume", "Open", "High", "Low"] + \
                 [f"Close_Ratio_{h}" for h in [2, 5, 60, 250, 1000]] + \
                 [f"Trend_{h}" for h in [2, 5, 60, 250, 1000]]

def prepare_today_data(ticker):
    df = yf.download(ticker, period="5y", interval="1d", multi_level_index=False)
    if df.empty:  # or however much history your indicators need
        return None
    
    df["Tomorrow"] = df["Close"].shift(-1)
    print(df)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

    horizons = [2, 5, 60, 250, 1000]

    for horizon in horizons:
        rolling_averages = df.rolling(horizon).mean()

        ratio_column = f"Close_Ratio_{horizon}"
        df[ratio_column] = df["Close"] / rolling_averages["Close"]

        trend_column = f"Trend_{horizon}"
        df[trend_column] = df.shift(1).rolling(horizon).sum()["Target"]
    
    print(df.dropna())
    try:
        return df[predictors].iloc[-1:].copy()  # final row, with correct columns
    except KeyError as e:
        print("Missing predictor columns:", e)
        return None

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        ticker = request.form.get("ticker")
        input_data = prepare_today_data(ticker)
        print(input_data)
        if input_data is not None:
            pred = model.predict(input_data)[0]
            conf = model.predict_proba(input_data)[0][1]
            result = {
                "ticker": ticker,
                "buy": bool(pred),
                "confidence": round(conf * 100, 2)
            }
        else:
            result = {"error": "Not enough data or invalid ticker."}
    
    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
