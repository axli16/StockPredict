# 📈 AI Stock Buy Predictor

This is a simple AI-powered web app that predicts whether you should **buy a NASDAQ stock today** based on historical market data. It uses a trained machine learning model to analyze stock trends and outputs a recommendation — all wrapped in a clean, one-page Flask app.

---

## 🔍 What It Does

Given a stock ticker (e.g., `AAPL`, `GOOGL`, `TSLA`), the app:
1. Fetches the past 1 year of daily data for that stock.
2. Calculates technical indicators and historical trends.
3. Feeds this data into a trained classifier model.
4. Predicts whether the stock is likely to end higher today than it opened.

> 💡 The model outputs a **buy/not-buy decision** along with a confidence score.

---

## 🧠 How the AI Works

- **Historical Data**: Pulled from [Yahoo Finance](https://finance.yahoo.com) using the `yfinance` Python package. (Last trained June 14, 2025)
- **Features Used**:
  - Price ratios over rolling windows (2, 5, 60, 250, 1000 days)
  - Upward trend frequencies over those windows
- **Model**: Trained using `scikit-learn` on labeled historical data (1 if next day was up, 0 otherwise).
- **Prediction Target**: Whether the closing price will be higher than the opening price today.

---
## 🚀 How to Run

Follow these steps to get the app running locally on your machine.

### 1. Clone the Repository

```bash
git clone https://github.com/axli16/StockPredict.git
cd StockPredict

#Train model -> produces a .joblib file that contains your model which is trained on the most recent data.
git clone https://github.com/rreichel3/US-Stock-Symbols.git
python train_model.py

# Run Local host
python app.py
```
