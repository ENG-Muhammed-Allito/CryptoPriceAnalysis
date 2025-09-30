# 📈 Predicting Cryptocurrency Price Dynamics

A research project analyzing the prediction of cryptocurrency price movements using **Machine Learning algorithms** across multiple time resolutions.

---

## 📌 Overview

Cryptocurrency markets are highly volatile and complex, making price prediction a challenging yet critical task.
This project compares **seven different ML algorithms** across **hourly, daily, and monthly datasets** for 15 popular cryptocurrencies.

---

## 🛠️ Tech Stack & Tools

* **Python** 🐍
* **Pandas, NumPy** (data preprocessing)
* **Scikit-learn, LightGBM** (ML models)
* **Matplotlib, Seaborn** (visualization)
* **Binance API** (data collection)

---

## 📊 Dataset

* Source: **Binance API**
* 15 cryptocurrencies analyzed (BTC, ETH, BNB, ADA, DOT, XRP, LTC, DOGE, UNI, LINK, ATOM, EOS, AAVE, IOTA, XLM).
* Time resolutions: **Hourly (~50k records), Daily (~3k records), Monthly (~80 records)**.
* Features include: `Open, High, Low, Close, Volume` + technical indicators (RSI, MACD, Bollinger Bands, Moving Averages, etc.).

---

## 🤖 Machine Learning Models

* Decision Tree Regression (DTR)
* Gradient Boosting Regression (GBR)
* K-Nearest Neighbors (KNN)
* LightGBM (LGBM)
* Random Forest Regression (RFR)
* Ridge Regression (RR)
* Support Vector Regression (SVR)

---

## 📈 Results & Findings

* **Short-term (Hourly/Daily):** Most models achieved near-perfect accuracy (R² ≈ 0.99).
* **Long-term (Monthly):** Model choice becomes critical.

  * **Ridge Regression (RR)** and **LightGBM** showed the most consistent performance.
  * **KNN & SVR** struggled in low-data / high-volatility conditions.
* Conclusion: Matching the right algorithm to the time scale significantly improves prediction accuracy.

---

## 🔮 Future Work

* Incorporating **Deep Learning (LSTM/GRU)** for sequential modeling.
* Expanding datasets with **social media sentiment** & **macro-economic indicators**.
* Deploying models as an **API for real-time price prediction**.

---

## 👨‍💻 Author

**Eng. Muhammed Allito**

* Computer Engineer | AI Enthusiast
* 📧 [engmuhammedallito@gmail.com](mailto:engmuhammedallito@gmail.com)

---
