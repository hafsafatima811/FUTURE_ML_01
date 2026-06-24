# Sales & Demand Forecasting

**Future Interns – Machine Learning Track | Task 1**

A machine learning pipeline that forecasts monthly sales using historical business data from the Sample Superstore dataset (2014–2017). The project includes data preprocessing, feature engineering, model training, evaluation, and an interactive HTML dashboard for business insights.

---

## 🎯 Objective

Build a model to predict future sales and demand using time-series and regression techniques, with clear visualizations and actionable business insights.

---

## 📊 Dataset

- **Source:** Sample - Superstore.csv
- **Period:** January 2014 – December 2017
- **Records:** 9,994 transactions
- **Fields:** Order details, products, customers, sales, profit, discounts, regions, segments

---

## 🤖 Models Implemented

| Model | Purpose | Key Strength |
|-------|---------|-------------|
| **Random Forest Regressor** | Primary forecast model | Captures non-linear patterns, robust to outliers |
| **Holt-Winters Exponential Smoothing** | Time-series baseline | Handles trend + seasonality explicitly |
| **Linear Regression** | Benchmark comparison | Simple, interpretable baseline |

---

## 🔧 Features Engineered

- **Temporal:** Month, Year, Quarter
- **Seasonal:** Sine/Cosine encoding for month cyclicality
- **Lag Features:** Previous 1, 2, 3 months sales (autoregressive)

---

## 📈 Results

| Metric | Random Forest | Holt-Winters | Linear Regression |
|--------|--------------|--------------|-------------------|
| MAE | ~$18,200 | ~$6,900 | ~$15,900 |
| RMSE | ~$22,300 | ~$8,400 | ~$20,700 |
| R² Score | 0.026 | N/A (in-sample) | 0.160 |

---

## 🗂️ Project Structure

```text
FUTURE_ML_01/
├── data/
│   └── Sample - Superstore.csv          # Raw dataset
├── src/
│   └── sales_forecast.py                # Main ML pipeline
├── output/
│   ├── forecast_data.json               # Generated forecast data
│   └── index.html                       # Interactive dashboard
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python src/sales_forecast.py
```

This will:
- Load and clean the dataset
- Engineer time-based features
- Train all three models
- Evaluate performance
- Generate 12-month forecast
- Save results to **output/**

---

### 3. View the Dashboard
Open **output/index.html** in any web browser to explore:
- KPI cards, forecast charts, model metrics
- Time series decomposition (trend, seasonal, residual)
- Category, regional, and segment analysis
- AI-powered business insights

---

## 💡 Key Business Insights

1. **Strong Q4 Seasonality:** Sales peak 40–60% in November–December
2. **Technology Leading:** Highest growth category
3. **West Region Opportunity:** Consistent growth trajectory
4. **Furniture Margin Warning:** Some items show negative profit
5. **Q1 Seasonal Dip:** January–February consistently underperform
6. **Corporate Segment Growth:** 15% year-over-year increase

---

## 🛠️ Tech Stack

- **Python:** pandas, numpy, scikit-learn, statsmodels
- **ML Models:** Random Forest, Linear Regression, Holt-Winters
- **Visualization:** Chart.js (embedded in HTML dashboard)
- **Frontend:** Vanilla JavaScript, CSS Grid/Flexbox

---

## 4. Output

> Open **output/index.html** to see the live interactive dashboard with forecast charts, decomposition analysis, and business insights.

---

**Built for Future Interns - Machine Learning Track (Task 1)**

---
