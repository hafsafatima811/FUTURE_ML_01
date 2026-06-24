import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SALES & DEMAND FORECASTING - TASK 1")
print("Future Interns - Machine Learning Track")
print("=" * 60)

# ============================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================
print("\n[1/7] Loading dataset...")
df = pd.read_csv('data/Sample - Superstore.csv', encoding='latin-1')
print(f"   Dataset shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

# Convert dates
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

print(f"   Date range: {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
print(f"   Total Sales: ${df['Sales'].sum():,.2f}")
print(f"   Total Profit: ${df['Profit'].sum():,.2f}")

# ============================================
# STEP 2: DATA CLEANING & FEATURE ENGINEERING
# ============================================
print("\n[2/7] Engineering time-based features...")

# Monthly aggregation
monthly_sales = df.groupby(df['Order Date'].dt.to_period('M')).agg({
    'Sales': 'sum',
    'Quantity': 'sum',
    'Profit': 'sum',
    'Discount': 'mean',
    'Order ID': 'nunique'
}).reset_index()
monthly_sales['Date'] = monthly_sales['Order Date'].dt.to_timestamp()
monthly_sales.columns = ['Period', 'Sales', 'Quantity', 'Profit', 'Avg_Discount', 'Num_Orders', 'Date']

print(f"   Monthly records: {len(monthly_sales)}")

# Time features
monthly_sales['Month'] = monthly_sales['Date'].dt.month
monthly_sales['Year'] = monthly_sales['Date'].dt.year
monthly_sales['Quarter'] = monthly_sales['Date'].dt.quarter
monthly_sales['Month_Sin'] = np.sin(2 * np.pi * monthly_sales['Month'] / 12)
monthly_sales['Month_Cos'] = np.cos(2 * np.pi * monthly_sales['Month'] / 12)

# Lag features (previous months' sales)
monthly_sales['Lag1'] = monthly_sales['Sales'].shift(1)
monthly_sales['Lag2'] = monthly_sales['Sales'].shift(2)
monthly_sales['Lag3'] = monthly_sales['Sales'].shift(3)

# Drop rows with NaN (first 3 rows due to lag features)
model_df = monthly_sales.dropna().copy()
print(f"   Model-ready records: {len(model_df)}")

# ============================================
# STEP 3: PREPARE TRAINING DATA
# ============================================
print("\n[3/7] Preparing training data...")

feature_cols = ['Month', 'Year', 'Quarter', 'Month_Sin', 'Month_Cos', 'Lag1', 'Lag2', 'Lag3']
X = model_df[feature_cols]
y = model_df['Sales']

# Use last 6 months as test set
split_idx = len(model_df) - 6
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# ============================================
# STEP 4: TRAIN MACHINE LEARNING MODELS
# ============================================
print("\n[4/7] Training models...")

# Model 1: Random Forest
print("   Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Model 2: Linear Regression (baseline)
print("   Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Model 3: Holt-Winters Exponential Smoothing
print("   Training Holt-Winters Exponential Smoothing...")
ts_data = model_df.set_index('Date')['Sales']
hw_model = ExponentialSmoothing(ts_data, trend='add', seasonal='add', seasonal_periods=12).fit()

# ============================================
# STEP 5: EVALUATE MODELS
# ============================================
print("\n[5/7] Model Evaluation Results:")
print("-" * 50)

print("\nRandom Forest Regressor:")
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)
print(f"   MAE:  ${rf_mae:,.2f}")
print(f"   RMSE: ${rf_rmse:,.2f}")
print(f"   R²:   {rf_r2:.4f}")

print("\nLinear Regression:")
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)
print(f"   MAE:  ${lr_mae:,.2f}")
print(f"   RMSE: ${lr_rmse:,.2f}")
print(f"   R²:   {lr_r2:.4f}")

print("\nHolt-Winters (in-sample):")
hw_fitted = hw_model.fittedvalues
hw_residuals = ts_data - hw_fitted
hw_mae = np.mean(np.abs(hw_residuals))
hw_rmse = np.sqrt(np.mean(hw_residuals**2))
print(f"   MAE:  ${hw_mae:,.2f}")
print(f"   RMSE: ${hw_rmse:,.2f}")

# Feature importance
print("\nFeature Importance (Random Forest):")
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
for _, row in importance.iterrows():
    print(f"   {row['Feature']:<15} {row['Importance']:.3f}")

# ============================================
# STEP 6: GENERATE 12-MONTH FORECAST
# ============================================
print("\n[6/7] Generating 12-month forecast...")

last_date = monthly_sales['Date'].max()
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')

# Random Forest iterative forecast
last_sales = monthly_sales['Sales'].tolist()
rf_future_sales = []

for i in range(12):
    lag1 = last_sales[-1] if len(last_sales) >= 1 else monthly_sales['Sales'].mean()
    lag2 = last_sales[-2] if len(last_sales) >= 2 else monthly_sales['Sales'].mean()
    lag3 = last_sales[-3] if len(last_sales) >= 3 else monthly_sales['Sales'].mean()

    row = pd.DataFrame({
        'Month': [future_dates[i].month],
        'Year': [future_dates[i].year],
        'Quarter': [future_dates[i].quarter],
        'Month_Sin': [np.sin(2 * np.pi * future_dates[i].month / 12)],
        'Month_Cos': [np.cos(2 * np.pi * future_dates[i].month / 12)],
        'Lag1': [lag1],
        'Lag2': [lag2],
        'Lag3': [lag3]
    })

    pred = rf_model.predict(row)[0]
    rf_future_sales.append(pred)
    last_sales.append(pred)

# Holt-Winters forecast
hw_forecast = hw_model.forecast(12)

print("\n12-Month Sales Forecast:")
print("-" * 50)
print(f"{'Month':<12} {'Random Forest':>15} {'Holt-Winters':>15}")
print("-" * 50)
for i, date in enumerate(future_dates):
    print(f"{date.strftime('%Y-%m'):<12} ${rf_future_sales[i]:>13,.0f} ${hw_forecast.iloc[i]:>13,.0f}")

# ============================================
# STEP 7: SAVE RESULTS
# ============================================
print("\n[7/7] Saving results...")

# Category data
cat_monthly = df.groupby([df['Order Date'].dt.to_period('M'), 'Category']).agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()
cat_monthly['Order Date'] = cat_monthly['Order Date'].dt.to_timestamp()
cat_monthly.columns = ['Date', 'Category', 'Sales', 'Profit']

# Regional data
regional_monthly = df.groupby([df['Order Date'].dt.to_period('M'), 'Region']).agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()
regional_monthly['Order Date'] = regional_monthly['Order Date'].dt.to_timestamp()

# Segment data
segment_monthly = df.groupby([df['Order Date'].dt.to_period('M'), 'Segment']).agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()
segment_monthly['Order Date'] = segment_monthly['Order Date'].dt.to_timestamp()

# Top subcategories
subcat_sales = df.groupby('Sub-Category').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).sort_values('Sales', ascending=False).head(15)

# Top states
state_sales = df.groupby('State').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).sort_values('Sales', ascending=False).head(15)

# Decomposition
decomposition = seasonal_decompose(ts_data, model='additive', period=12)

# Build forecast data
forecast_data = {
    'historical': {
        'dates': [d.strftime('%Y-%m-%d') for d in monthly_sales['Date']],
        'sales': monthly_sales['Sales'].tolist(),
        'profit': monthly_sales['Profit'].tolist(),
        'quantity': monthly_sales['Quantity'].tolist(),
        'orders': monthly_sales['Num_Orders'].tolist()
    },
    'forecast': {
        'dates': [d.strftime('%Y-%m-%d') for d in future_dates],
        'rf_sales': rf_future_sales,
        'hw_sales': hw_forecast.tolist()
    },
    'decomposition': {
        'trend': decomposition.trend.dropna().tolist(),
        'seasonal': decomposition.seasonal.dropna().tolist(),
        'residual': decomposition.resid.dropna().tolist(),
        'dates': [d.strftime('%Y-%m-%d') for d in decomposition.trend.dropna().index]
    },
    'metrics': {
        'rf_mae': float(rf_mae),
        'rf_rmse': float(rf_rmse),
        'rf_r2': float(rf_r2),
        'lr_mae': float(lr_mae),
        'lr_rmse': float(lr_rmse),
        'lr_r2': float(lr_r2),
        'hw_mae': float(hw_mae),
        'hw_rmse': float(hw_rmse)
    },
    'category_data': {
        'dates': [d.strftime('%Y-%m-%d') for d in cat_monthly['Date'].unique()],
        'categories': {}
    },
    'regional_data': {
        'dates': [d.strftime('%Y-%m-%d') for d in regional_monthly['Order Date'].unique()],
        'regions': {}
    },
    'segment_data': {
        'dates': [d.strftime('%Y-%m-%d') for d in segment_monthly['Order Date'].unique()],
        'segments': {}
    },
    'top_subcategories': {
        'names': subcat_sales.index.tolist(),
        'sales': subcat_sales['Sales'].tolist(),
        'profit': subcat_sales['Profit'].tolist(),
        'quantity': subcat_sales['Quantity'].tolist()
    },
    'top_states': {
        'names': state_sales.index.tolist(),
        'sales': state_sales['Sales'].tolist(),
        'profit': state_sales['Profit'].tolist()
    },
    'summary': {
        'total_sales': float(df['Sales'].sum()),
        'total_profit': float(df['Profit'].sum()),
        'total_orders': int(df['Order ID'].nunique()),
        'total_customers': int(df['Customer ID'].nunique()),
        'avg_order_value': float(df['Sales'].sum() / df['Order ID'].nunique()),
        'profit_margin': float(df['Profit'].sum() / df['Sales'].sum() * 100),
        'date_range': f"{df['Order Date'].min().strftime('%Y-%m-%d')} to {df['Order Date'].max().strftime('%Y-%m-%d')}",
        'forecast_period': f"{future_dates[0].strftime('%Y-%m-%d')} to {future_dates[-1].strftime('%Y-%m-%d')}"
    }
}

# Add category data
for cat in cat_monthly['Category'].unique():
    cat_data = cat_monthly[cat_monthly['Category'] == cat]
    forecast_data['category_data']['categories'][cat] = {
        'sales': cat_data['Sales'].tolist(),
        'profit': cat_data['Profit'].tolist()
    }

# Add regional data
for region in regional_monthly['Region'].unique():
    reg_data = regional_monthly[regional_monthly['Region'] == region]
    forecast_data['regional_data']['regions'][region] = {
        'sales': reg_data['Sales'].tolist(),
        'profit': reg_data['Profit'].tolist()
    }

# Add segment data
for seg in segment_monthly['Segment'].unique():
    seg_data = segment_monthly[segment_monthly['Segment'] == seg]
    forecast_data['segment_data']['segments'][seg] = {
        'sales': seg_data['Sales'].tolist(),
        'profit': seg_data['Profit'].tolist()
    }

# Save JSON
with open('output/forecast_data.json', 'w') as f:
    json.dump(forecast_data, f)

print("   ✅ forecast_data.json saved to output/")
print("   ✅ All outputs generated successfully!")
print("\n" + "=" * 60)
print("NEXT STEP: Open output/index.html in your browser")
print("=" * 60)
