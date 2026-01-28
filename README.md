# CSIS 4260 - Assignment 1: Stock Price Analysis & Prediction

A comprehensive analysis project evaluating data storage efficiency, comparing dataframe libraries, and building predictive models for S&P 500 stock prices.

## Table of Contents
- [Project Overview](#project-overview)
- [Setup Instructions](#setup-instructions)
- [Part 1: Storage Benchmarking](#part-1-storage-benchmarking--scalability)
- [Part 2: Data Analysis & Modeling](#part-2-data-manipulation-analysis--predictive-modeling)
- [Part 3: Interactive Dashboard](#part-3-interactive-dashboard)
- [Project Structure](#project-structure)

---

## Project Overview

This project analyzes 5 years of daily stock prices for 505 S&P 500 companies (619,040 rows) across three phases.

### Dataset
- **Source**: S&P 500 Daily Stock Prices
- **Period**: 2013-02-08 to 2018-02-07
- **Records**: 619,040 rows
- **Companies**: 505 S&P 500 companies
- **Features**: date, open, high, low, close, volume, name

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd stock-analysis-a1
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place the data file
Download `all_stocks_5yr.csv` and place it in the `data/` directory.

### 5. Run Jupyter Notebooks (Part 1 & 2)
```bash
jupyter notebook notebooks/
```

### 6. Run Dashboard (Part 3)
```bash
streamlit run src/dashboard/app.py
```

---

## Part 1: Storage Benchmarking & Scalability

### Objective
Evaluate whether to keep data in CSV format or convert to Parquet format for storing and retrieving time-series stock data.

### Methodology
- Benchmark read/write performance at **1x, 10x, and 100x** data scales
- Compare formats: CSV, Parquet (No compression), Parquet (Snappy), Parquet (GZIP), Parquet (Brotli)
- Measure: write time, read time, and file size

### Library Choice: PyArrow
**Why PyArrow?**
- Native support for Parquet format with multiple compression codecs
- High-performance columnar memory format
- Excellent integration with Pandas
- Industry standard for big data processing

### Results Summary

| Scale | Format | Write Time | Read Time | File Size |
|-------|--------|------------|-----------|-----------|
| 1x | CSV | 1.109s | 0.166s | 28.21 MB |
| 1x | Parquet (Snappy) | 0.102s | 0.021s | 10.03 MB |
| 10x | CSV | 11.359s | 1.646s | 298.04 MB |
| 10x | Parquet (Snappy) | 1.019s | 0.156s | 94.93 MB |
| 100x | CSV | 118.362s | 16.901s | 3049.49 MB |
| 100x | Parquet (Snappy) | 10.242s | 1.508s | 948.05 MB |

### Analysis & Recommendations

**Key Findings:**
1. **Read Performance**: Parquet consistently outperforms CSV
   - 1x scale: 8x faster
   - 10x scale: 10x faster
   - 100x scale: 11x faster

2. **Write Performance**: Parquet (Snappy) is 10-12x faster than CSV

3. **Storage Efficiency**: Parquet reduces file size by 65-70%

**Compression Comparison:**
| Compression | Speed | Compression Ratio | Best Use Case |
|-------------|-------|-------------------|---------------|
| Snappy | Fastest | Good (65%) | **General purpose - Recommended** |
| GZIP | Slow | Better (74%) | Cold storage, archival |
| Brotli | Moderate | Best (75%) | When size is critical |

**Final Recommendation**: **Parquet with Snappy compression** for all scales
- Best balance of read/write speed and compression
- Significant storage savings
- Type preservation (no parsing errors)
- Better scalability for future data growth

---

## Part 2: Data Manipulation, Analysis & Predictive Modeling

### Objective
1. Compare Pandas vs Polars performance
2. Enhance dataset with technical indicators
3. Build prediction models for next-day closing price

### Library Comparison: Pandas vs Polars

**Why compare these two?**
- Pandas: Industry standard, mature ecosystem
- Polars: Modern alternative with Rust backend, designed for performance

### Performance Results

| Operation | Pandas | Polars | Speedup |
|-----------|--------|--------|---------|
| GroupBy Aggregation | 0.0248s | 0.0048s | **5.21x** |
| Filtering | 0.0062s | 0.0015s | **4.16x** |
| Sorting | 0.0542s | 0.0140s | **3.87x** |
| Rolling Mean (20-day) | 0.0756s | 0.0060s | **12.68x** |
| Add Calculated Column | 0.0054s | 0.0009s | **5.95x** |
| **Average** | - | - | **6.38x** |

**Analysis:**
- Polars is **6.4x faster on average** across all operations
- Most significant improvement in window/rolling operations (12.68x)
- Polars uses parallel processing and Rust's memory efficiency

**Recommendation**: Use **Polars** for production workloads with large datasets. Use Pandas when ecosystem compatibility is required.

### Feature Engineering: Technical Indicators

**Indicators Implemented:**
1. **Simple Moving Average (SMA)** - 20-day and 50-day
   - Identifies price trends
   - SMA 20 captures short-term momentum
   - SMA 50 captures medium-term trend

2. **Relative Strength Index (RSI)** - 14-day
   - Momentum oscillator (0-100 scale)
   - >70: Overbought condition
   - <30: Oversold condition

3. **Volatility** - 20-day rolling standard deviation
   - Measures price variability
   - Higher values indicate higher risk

4. **Price Momentum** - Close price minus SMA 20
   - Positive: Bullish momentum
   - Negative: Bearish momentum

### Predictive Models

**Model 1: Linear Regression**
- Simple, interpretable baseline
- Fast training (<1 second)
- Assumes linear relationship between features and target

**Model 2: LSTM (Long Short-Term Memory)**
- Deep learning model for sequential data
- Can capture non-linear patterns
- Architecture: 2 LSTM layers (64, 32 units) + Dense layers
- Trained with early stopping to prevent overfitting

### Model Performance (80-20 Train-Test Split)

| Metric | Linear Regression | LSTM |
|--------|-------------------|------|
| R² Score | **0.9992** | 0.9970 |
| RMSE | **$1.33** | $2.49 |
| MAE | **$0.78** | $1.82 |
| Training Time | **0.05s** | 41.58s |

**Analysis:**
- Both models achieve excellent R² scores (>0.99)
- Linear Regression slightly outperforms LSTM for this dataset
- The strong predictive power is largely due to stock price autocorrelation (today's price is highly predictive of tomorrow's price)
- Linear Regression is preferred for its simplicity, interpretability, and faster training

**Feature Importance (Linear Regression):**
1. `close` (current closing price) - Highest coefficient
2. `price_momentum` - Trend indicator
3. `sma_20` - Short-term moving average
4. `high`, `low`, `open` - Daily price range

---

## Part 3: Interactive Dashboard

### Technology Choice: Streamlit

**Why Streamlit?**
- Python-native: No JavaScript/HTML required
- Rapid prototyping: Build interactive apps in minutes
- Built-in widgets: Dropdowns, sliders, date pickers
- Plotly integration: Interactive, professional charts
- Easy deployment: Streamlit Cloud or self-hosted

### Dashboard Features

1. **Stock Selection Panel**
   - Search by ticker symbol
   - Dropdown with all 505 companies
   - Date range filter

2. **Model Performance Metrics**
   - R², RMSE, MAE for both models
   - Displayed in sidebar

3. **Interactive Charts**
   - Price chart with predictions overlay
   - Technical indicators (RSI, Volatility)
   - Actual vs Predicted scatter plots
   - Error distribution histograms

4. **Data Table**
   - Recent 20 rows of selected stock
   - Shows predictions alongside actual prices

### Running the Dashboard
```bash
streamlit run src/dashboard/app.py
```
Access at: http://localhost:8501

---

## Project Structure

```
stock-analysis-a1/
├── data/                           # Data files (not tracked in git)
│   └── all_stocks_5yr.csv
├── notebooks/
│   ├── part1_storage_benchmarking.ipynb   # Storage format benchmarks
│   └── part2_analysis_modeling.ipynb      # Analysis and ML models
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py         # Technical indicators
│   │   ├── linear_regression_model.py     # LR model class
│   │   └── random_forest_model.py         # RF model class
│   └── dashboard/
│       └── app.py                         # Streamlit dashboard
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Dependencies

Key libraries used:
- **pandas** (2.0+): Data manipulation
- **polars** (0.20+): High-performance data processing
- **pyarrow** (14.0+): Parquet file support
- **scikit-learn** (1.3+): Machine learning models
- **tensorflow** (2.15+): LSTM deep learning model
- **streamlit** (1.29+): Interactive dashboard
- **plotly** (5.18+): Interactive visualizations
- **matplotlib** (3.7+): Static plots in notebooks

---

## Author

CSIS 4260 - Data Management and Visualization
Douglas College

## License

This project is for educational purposes only.
