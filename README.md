# CSIS 4260 - Assignment 1: Stock Price Analysis & Prediction

A comprehensive analysis project evaluating data storage efficiency, comparing dataframe libraries, and building predictive models for S&P 500 stock prices.

## Project Overview

This project analyzes 5 years of daily stock prices for 505 S&P 500 companies (619,040 rows) across three phases:

### Part 1: Storage Benchmarking & Scalability
- Compare CSV vs Parquet storage formats (with compression schemes)
- Benchmark read/write performance at 1x, 10x, and 100x data scales
- Provide recommendations based on performance metrics

### Part 2: Data Manipulation & Predictive Modeling
- Compare Pandas vs Polars library performance
- Feature engineering with technical indicators (SMA, RSI)
- Build two prediction models (Linear Regression, Random Forest)
- 80-20 train-test split for backtesting

### Part 3: Interactive Dashboard
- Streamlit-based visual dashboard
- Company ticker search/selection
- Dynamic chart updates with predictions

## Project Structure

```
stock-analysis-a1/
├── data/                    # Data files (not tracked in git)
├── notebooks/
│   ├── part1_storage_benchmarking.ipynb
│   └── part2_analysis_modeling.ipynb
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   ├── linear_regression_model.py
│   │   └── random_forest_model.py
│   └── dashboard/
│       └── app.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/qurore/stock-analysis-a1.git
cd stock-analysis-a1
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place the data file:
   - Download `all_stocks_5yr.csv` and place it in the `data/` directory

## Running the Project

### Part 1 & 2: Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

### Part 3: Dashboard
```bash
streamlit run src/dashboard/app.py
```

## Dataset

- **Source**: S&P 500 Daily Stock Prices
- **Period**: 2013-02-08 to 2018-02-07
- **Records**: 619,040 rows
- **Companies**: 505 S&P 500 companies
- **Features**: date, open, high, low, close, volume, name

## Technical Indicators

1. **Simple Moving Average (SMA)**: 20-day and 50-day moving averages
2. **Relative Strength Index (RSI)**: 14-day momentum indicator

## Prediction Models

1. **Linear Regression**: Baseline model for price prediction
2. **Random Forest Regressor**: Ensemble method for improved accuracy

## Key Findings

### Storage Recommendations
- **1x Scale**: CSV is acceptable for simplicity
- **10x Scale**: Parquet with Snappy compression recommended
- **100x Scale**: Parquet with Snappy compression strongly recommended (5-10x faster)

### Library Performance
- Polars significantly outperforms Pandas for large-scale data operations
- Polars is recommended for production use with large datasets

## Author

CSIS 4260 - Data Management and Visualization

## License

This project is for educational purposes only.
