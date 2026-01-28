"""
Stock Price Prediction Dashboard

CSIS 4260 - Assignment 1, Part 3
Interactive dashboard for visualizing stock price predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Try to import TensorFlow (may not be available on Python 3.13)
TF_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    pass

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Dark Theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4da6ff;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    [data-testid="stMetric"] {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3d3d3d;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    [data-testid="stMetricLabel"] {
        color: #fafafa;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff;
    }
    [data-testid="stHorizontalBlock"] > div {
        flex: 1;
        min-width: 0;
    }
    .stApp {
        background-color: #0e1117;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and prepare the stock data."""
    # Try multiple paths for flexibility
    data_paths = [
        Path(__file__).parent.parent.parent / 'data' / 'all_stocks_5yr.csv',
        Path(__file__).parent.parent.parent / 'all_stocks_5yr.csv',
        Path('data/all_stocks_5yr.csv'),
        Path('all_stocks_5yr.csv'),
    ]

    df = None
    for path in data_paths:
        if path.exists():
            df = pd.read_csv(path)
            break

    if df is None:
        st.error("Data file not found. Please ensure 'all_stocks_5yr.csv' is in the data directory.")
        st.stop()

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['name', 'date']).reset_index(drop=True)

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe."""
    df = df.copy()

    # SMA 20 and 50
    df['sma_20'] = df.groupby('name')['close'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    df['sma_50'] = df.groupby('name')['close'].transform(
        lambda x: x.rolling(window=50, min_periods=1).mean()
    )

    # RSI 14
    def calc_rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    df['rsi_14'] = df.groupby('name')['close'].transform(calc_rsi)

    # Price momentum
    # Daily return
    df['daily_return'] = df.groupby('name')['close'].transform(
        lambda x: x.pct_change() * 100
    )

    # Target: next day's close
    df['next_close'] = df.groupby('name')['close'].shift(-1)

    return df


@st.cache_resource
def load_trained_models(load_lstm: bool = True):
    """Load pre-trained models from Part 2."""
    models_dir = Path(__file__).parent.parent / 'models'
    
    result = {}
    
    # Load feature columns first
    try:
        with open(models_dir / 'feature_columns.txt', 'r') as f:
            feature_cols = f.read().strip().split('\n')
        result['feature_cols'] = feature_cols
    except:
        result['feature_cols'] = ['open', 'high', 'low', 'close', 'volume',
                                   'sma_20', 'sma_50', 'rsi_14']
    
    # Load cached metrics if available
    try:
        with open(models_dir / 'model_metrics.json', 'r') as f:
            metrics = json.load(f)
            result['lr_metrics'] = metrics['lr_metrics']
            result['lstm_metrics'] = metrics['lstm_metrics']
            result['metrics_loaded'] = True
    except:
        result['metrics_loaded'] = False
    
    # Load Linear Regression (fast)
    try:
        lr_model = joblib.load(models_dir / 'linear_regression.joblib')
        result['lr_model'] = lr_model
    except Exception as e:
        st.warning(f"Could not load Linear Regression model: {e}")
    
    # Load LSTM model and scalers only if requested (slow due to TensorFlow)
    if load_lstm and TF_AVAILABLE:
        try:
            lstm_model = tf.keras.models.load_model(models_dir / 'lstm_model.keras')
            scaler_X = joblib.load(models_dir / 'scaler_X.joblib')
            scaler_y = joblib.load(models_dir / 'scaler_y.joblib')
            result['lstm_model'] = lstm_model
            result['scaler_X'] = scaler_X
            result['scaler_y'] = scaler_y
        except Exception as e:
            st.warning(f"Could not load LSTM model: {e}")
    
    return result


@st.cache_resource
def train_models(_df_hash: str):
    """Load models and metrics. Calculate metrics only if not cached."""
    models_dir = Path(__file__).parent.parent / 'models'

    # Always load all models including LSTM for predictions
    models = load_trained_models(load_lstm=True)
    feature_cols = models['feature_cols']

    # If metrics are already loaded from cache, we're done
    if models.get('metrics_loaded', False):
        return models

    # Metrics not cached - need to calculate them
    st.info("Calculating metrics for the first time... This will be cached for future runs.")
    
    # Load data for metrics calculation
    raw_df = load_data()
    df = add_technical_indicators(raw_df)
    
    # Prepare data for metrics calculation
    model_df = df.dropna(subset=feature_cols + ['next_close']).copy()
    X = model_df[feature_cols]
    y = model_df['next_close']
    
    # 80-20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Linear Regression
    if 'lr_model' not in models:
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        models['lr_model'] = lr_model
    
    lr_pred = models['lr_model'].predict(X_test)
    lr_metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, lr_pred))),
        'mae': float(mean_absolute_error(y_test, lr_pred)),
        'r2': float(r2_score(y_test, lr_pred))
    }
    
    # LSTM predictions
    if 'lstm_model' in models:
        X_test_scaled = models['scaler_X'].transform(X_test)
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        lstm_pred_scaled = models['lstm_model'].predict(X_test_lstm, verbose=0)
        lstm_pred = models['scaler_y'].inverse_transform(lstm_pred_scaled).flatten()
        
        lstm_metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, lstm_pred))),
            'mae': float(mean_absolute_error(y_test, lstm_pred)),
            'r2': float(r2_score(y_test, lstm_pred))
        }
    else:
        lstm_metrics = {'rmse': 0, 'mae': 0, 'r2': 0}
    
    models['lr_metrics'] = lr_metrics
    models['lstm_metrics'] = lstm_metrics
    
    # Save metrics to cache file for future runs
    try:
        metrics_cache = {
            'lr_metrics': lr_metrics,
            'lstm_metrics': lstm_metrics
        }
        with open(models_dir / 'model_metrics.json', 'w') as f:
            json.dump(metrics_cache, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save metrics cache: {e}")
    
    return models


def predict_for_stock(df: pd.DataFrame, ticker: str, models: dict) -> pd.DataFrame:
    """Generate predictions for a specific stock."""
    stock_df = df[df['name'] == ticker].copy()

    feature_cols = models['feature_cols']
    valid_df = stock_df.dropna(subset=feature_cols).copy()

    if len(valid_df) == 0:
        return pd.DataFrame()

    X = valid_df[feature_cols]

    # Linear Regression prediction
    valid_df['lr_prediction'] = models['lr_model'].predict(X)
    
    # LSTM prediction
    if 'lstm_model' in models:
        X_scaled = models['scaler_X'].transform(X)
        X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        lstm_pred_scaled = models['lstm_model'].predict(X_lstm, verbose=0)
        valid_df['lstm_prediction'] = models['scaler_y'].inverse_transform(lstm_pred_scaled).flatten()
    else:
        valid_df['lstm_prediction'] = valid_df['lr_prediction']

    return valid_df


def create_price_chart(df: pd.DataFrame, ticker: str):
    """Create an interactive price chart with predictions."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            f'{ticker} - Price & Predictions',
            'RSI (14-day)'
        )
    )

    # Price and predictions
    fig.add_trace(
        go.Scatter(
            x=df['date'], y=df['close'],
            name='Actual Close',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['date'], y=df['lr_prediction'],
            name='Linear Regression',
            line=dict(color='#ff7f0e', width=1.5, dash='dash')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['date'], y=df['lstm_prediction'],
            name='LSTM',
            line=dict(color='#2ca02c', width=1.5, dash='dot')
        ),
        row=1, col=1
    )

    # SMA lines
    fig.add_trace(
        go.Scatter(
            x=df['date'], y=df['sma_20'],
            name='SMA 20',
            line=dict(color='rgba(255, 165, 0, 0.5)', width=1)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df['date'], y=df['sma_50'],
            name='SMA 50',
            line=dict(color='rgba(128, 0, 128, 0.5)', width=1)
        ),
        row=1, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=df['date'], y=df['rsi_14'],
            name='RSI',
            line=dict(color='#9467bd', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(148, 103, 189, 0.2)'
        ),
        row=2, col=1
    )

    # RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified'
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig


def create_prediction_comparison(df: pd.DataFrame):
    """Create a scatter plot comparing predictions vs actual prices."""
    # Sample for performance
    sample_size = min(5000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Linear Regression', 'LSTM')
    )

    # Linear Regression
    fig.add_trace(
        go.Scatter(
            x=sample_df['next_close'],
            y=sample_df['lr_prediction'],
            mode='markers',
            marker=dict(size=4, opacity=0.5, color='#ff7f0e'),
            name='LR Predictions'
        ),
        row=1, col=1
    )

    # LSTM
    fig.add_trace(
        go.Scatter(
            x=sample_df['next_close'],
            y=sample_df['lstm_prediction'],
            mode='markers',
            marker=dict(size=4, opacity=0.5, color='#2ca02c'),
            name='LSTM Predictions'
        ),
        row=1, col=2
    )

    # Perfect prediction line
    min_val = sample_df[['next_close', 'lr_prediction', 'lstm_prediction']].min().min()
    max_val = sample_df[['next_close', 'lr_prediction', 'lstm_prediction']].max().max()

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction',
            showlegend=False
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=400,
        showlegend=True
    )

    fig.update_xaxes(title_text="Actual Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Actual Price ($)", row=1, col=2)
    fig.update_yaxes(title_text="Predicted Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Price ($)", row=1, col=2)

    return fig


def main():
    """Main dashboard application."""
    st.markdown('<h1 class="main-header">Stock Price Prediction Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**CSIS 4260 - Assignment 1** | Interactive visualization of stock price predictions")

    # Load models first (uses cached metrics - very fast)
    with st.spinner("Loading models..."):
        models = train_models("static_key")

    # Check if LSTM model is available
    if 'lstm_model' not in models:
        if not TF_AVAILABLE:
            st.warning("TensorFlow is not available. LSTM predictions will use Linear Regression values as fallback.")
        else:
            st.warning("LSTM model could not be loaded. LSTM predictions will use Linear Regression values as fallback.")

    # Load data
    with st.spinner("Loading data..."):
        raw_df = load_data()
        df = add_technical_indicators(raw_df)

    # Get unique tickers
    tickers = sorted(df['name'].unique())

    # Controls at the top (no sidebar)
    col_search, col_ticker, col_date = st.columns(3)

    with col_search:
        search = st.text_input("Search Ticker", "").upper()

    if search:
        filtered_tickers = [t for t in tickers if search in t]
    else:
        filtered_tickers = tickers

    # Handle case when no tickers match the search
    if not filtered_tickers:
        with col_ticker:
            st.selectbox("Select Ticker Code", ["No matches found"], disabled=True)
        st.warning(f"No ticker found matching '{search}'. Please try a different search term.")
        st.stop()

    with col_ticker:
        selected_ticker = st.selectbox(
            "Select Ticker Code",
            filtered_tickers,
            index=filtered_tickers.index('AAPL') if 'AAPL' in filtered_tickers else 0
        )

    # Date range filter
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()

    with col_date:
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # Model Performance section
    st.markdown("---")
    st.subheader("Model Performance")

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.markdown("**Linear Regression**")
        lr_m1, lr_m2, lr_m3 = st.columns(3)
        lr_m1.metric("R²", f"{models['lr_metrics']['r2']:.4f}")
        lr_m2.metric("RMSE", f"${models['lr_metrics']['rmse']:.2f}")
        lr_m3.metric("MAE", f"${models['lr_metrics']['mae']:.2f}")

    with perf_col2:
        st.markdown("**LSTM**")
        lstm_m1, lstm_m2, lstm_m3 = st.columns(3)
        lstm_m1.metric("R²", f"{models['lstm_metrics']['r2']:.4f}")
        lstm_m2.metric("RMSE", f"${models['lstm_metrics']['rmse']:.2f}")
        lstm_m3.metric("MAE", f"${models['lstm_metrics']['mae']:.2f}")

    st.markdown("---")

    # Main content
    # Get predictions for selected stock
    pred_df = predict_for_stock(df, selected_ticker, models)

    if len(pred_df) == 0 or 'date' not in pred_df.columns:
        st.warning(f"No data available for {selected_ticker}. Please select another ticker.")
        st.stop()

    # Filter by date
    pred_df = pred_df[
        (pred_df['date'].dt.date >= start_date) &
        (pred_df['date'].dt.date <= end_date)
    ]

    if len(pred_df) == 0:
        st.warning(f"No data available for {selected_ticker} in the selected date range.")
        st.stop()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    latest = pred_df.iloc[-1]

    with col1:
        st.metric(
            label="Latest Close",
            value=f"${latest['close']:.2f}",
            delta=f"{latest['daily_return']:.2f}%" if pd.notna(latest['daily_return']) else None
        )

    with col2:
        st.metric(
            label="LR Prediction",
            value=f"${latest['lr_prediction']:.2f}",
            delta=f"{((latest['lr_prediction'] / latest['close']) - 1) * 100:.2f}%"
        )

    with col3:
        st.metric(
            label="LSTM Prediction",
            value=f"${latest['lstm_prediction']:.2f}",
            delta=f"{((latest['lstm_prediction'] / latest['close']) - 1) * 100:.2f}%"
        )

    with col4:
        st.metric(
            label="RSI (14)",
            value=f"{latest['rsi_14']:.1f}"
        )

    # Main chart
    st.subheader(f"{selected_ticker} - Price Analysis & Predictions")
    price_chart = create_price_chart(pred_df, selected_ticker)
    st.plotly_chart(price_chart, width='stretch', key='price_chart')

    # Prediction accuracy
    st.subheader("Prediction Accuracy Analysis")

    # Only calculate for rows with actual next_close
    accuracy_df = pred_df.dropna(subset=['next_close'])

    if len(accuracy_df) > 0:
        lr_error = accuracy_df['next_close'] - accuracy_df['lr_prediction']
        lstm_error = accuracy_df['next_close'] - accuracy_df['lstm_prediction']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Linear Regression Error Distribution**")
            fig_lr = px.histogram(
                lr_error,
                nbins=50,
                title="Prediction Error (Actual - Predicted)"
            )
            fig_lr.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_lr, width='stretch', key='lr_error_hist')

        with col2:
            st.markdown("**LSTM Error Distribution**")
            fig_lstm = px.histogram(
                lstm_error,
                nbins=50,
                title="Prediction Error (Actual - Predicted)"
            )
            fig_lstm.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_lstm, width='stretch', key='lstm_error_hist')

    # Comparison scatter plot
    st.subheader("Actual vs Predicted Prices (All Stocks)")
    all_pred_df = df.copy()
    all_pred_df = all_pred_df.dropna(subset=models['feature_cols'] + ['next_close'])
    X_all = all_pred_df[models['feature_cols']]
    all_pred_df['lr_prediction'] = models['lr_model'].predict(X_all)
    if 'lstm_model' in models:
        X_all_scaled = models['scaler_X'].transform(X_all)
        X_all_lstm = X_all_scaled.reshape((X_all_scaled.shape[0], 1, X_all_scaled.shape[1]))
        all_pred_df['lstm_prediction'] = models['scaler_y'].inverse_transform(
            models['lstm_model'].predict(X_all_lstm, verbose=0)
        ).flatten()
    else:
        all_pred_df['lstm_prediction'] = all_pred_df['lr_prediction']

    comparison_chart = create_prediction_comparison(all_pred_df)
    st.plotly_chart(comparison_chart, width='stretch', key='comparison_chart')

    # Data table
    st.subheader("Recent Data")
    display_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                    'sma_20', 'rsi_14', 'lr_prediction', 'lstm_prediction']
    st.dataframe(
        pred_df[display_cols].tail(20).round(2),
        width='stretch'
    )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Technical Indicators:**
        - **SMA**: Simple Moving Average (20-day & 50-day)
        - **RSI**: Relative Strength Index (14-day) - Overbought >70, Oversold <30

        **Models:**
        - **Linear Regression**: Baseline model using all features
        - **LSTM**: Deep learning model for time-series prediction (2 LSTM layers + Dense layers)
        """
    )


if __name__ == "__main__":
    main()
