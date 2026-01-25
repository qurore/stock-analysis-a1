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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
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

    # Volatility
    df['volatility'] = df.groupby('name')['close'].transform(
        lambda x: x.rolling(window=20, min_periods=1).std()
    )

    # Price momentum
    df['price_momentum'] = df['close'] - df['sma_20']

    # Daily return
    df['daily_return'] = df.groupby('name')['close'].transform(
        lambda x: x.pct_change() * 100
    )

    # Target: next day's close
    df['next_close'] = df.groupby('name')['close'].shift(-1)

    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    """Train both prediction models."""
    feature_cols = ['open', 'high', 'low', 'close', 'volume',
                    'sma_20', 'sma_50', 'rsi_14', 'volatility', 'price_momentum']

    # Prepare data
    model_df = df.dropna(subset=feature_cols + ['next_close']).copy()

    X = model_df[feature_cols]
    y = model_df['next_close']

    # 80-20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    lr_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'mae': mean_absolute_error(y_test, lr_pred),
        'r2': r2_score(y_test, lr_pred)
    }

    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    rf_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'mae': mean_absolute_error(y_test, rf_pred),
        'r2': r2_score(y_test, rf_pred)
    }

    return {
        'lr_model': lr_model,
        'rf_model': rf_model,
        'lr_metrics': lr_metrics,
        'rf_metrics': rf_metrics,
        'feature_cols': feature_cols,
        'test_idx': model_df.index[split_idx:]
    }


def predict_for_stock(df: pd.DataFrame, ticker: str, models: dict) -> pd.DataFrame:
    """Generate predictions for a specific stock."""
    stock_df = df[df['name'] == ticker].copy()

    feature_cols = models['feature_cols']
    valid_df = stock_df.dropna(subset=feature_cols).copy()

    if len(valid_df) == 0:
        return pd.DataFrame()

    X = valid_df[feature_cols]

    valid_df['lr_prediction'] = models['lr_model'].predict(X)
    valid_df['rf_prediction'] = models['rf_model'].predict(X)

    return valid_df


def create_price_chart(df: pd.DataFrame, ticker: str):
    """Create an interactive price chart with predictions."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            f'{ticker} - Price & Predictions',
            'RSI (14-day)',
            'Volatility (20-day)'
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
            x=df['date'], y=df['rf_prediction'],
            name='Random Forest',
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

    # Volatility
    fig.add_trace(
        go.Scatter(
            x=df['date'], y=df['volatility'],
            name='Volatility',
            line=dict(color='#d62728', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(214, 39, 40, 0.2)'
        ),
        row=3, col=1
    )

    fig.update_layout(
        height=800,
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
    fig.update_yaxes(title_text="Volatility", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    return fig


def create_prediction_comparison(df: pd.DataFrame):
    """Create a scatter plot comparing predictions vs actual prices."""
    # Sample for performance
    sample_size = min(5000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Linear Regression', 'Random Forest')
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

    # Random Forest
    fig.add_trace(
        go.Scatter(
            x=sample_df['next_close'],
            y=sample_df['rf_prediction'],
            mode='markers',
            marker=dict(size=4, opacity=0.5, color='#2ca02c'),
            name='RF Predictions'
        ),
        row=1, col=2
    )

    # Perfect prediction line
    min_val = sample_df[['next_close', 'lr_prediction', 'rf_prediction']].min().min()
    max_val = sample_df[['next_close', 'lr_prediction', 'rf_prediction']].max().max()

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

    # Load data
    with st.spinner("Loading data..."):
        raw_df = load_data()
        df = add_technical_indicators(raw_df)

    # Train models
    with st.spinner("Training models..."):
        models = train_models(df)

    # Sidebar
    st.sidebar.header("Stock Selection")

    # Get unique tickers
    tickers = sorted(df['name'].unique())

    # Search functionality
    search = st.sidebar.text_input("Search Ticker", "").upper()
    if search:
        filtered_tickers = [t for t in tickers if search in t]
    else:
        filtered_tickers = tickers

    selected_ticker = st.sidebar.selectbox(
        "Select Company",
        filtered_tickers,
        index=filtered_tickers.index('AAPL') if 'AAPL' in filtered_tickers else 0
    )

    # Date range filter
    st.sidebar.subheader("Date Range")
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()

    date_range = st.sidebar.date_input(
        "Select Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    # Model metrics section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Performance")

    st.sidebar.markdown("**Linear Regression**")
    st.sidebar.text(f"R²: {models['lr_metrics']['r2']:.4f}")
    st.sidebar.text(f"RMSE: ${models['lr_metrics']['rmse']:.2f}")
    st.sidebar.text(f"MAE: ${models['lr_metrics']['mae']:.2f}")

    st.sidebar.markdown("**Random Forest**")
    st.sidebar.text(f"R²: {models['rf_metrics']['r2']:.4f}")
    st.sidebar.text(f"RMSE: ${models['rf_metrics']['rmse']:.2f}")
    st.sidebar.text(f"MAE: ${models['rf_metrics']['mae']:.2f}")

    # Main content
    # Get predictions for selected stock
    pred_df = predict_for_stock(df, selected_ticker, models)

    # Filter by date
    pred_df = pred_df[
        (pred_df['date'].dt.date >= start_date) &
        (pred_df['date'].dt.date <= end_date)
    ]

    if len(pred_df) == 0:
        st.warning(f"No data available for {selected_ticker} in the selected date range.")
        st.stop()

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

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
            label="RF Prediction",
            value=f"${latest['rf_prediction']:.2f}",
            delta=f"{((latest['rf_prediction'] / latest['close']) - 1) * 100:.2f}%"
        )

    with col4:
        st.metric(
            label="RSI (14)",
            value=f"{latest['rsi_14']:.1f}"
        )

    with col5:
        st.metric(
            label="Volatility",
            value=f"${latest['volatility']:.2f}"
        )

    # Main chart
    st.subheader(f"{selected_ticker} - Price Analysis & Predictions")
    price_chart = create_price_chart(pred_df, selected_ticker)
    st.plotly_chart(price_chart, use_container_width=True)

    # Prediction accuracy
    st.subheader("Prediction Accuracy Analysis")

    # Only calculate for rows with actual next_close
    accuracy_df = pred_df.dropna(subset=['next_close'])

    if len(accuracy_df) > 0:
        lr_error = accuracy_df['next_close'] - accuracy_df['lr_prediction']
        rf_error = accuracy_df['next_close'] - accuracy_df['rf_prediction']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Linear Regression Error Distribution**")
            fig_lr = px.histogram(
                lr_error,
                nbins=50,
                title="Prediction Error (Actual - Predicted)"
            )
            fig_lr.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_lr, use_container_width=True)

        with col2:
            st.markdown("**Random Forest Error Distribution**")
            fig_rf = px.histogram(
                rf_error,
                nbins=50,
                title="Prediction Error (Actual - Predicted)"
            )
            fig_rf.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_rf, use_container_width=True)

    # Comparison scatter plot
    st.subheader("Actual vs Predicted Prices (All Stocks)")
    all_pred_df = df.copy()
    all_pred_df = all_pred_df.dropna(subset=models['feature_cols'] + ['next_close'])
    X_all = all_pred_df[models['feature_cols']]
    all_pred_df['lr_prediction'] = models['lr_model'].predict(X_all)
    all_pred_df['rf_prediction'] = models['rf_model'].predict(X_all)

    comparison_chart = create_prediction_comparison(all_pred_df)
    st.plotly_chart(comparison_chart, use_container_width=True)

    # Data table
    st.subheader("Recent Data")
    display_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                    'sma_20', 'rsi_14', 'lr_prediction', 'rf_prediction']
    st.dataframe(
        pred_df[display_cols].tail(20).round(2),
        use_container_width=True
    )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Technical Indicators:**
        - **SMA**: Simple Moving Average (20-day & 50-day)
        - **RSI**: Relative Strength Index (14-day) - Overbought >70, Oversold <30
        - **Volatility**: 20-day rolling standard deviation of closing prices

        **Models:**
        - **Linear Regression**: Baseline model using all features
        - **Random Forest**: Ensemble method with 100 trees, max depth 10
        """
    )


if __name__ == "__main__":
    main()
