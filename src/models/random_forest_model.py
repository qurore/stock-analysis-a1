"""
Random Forest Model for Stock Price Prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional

from .feature_engineering import FEATURE_COLUMNS


class RandomForestModel:
    """
    Random Forest Regressor for predicting next-day stock closing prices.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 10,
        random_state: int = 42
    ):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_columns = FEATURE_COLUMNS
        self.is_fitted = False
        self.metrics = {}

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target from dataframe.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features and target (next_close)

        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            Features (X) and target (y)
        """
        # Drop rows with NaN in required columns
        required_cols = self.feature_columns + ['next_close']
        clean_df = df.dropna(subset=required_cols)

        X = clean_df[self.feature_columns]
        y = clean_df['next_close']

        return X, y

    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Train the model using 80-20 split.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with all features
        test_size : float
            Proportion of data to use for testing

        Returns:
        --------
        Dict
            Dictionary containing training metrics
        """
        X, y = self.prepare_data(df)

        # Time-based split
        split_idx = int(len(X) * (1 - test_size))

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test)

        self.metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        return self.metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions for next-day closing prices.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with feature columns

        Returns:
        --------
        np.ndarray
            Predicted prices
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")

        X = df[self.feature_columns]
        return self.model.predict(X)

    def predict_single(self, features: Dict) -> float:
        """
        Predict for a single data point.

        Parameters:
        -----------
        features : Dict
            Dictionary of feature values

        Returns:
        --------
        float
            Predicted next-day closing price
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")

        X = pd.DataFrame([features])[self.feature_columns]
        return self.model.predict(X)[0]

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from Random Forest.

        Returns:
        --------
        pd.DataFrame
            DataFrame with features and their importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")

        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        })

        return importance_df.sort_values('importance', ascending=False)

    def save(self, path: str):
        """Save model to file."""
        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load model from file."""
        self.model = joblib.load(path)
        self.is_fitted = True
