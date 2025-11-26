"""Data preprocessing module for the ML Loan Eligibility Platform."""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ..utils.logging import LoggerMixin


class DataPreprocessor(LoggerMixin):
    """Preprocessor for data cleaning and transformation."""

    def __init__(self):
        """Initialize data preprocessor."""
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.logger.info("data_preprocessor_initialized")

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "auto",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Handle missing values in DataFrame.

        Args:
            df: DataFrame to process
            strategy: Strategy for handling missing values
                     ('auto', 'drop', 'mean', 'median', 'mode', 'forward_fill')
            columns: Specific columns to process (if None, processes all)

        Returns:
            DataFrame with missing values handled
        """
        df_copy = df.copy()
        columns = columns or df.columns.tolist()

        for col in columns:
            if col not in df_copy.columns:
                continue

            missing_count = df_copy[col].isna().sum()
            if missing_count == 0:
                continue

            if strategy == "auto":
                # Choose strategy based on data type and missing percentage
                missing_pct = missing_count / len(df_copy)

                if missing_pct > 0.5:
                    # Too many missing values, drop column
                    self.logger.warning(
                        "dropping_column_too_many_missing",
                        column=col,
                        missing_percentage=missing_pct,
                    )
                    df_copy = df_copy.drop(columns=[col])
                    continue

                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    # Use median for numeric columns
                    fill_value = df_copy[col].median()
                    df_copy[col].fillna(fill_value, inplace=True)
                else:
                    # Use mode for categorical columns
                    fill_value = df_copy[col].mode()[0] if not df_copy[col].mode().empty else "Unknown"
                    df_copy[col].fillna(fill_value, inplace=True)

            elif strategy == "drop":
                df_copy = df_copy.dropna(subset=[col])

            elif strategy == "mean":
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col].fillna(df_copy[col].mean(), inplace=True)

            elif strategy == "median":
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    df_copy[col].fillna(df_copy[col].median(), inplace=True)

            elif strategy == "mode":
                mode_val = df_copy[col].mode()[0] if not df_copy[col].mode().empty else "Unknown"
                df_copy[col].fillna(mode_val, inplace=True)

            elif strategy == "forward_fill":
                df_copy[col].fillna(method="ffill", inplace=True)

            self.logger.info(
                "missing_values_handled",
                column=col,
                strategy=strategy,
                missing_count=missing_count,
            )

        return df_copy

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "first",
    ) -> pd.DataFrame:
        """Remove duplicate rows.

        Args:
            df: DataFrame to process
            subset: Columns to consider for identifying duplicates
            keep: Which duplicates to keep ('first', 'last', False)

        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)
        df_copy = df.drop_duplicates(subset=subset, keep=keep)
        removed_count = initial_count - len(df_copy)

        self.logger.info(
            "duplicates_removed",
            initial_rows=initial_count,
            final_rows=len(df_copy),
            removed=removed_count,
        )

        return df_copy

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> pd.DataFrame:
        """Remove outliers from numeric columns.

        Args:
            df: DataFrame to process
            columns: Columns to check for outliers
            method: Method for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection (IQR multiplier or z-score)

        Returns:
            DataFrame with outliers removed
        """
        df_copy = df.copy()
        initial_count = len(df_copy)

        for col in columns:
            if col not in df_copy.columns or not pd.api.types.is_numeric_dtype(df_copy[col]):
                continue

            if method == "iqr":
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                mask = (df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)
                df_copy = df_copy[mask]

            elif method == "zscore":
                z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                df_copy = df_copy[z_scores < threshold]

        removed_count = initial_count - len(df_copy)
        self.logger.info(
            "outliers_removed",
            method=method,
            initial_rows=initial_count,
            final_rows=len(df_copy),
            removed=removed_count,
        )

        return df_copy

    def normalize_numeric_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "standard",
    ) -> pd.DataFrame:
        """Normalize numeric features.

        Args:
            df: DataFrame to process
            columns: Columns to normalize (if None, normalizes all numeric columns)
            method: Normalization method ('standard' or 'minmax')

        Returns:
            DataFrame with normalized features
        """
        df_copy = df.copy()

        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col not in df_copy.columns:
                continue

            if method == "standard":
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df_copy[col] = self.scalers[col].fit_transform(df_copy[[col]])
                else:
                    df_copy[col] = self.scalers[col].transform(df_copy[[col]])

            elif method == "minmax":
                min_val = df_copy[col].min()
                max_val = df_copy[col].max()
                if max_val > min_val:
                    df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)

        self.logger.info(
            "features_normalized",
            method=method,
            columns=columns,
        )

        return df_copy

    def encode_categorical_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "label",
    ) -> pd.DataFrame:
        """Encode categorical features.

        Args:
            df: DataFrame to process
            columns: Columns to encode (if None, encodes all object columns)
            method: Encoding method ('label' or 'onehot')

        Returns:
            DataFrame with encoded features
        """
        df_copy = df.copy()

        if columns is None:
            columns = df_copy.select_dtypes(include=["object"]).columns.tolist()

        for col in columns:
            if col not in df_copy.columns:
                continue

            if method == "label":
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_copy[col] = self.encoders[col].fit_transform(df_copy[col].astype(str))
                else:
                    df_copy[col] = self.encoders[col].transform(df_copy[col].astype(str))

            elif method == "onehot":
                dummies = pd.get_dummies(df_copy[col], prefix=col)
                df_copy = pd.concat([df_copy, dummies], axis=1)
                df_copy = df_copy.drop(columns=[col])

        self.logger.info(
            "features_encoded",
            method=method,
            columns=columns,
        )

        return df_copy

    def create_time_features(
        self,
        df: pd.DataFrame,
        date_column: str,
    ) -> pd.DataFrame:
        """Create time-based features from datetime column.

        Args:
            df: DataFrame to process
            date_column: Name of datetime column

        Returns:
            DataFrame with additional time features
        """
        df_copy = df.copy()

        if date_column not in df_copy.columns:
            self.logger.warning("date_column_not_found", column=date_column)
            return df_copy

        # Ensure column is datetime type
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])

        # Extract time features
        df_copy[f"{date_column}_year"] = df_copy[date_column].dt.year
        df_copy[f"{date_column}_month"] = df_copy[date_column].dt.month
        df_copy[f"{date_column}_day"] = df_copy[date_column].dt.day
        df_copy[f"{date_column}_dayofweek"] = df_copy[date_column].dt.dayofweek
        df_copy[f"{date_column}_hour"] = df_copy[date_column].dt.hour
        df_copy[f"{date_column}_is_weekend"] = df_copy[date_column].dt.dayofweek.isin([5, 6]).astype(int)
        df_copy[f"{date_column}_quarter"] = df_copy[date_column].dt.quarter

        self.logger.info(
            "time_features_created",
            date_column=date_column,
            features_added=7,
        )

        return df_copy

    def aggregate_user_transactions(
        self,
        df: pd.DataFrame,
        user_id_col: str = "user_id",
        date_col: str = "transaction_date",
        amount_col: str = "amount",
    ) -> pd.DataFrame:
        """Aggregate transaction data by user.

        Args:
            df: Transaction DataFrame
            user_id_col: Name of user ID column
            date_col: Name of date column
            amount_col: Name of amount column

        Returns:
            DataFrame with aggregated user-level features
        """
        agg_features = df.groupby(user_id_col).agg({
            amount_col: ["count", "sum", "mean", "std", "min", "max"],
            date_col: ["min", "max"],
        })

        # Flatten column names
        agg_features.columns = [f"{col[0]}_{col[1]}" for col in agg_features.columns]
        agg_features = agg_features.reset_index()

        # Calculate transaction frequency
        agg_features["transaction_period_days"] = (
            (agg_features[f"{date_col}_max"] - agg_features[f"{date_col}_min"]).dt.days
        )

        self.logger.info(
            "transactions_aggregated",
            users=len(agg_features),
            features=len(agg_features.columns),
        )

        return agg_features
