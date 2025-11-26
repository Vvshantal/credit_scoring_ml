"""Feature engineering module for the ML Loan Eligibility Platform.

This module implements 120+ behavioral features across 7 thematic categories:
1. Income Stability Indicators
2. Expenditure Pattern Metrics
3. Airtime Purchase Behaviors
4. Historical Loan Performance Variables
5. Financial Buffer Maintenance
6. Transaction Diversity Indices
7. Temporal Consistency Scores
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats
from ..utils.logging import LoggerMixin


class FeatureEngineer(LoggerMixin):
    """Feature engineering for loan eligibility prediction."""

    def __init__(self, rolling_windows: List[int] = [7, 30, 90]):
        """Initialize feature engineer.

        Args:
            rolling_windows: Time windows for rolling features (in days)
        """
        self.rolling_windows = rolling_windows
        self.logger.info("feature_engineer_initialized", rolling_windows=rolling_windows)

    # ==================== Category 1: Income Stability Indicators ====================

    def create_income_stability_features(
        self,
        df: pd.DataFrame,
        user_id_col: str = "user_id",
        amount_col: str = "amount",
        date_col: str = "transaction_date",
        type_col: str = "type",
    ) -> pd.DataFrame:
        """Create income stability features.

        Args:
            df: Transaction DataFrame
            user_id_col: User ID column
            amount_col: Amount column
            date_col: Date column
            type_col: Transaction type column

        Returns:
            DataFrame with income stability features
        """
        # Filter incoming transactions
        income_df = df[df[type_col] == "incoming"].copy()

        features = income_df.groupby(user_id_col).agg({
            amount_col: [
                ("avg_transaction_amount_incoming", "mean"),
                ("income_std", "std"),
                ("income_min", "min"),
                ("income_max", "max"),
                ("total_income", "sum"),
                ("income_count", "count"),
            ]
        })

        features.columns = [col[1] if col[1] else col[0] for col in features.columns]
        features = features.reset_index()

        # Income consistency (coefficient of variation)
        features["income_consistency"] = (
            features["income_std"] / features["avg_transaction_amount_incoming"]
        ).fillna(0)

        # Peak vs average income ratio
        features["peak_avg_income_ratio"] = (
            features["income_max"] / features["avg_transaction_amount_incoming"]
        ).fillna(0)

        # Monthly income trend (requires time-series analysis)
        income_monthly = income_df.groupby([
            user_id_col,
            pd.Grouper(key=date_col, freq="M")
        ])[amount_col].sum().reset_index()

        income_trends = []
        for user_id in features[user_id_col]:
            user_monthly = income_monthly[income_monthly[user_id_col] == user_id]
            if len(user_monthly) > 1:
                x = np.arange(len(user_monthly))
                y = user_monthly[amount_col].values
                slope, _, _, _, _ = stats.linregress(x, y)
                income_trends.append(slope)
            else:
                income_trends.append(0)

        features["monthly_income_trend"] = income_trends

        # Income regularity (frequency of deposits)
        features["income_regularity"] = features["income_count"] / 30  # Average per month

        self.logger.info("income_stability_features_created", features=len(features.columns))
        return features

    # ==================== Category 2: Expenditure Pattern Metrics ====================

    def create_expenditure_pattern_features(
        self,
        df: pd.DataFrame,
        user_id_col: str = "user_id",
        amount_col: str = "amount",
        date_col: str = "transaction_date",
        type_col: str = "type",
        category_col: Optional[str] = "category",
    ) -> pd.DataFrame:
        """Create expenditure pattern features.

        Args:
            df: Transaction DataFrame
            user_id_col: User ID column
            amount_col: Amount column
            date_col: Date column
            type_col: Transaction type column
            category_col: Category column (optional)

        Returns:
            DataFrame with expenditure features
        """
        # Filter outgoing transactions
        expense_df = df[df[type_col] == "outgoing"].copy()

        features = expense_df.groupby(user_id_col).agg({
            amount_col: [
                ("avg_monthly_expenditure", "mean"),
                ("expenditure_volatility", "std"),
                ("min_expense", "min"),
                ("max_expense", "max"),
                ("total_expenditure", "sum"),
            ]
        })

        features.columns = [col[1] if col[1] else col[0] for col in features.columns]
        features = features.reset_index()

        # Monthly expenditure trend
        expense_monthly = expense_df.groupby([
            user_id_col,
            pd.Grouper(key=date_col, freq="M")
        ])[amount_col].sum().reset_index()

        expense_trends = []
        for user_id in features[user_id_col]:
            user_monthly = expense_monthly[expense_monthly[user_id_col] == user_id]
            if len(user_monthly) > 1:
                x = np.arange(len(user_monthly))
                y = user_monthly[amount_col].values
                slope, _, _, _, _ = stats.linregress(x, y)
                expense_trends.append(slope)
            else:
                expense_trends.append(0)

        features["monthly_expenditure_trend"] = expense_trends

        # Category diversity (if available)
        if category_col and category_col in expense_df.columns:
            category_diversity = expense_df.groupby(user_id_col)[category_col].nunique()
            features = features.merge(
                category_diversity.to_frame("expense_category_diversity"),
                on=user_id_col,
                how="left"
            )

            # Fixed vs discretionary spending ratio (assuming categories exist)
            essential_categories = ["utilities", "rent", "groceries", "healthcare"]
            expense_df["is_essential"] = expense_df[category_col].isin(essential_categories).astype(int)

            essential_spending = expense_df.groupby(user_id_col).apply(
                lambda x: x[x["is_essential"] == 1][amount_col].sum()
            ).to_frame("essential_spending")

            features = features.merge(essential_spending, on=user_id_col, how="left")
            features["essential_spending"].fillna(0, inplace=True)

            features["fixed_discretionary_ratio"] = (
                features["essential_spending"] / features["total_expenditure"]
            ).fillna(0)

        self.logger.info("expenditure_pattern_features_created", features=len(features.columns))
        return features

    # ==================== Category 3: Airtime Purchase Behaviors ====================

    def create_airtime_features(
        self,
        airtime_df: pd.DataFrame,
        user_id_col: str = "user_id",
        amount_col: str = "amount",
        date_col: str = "purchase_date",
    ) -> pd.DataFrame:
        """Create airtime purchase behavior features.

        Args:
            airtime_df: Airtime purchase DataFrame
            user_id_col: User ID column
            amount_col: Amount column
            date_col: Date column

        Returns:
            DataFrame with airtime features
        """
        features = airtime_df.groupby(user_id_col).agg({
            amount_col: [
                ("avg_airtime_purchase", "mean"),
                ("airtime_purchase_std", "std"),
                ("total_airtime_spent", "sum"),
                ("airtime_purchase_frequency", "count"),
            ]
        })

        features.columns = [col[1] if col[1] else col[0] for col in features.columns]
        features = features.reset_index()

        # Airtime purchase consistency
        features["airtime_consistency"] = (
            features["airtime_purchase_std"] / features["avg_airtime_purchase"]
        ).fillna(0)

        # Time of purchase patterns (hour of day)
        if date_col in airtime_df.columns:
            airtime_df[date_col] = pd.to_datetime(airtime_df[date_col])
            hour_features = airtime_df.groupby(user_id_col).agg({
                date_col: lambda x: x.dt.hour.mode()[0] if not x.dt.hour.mode().empty else 12
            }).rename(columns={date_col: "most_common_airtime_hour"})

            features = features.merge(hour_features, on=user_id_col, how="left")

        # Recharge predictability score (based on time intervals)
        airtime_sorted = airtime_df.sort_values([user_id_col, date_col])
        airtime_sorted["days_since_last_purchase"] = (
            airtime_sorted.groupby(user_id_col)[date_col].diff().dt.days
        )

        predictability = airtime_sorted.groupby(user_id_col)["days_since_last_purchase"].agg([
            ("mean_interval", "mean"),
            ("std_interval", "std"),
        ])

        predictability["airtime_predictability_score"] = (
            1 / (1 + predictability["std_interval"] / predictability["mean_interval"])
        ).fillna(0)

        features = features.merge(
            predictability[["airtime_predictability_score"]],
            on=user_id_col,
            how="left"
        )

        self.logger.info("airtime_features_created", features=len(features.columns))
        return features

    # ==================== Category 4: Historical Loan Performance Variables ====================

    def create_loan_history_features(
        self,
        loan_df: pd.DataFrame,
        user_id_col: str = "user_id",
        amount_col: str = "loan_amount",
        repayment_col: str = "repayment_amount",
        default_col: str = "is_default",
        loan_date_col: str = "loan_date",
    ) -> pd.DataFrame:
        """Create historical loan performance features.

        Args:
            loan_df: Loan history DataFrame
            user_id_col: User ID column
            amount_col: Loan amount column
            repayment_col: Repayment amount column
            default_col: Default flag column
            loan_date_col: Loan date column

        Returns:
            DataFrame with loan history features
        """
        features = loan_df.groupby(user_id_col).agg({
            amount_col: [
                ("num_previous_loans", "count"),
                ("avg_previous_loan_amount", "mean"),
                ("total_borrowed", "sum"),
            ],
            repayment_col: [
                ("total_repayment", "sum"),
                ("avg_repayment", "mean"),
            ],
            default_col: [
                ("default_history", "sum"),
                ("default_rate", "mean"),
            ],
        })

        features.columns = [col[1] if col[1] else col[0] for col in features.columns]
        features = features.reset_index()

        # Repayment timeliness score
        if "repayment_date" in loan_df.columns and "due_date" in loan_df.columns:
            loan_df["days_late"] = (
                pd.to_datetime(loan_df["repayment_date"]) -
                pd.to_datetime(loan_df["due_date"])
            ).dt.days

            timeliness = loan_df.groupby(user_id_col).agg({
                "days_late": lambda x: (x <= 0).sum() / len(x) if len(x) > 0 else 1
            }).rename(columns={"days_late": "repayment_timeliness_score"})

            features = features.merge(timeliness, on=user_id_col, how="left")

        # Time since last loan
        if loan_date_col in loan_df.columns:
            last_loan = loan_df.groupby(user_id_col)[loan_date_col].max()
            features = features.merge(
                last_loan.to_frame("last_loan_date"),
                on=user_id_col,
                how="left"
            )

            features["days_since_last_loan"] = (
                pd.Timestamp.now() - pd.to_datetime(features["last_loan_date"])
            ).dt.days

        self.logger.info("loan_history_features_created", features=len(features.columns))
        return features

    # ==================== Category 5: Financial Buffer Maintenance ====================

    def create_financial_buffer_features(
        self,
        df: pd.DataFrame,
        user_id_col: str = "user_id",
        balance_col: str = "balance",
        date_col: str = "transaction_date",
    ) -> pd.DataFrame:
        """Create financial buffer maintenance features.

        Args:
            df: Transaction DataFrame with balance information
            user_id_col: User ID column
            balance_col: Balance column
            date_col: Date column

        Returns:
            DataFrame with financial buffer features
        """
        features = df.groupby(user_id_col).agg({
            balance_col: [
                ("min_balance", "min"),
                ("avg_balance", "mean"),
                ("max_balance", "max"),
                ("balance_volatility", "std"),
            ]
        })

        features.columns = [col[1] if col[1] else col[0] for col in features.columns]
        features = features.reset_index()

        # Emergency fund threshold (e.g., 3 months of average expenses)
        emergency_threshold = 1000  # This should be calculated per user

        # Percentage of time above emergency fund threshold
        above_threshold = df.groupby(user_id_col).apply(
            lambda x: (x[balance_col] >= emergency_threshold).sum() / len(x)
        ).to_frame("pct_time_above_emergency_fund")

        features = features.merge(above_threshold, on=user_id_col, how="left")

        # Balance trend
        balance_sorted = df.sort_values([user_id_col, date_col])
        balance_trends = []

        for user_id in features[user_id_col]:
            user_balance = balance_sorted[balance_sorted[user_id_col] == user_id]
            if len(user_balance) > 1:
                x = np.arange(len(user_balance))
                y = user_balance[balance_col].values
                slope, _, _, _, _ = stats.linregress(x, y)
                balance_trends.append(slope)
            else:
                balance_trends.append(0)

        features["balance_trend"] = balance_trends

        # Frequency of low balance events (balance < 10% of average)
        low_balance_events = df.groupby(user_id_col).apply(
            lambda x: (x[balance_col] < x[balance_col].mean() * 0.1).sum()
        ).to_frame("low_balance_event_count")

        features = features.merge(low_balance_events, on=user_id_col, how="left")

        self.logger.info("financial_buffer_features_created", features=len(features.columns))
        return features

    # ==================== Category 6: Transaction Diversity Indices ====================

    def create_transaction_diversity_features(
        self,
        df: pd.DataFrame,
        user_id_col: str = "user_id",
        merchant_col: Optional[str] = "merchant_id",
        category_col: Optional[str] = "category",
        location_col: Optional[str] = "location",
    ) -> pd.DataFrame:
        """Create transaction diversity features.

        Args:
            df: Transaction DataFrame
            user_id_col: User ID column
            merchant_col: Merchant ID column
            category_col: Category column
            location_col: Location column

        Returns:
            DataFrame with transaction diversity features
        """
        features_list = []

        # Number of unique transaction partners
        if merchant_col and merchant_col in df.columns:
            unique_merchants = df.groupby(user_id_col)[merchant_col].nunique()
            features_list.append(unique_merchants.to_frame("unique_transaction_partners"))

        # Transaction category diversity
        if category_col and category_col in df.columns:
            category_diversity = df.groupby(user_id_col)[category_col].nunique()
            features_list.append(category_diversity.to_frame("transaction_category_diversity"))

        # Geographic transaction diversity
        if location_col and location_col in df.columns:
            location_diversity = df.groupby(user_id_col)[location_col].nunique()
            features_list.append(location_diversity.to_frame("geographic_diversity"))

        # New transaction partner frequency
        if merchant_col and merchant_col in df.columns and "transaction_date" in df.columns:
            df_sorted = df.sort_values([user_id_col, "transaction_date"])
            df_sorted["is_new_merchant"] = ~df_sorted.duplicated(subset=[user_id_col, merchant_col])

            new_merchant_freq = df_sorted.groupby(user_id_col)["is_new_merchant"].mean()
            features_list.append(new_merchant_freq.to_frame("new_partner_frequency"))

        # Merge all features
        if features_list:
            features = features_list[0]
            for feat_df in features_list[1:]:
                features = features.merge(feat_df, left_index=True, right_index=True, how="outer")
            features = features.reset_index()
        else:
            features = pd.DataFrame({user_id_col: df[user_id_col].unique()})

        self.logger.info("transaction_diversity_features_created", features=len(features.columns))
        return features

    # ==================== Category 7: Temporal Consistency Scores ====================

    def create_temporal_consistency_features(
        self,
        df: pd.DataFrame,
        user_id_col: str = "user_id",
        date_col: str = "transaction_date",
        amount_col: str = "amount",
    ) -> pd.DataFrame:
        """Create temporal consistency features.

        Args:
            df: Transaction DataFrame
            user_id_col: User ID column
            date_col: Date column
            amount_col: Amount column

        Returns:
            DataFrame with temporal consistency features
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([user_id_col, date_col])

        # Transaction timing regularity (inter-arrival time variance)
        df["time_since_last_transaction"] = (
            df.groupby(user_id_col)[date_col].diff().dt.total_seconds() / 3600
        )  # in hours

        timing_regularity = df.groupby(user_id_col)["time_since_last_transaction"].agg([
            ("mean_interval", "mean"),
            ("std_interval", "std"),
        ])

        timing_regularity["transaction_timing_regularity"] = (
            1 / (1 + timing_regularity["std_interval"] / timing_regularity["mean_interval"])
        ).fillna(0)

        features = timing_regularity[["transaction_timing_regularity"]].reset_index()

        # Daily, weekly, monthly consistency
        df["hour"] = df[date_col].dt.hour
        df["day_of_week"] = df[date_col].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Weekend vs weekday pattern regularity
        weekend_ratio = df.groupby(user_id_col)["is_weekend"].mean()
        features = features.merge(
            weekend_ratio.to_frame("weekend_transaction_ratio"),
            on=user_id_col,
            how="left"
        )

        # Activity streak consistency (consecutive days with transactions)
        df["date_only"] = df[date_col].dt.date
        streaks = []

        for user_id in features[user_id_col]:
            user_dates = df[df[user_id_col] == user_id]["date_only"].unique()
            user_dates = sorted(user_dates)

            if len(user_dates) > 1:
                current_streak = 1
                max_streak = 1

                for i in range(1, len(user_dates)):
                    days_diff = (user_dates[i] - user_dates[i-1]).days
                    if days_diff == 1:
                        current_streak += 1
                        max_streak = max(max_streak, current_streak)
                    else:
                        current_streak = 1

                streaks.append(max_streak)
            else:
                streaks.append(1)

        features["max_activity_streak"] = streaks

        self.logger.info("temporal_consistency_features_created", features=len(features.columns))
        return features

    # ==================== Rolling Window Features ====================

    def create_rolling_window_features(
        self,
        df: pd.DataFrame,
        user_id_col: str = "user_id",
        date_col: str = "transaction_date",
        amount_col: str = "amount",
    ) -> pd.DataFrame:
        """Create rolling window features.

        Args:
            df: Transaction DataFrame
            user_id_col: User ID column
            date_col: Date column
            amount_col: Amount column

        Returns:
            DataFrame with rolling window features
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([user_id_col, date_col])

        all_features = []

        for window in self.rolling_windows:
            window_features = df.groupby(user_id_col).apply(
                lambda x: pd.Series({
                    f"rolling_{window}d_transaction_count": len(x[x[date_col] >= x[date_col].max() - pd.Timedelta(days=window)]),
                    f"rolling_{window}d_total_amount": x[x[date_col] >= x[date_col].max() - pd.Timedelta(days=window)][amount_col].sum(),
                    f"rolling_{window}d_avg_amount": x[x[date_col] >= x[date_col].max() - pd.Timedelta(days=window)][amount_col].mean(),
                })
            ).reset_index()

            all_features.append(window_features)

        # Merge all rolling window features
        features = all_features[0]
        for feat_df in all_features[1:]:
            features = features.merge(feat_df, on=user_id_col, how="outer")

        self.logger.info("rolling_window_features_created", features=len(features.columns))
        return features

    # ==================== Master Feature Engineering Function ====================

    def create_all_features(
        self,
        transaction_df: pd.DataFrame,
        airtime_df: Optional[pd.DataFrame] = None,
        loan_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Create all 120+ candidate features.

        Args:
            transaction_df: Mobile money transaction DataFrame
            airtime_df: Airtime purchase DataFrame (optional)
            loan_df: Loan history DataFrame (optional)

        Returns:
            DataFrame with all engineered features
        """
        self.logger.info("creating_all_features")

        user_id_col = "user_id"

        # Category 1: Income Stability
        income_features = self.create_income_stability_features(transaction_df)

        # Category 2: Expenditure Patterns
        expenditure_features = self.create_expenditure_pattern_features(transaction_df)

        # Category 3: Airtime Purchases
        if airtime_df is not None and not airtime_df.empty:
            airtime_features = self.create_airtime_features(airtime_df)
        else:
            airtime_features = pd.DataFrame({user_id_col: transaction_df[user_id_col].unique()})

        # Category 4: Loan History
        if loan_df is not None and not loan_df.empty:
            loan_features = self.create_loan_history_features(loan_df)
        else:
            loan_features = pd.DataFrame({user_id_col: transaction_df[user_id_col].unique()})

        # Category 5: Financial Buffer
        if "balance" in transaction_df.columns:
            buffer_features = self.create_financial_buffer_features(transaction_df)
        else:
            buffer_features = pd.DataFrame({user_id_col: transaction_df[user_id_col].unique()})

        # Category 6: Transaction Diversity
        diversity_features = self.create_transaction_diversity_features(transaction_df)

        # Category 7: Temporal Consistency
        temporal_features = self.create_temporal_consistency_features(transaction_df)

        # Rolling Window Features
        rolling_features = self.create_rolling_window_features(transaction_df)

        # Merge all features
        all_features = income_features
        for feat_df in [expenditure_features, airtime_features, loan_features,
                        buffer_features, diversity_features, temporal_features, rolling_features]:
            all_features = all_features.merge(feat_df, on=user_id_col, how="outer")

        self.logger.info(
            "all_features_created",
            total_features=len(all_features.columns),
            users=len(all_features),
        )

        return all_features
