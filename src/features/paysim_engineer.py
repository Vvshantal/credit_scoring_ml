"""Feature engineering module adapted for PaySim dataset.

This module creates credit risk features from PaySim mobile money transaction data.
Features are engineered per customer to predict risky financial behavior.

PaySim columns:
- step: time unit (1 step = 1 hour, total 743 steps = ~31 days)
- type: CASH_IN, CASH_OUT, PAYMENT, TRANSFER, DEBIT
- amount: transaction amount
- nameOrig: customer who started transaction
- oldbalanceOrg: balance before transaction (sender)
- newbalanceOrig: balance after transaction (sender)
- nameDest: recipient
- oldbalanceDest: balance before transaction (recipient)
- newbalanceDest: balance after transaction (recipient)
- isFraud: fraud label (used as proxy for high-risk behavior)
- isFlaggedFraud: flagged fraud
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats


class PaySimFeatureEngineer:
    """Feature engineering for PaySim mobile money transaction data."""

    def __init__(self, rolling_windows: List[int] = [24, 168, 336]):
        """Initialize feature engineer.

        Args:
            rolling_windows: Time windows for rolling features (in hours/steps)
                           Default: 24h (1 day), 168h (7 days), 336h (14 days)
        """
        self.rolling_windows = rolling_windows
        self.transaction_types = ['CASH_IN', 'CASH_OUT', 'PAYMENT', 'TRANSFER', 'DEBIT']

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and clean PaySim data.

        Args:
            df: Raw PaySim DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Ensure correct types
        df['step'] = df['step'].astype(int)
        df['amount'] = df['amount'].astype(float)
        df['isFraud'] = df['isFraud'].astype(int)

        # Create derived columns
        df['balance_change'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['is_outgoing'] = df['type'].isin(['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT']).astype(int)
        df['is_incoming'] = df['type'].isin(['CASH_IN']).astype(int)
        df['day'] = df['step'] // 24
        df['hour_of_day'] = df['step'] % 24

        return df

    # ==================== Category 1: Income/Cash-In Stability ====================

    def create_income_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create income (CASH_IN) stability features per customer."""

        # Filter incoming transactions
        income_df = df[df['type'] == 'CASH_IN'].copy()

        if income_df.empty:
            return pd.DataFrame({'nameOrig': df['nameOrig'].unique()})

        features = income_df.groupby('nameOrig').agg({
            'amount': [
                ('income_total', 'sum'),
                ('income_mean', 'mean'),
                ('income_std', 'std'),
                ('income_min', 'min'),
                ('income_max', 'max'),
                ('income_count', 'count'),
            ],
            'step': [
                ('income_first_step', 'min'),
                ('income_last_step', 'max'),
            ]
        })

        features.columns = [col[1] for col in features.columns]
        features = features.reset_index()

        # Derived features
        features['income_cv'] = (features['income_std'] / features['income_mean']).fillna(0)
        features['income_range_ratio'] = (
            (features['income_max'] - features['income_min']) / features['income_mean']
        ).fillna(0)
        features['income_activity_span'] = features['income_last_step'] - features['income_first_step']

        # Drop intermediate columns
        features = features.drop(['income_first_step', 'income_last_step'], axis=1)

        return features

    # ==================== Category 2: Expenditure/Cash-Out Patterns ====================

    def create_expenditure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create expenditure (outgoing) pattern features per customer."""

        # Filter outgoing transactions
        expense_df = df[df['is_outgoing'] == 1].copy()

        if expense_df.empty:
            return pd.DataFrame({'nameOrig': df['nameOrig'].unique()})

        features = expense_df.groupby('nameOrig').agg({
            'amount': [
                ('expense_total', 'sum'),
                ('expense_mean', 'mean'),
                ('expense_std', 'std'),
                ('expense_min', 'min'),
                ('expense_max', 'max'),
                ('expense_count', 'count'),
            ]
        })

        features.columns = [col[1] for col in features.columns]
        features = features.reset_index()

        # Derived features
        features['expense_cv'] = (features['expense_std'] / features['expense_mean']).fillna(0)
        features['expense_range_ratio'] = (
            (features['expense_max'] - features['expense_min']) / features['expense_mean']
        ).fillna(0)

        # Transaction type breakdown
        type_counts = expense_df.groupby(['nameOrig', 'type']).size().unstack(fill_value=0)
        for txn_type in ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT']:
            if txn_type in type_counts.columns:
                type_counts[f'{txn_type.lower()}_count'] = type_counts[txn_type]
            else:
                type_counts[f'{txn_type.lower()}_count'] = 0

        type_counts = type_counts[[c for c in type_counts.columns if '_count' in c]]
        type_counts = type_counts.reset_index()

        features = features.merge(type_counts, on='nameOrig', how='left')

        return features

    # ==================== Category 3: Balance Maintenance ====================

    def create_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial balance maintenance features per customer."""

        features = df.groupby('nameOrig').agg({
            'oldbalanceOrg': [
                ('balance_min', 'min'),
                ('balance_mean', 'mean'),
                ('balance_max', 'max'),
                ('balance_std', 'std'),
            ],
            'newbalanceOrig': [
                ('final_balance_mean', 'mean'),
            ],
            'balance_change': [
                ('avg_balance_change', 'mean'),
                ('balance_change_std', 'std'),
            ]
        })

        features.columns = [col[1] for col in features.columns]
        features = features.reset_index()

        # Derived balance features
        features['balance_cv'] = (features['balance_std'] / features['balance_mean']).fillna(0)
        features['balance_range'] = features['balance_max'] - features['balance_min']

        # Zero balance frequency
        zero_balance = df.groupby('nameOrig', as_index=False).apply(
            lambda x: pd.Series({'zero_balance_freq': (x['oldbalanceOrg'] == 0).sum() / len(x)}),
            include_groups=False
        )
        features = features.merge(zero_balance, on='nameOrig', how='left')

        # Low balance frequency (below 10% of mean)
        low_balance = df.groupby('nameOrig', as_index=False).apply(
            lambda x: pd.Series({
                'low_balance_freq': (x['oldbalanceOrg'] < x['oldbalanceOrg'].mean() * 0.1).sum() / len(x)
                if x['oldbalanceOrg'].mean() > 0 else 0
            }),
            include_groups=False
        )
        features = features.merge(low_balance, on='nameOrig', how='left')

        return features

    # ==================== Category 4: Transaction Diversity ====================

    def create_diversity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transaction diversity features per customer."""

        features = df.groupby('nameOrig').agg({
            'nameDest': [
                ('unique_recipients', 'nunique'),
            ],
            'type': [
                ('unique_txn_types', 'nunique'),
            ],
            'amount': [
                ('total_transactions', 'count'),
            ]
        })

        features.columns = [col[1] for col in features.columns]
        features = features.reset_index()

        # Transaction type diversity (entropy)
        type_entropy = df.groupby('nameOrig', as_index=False).apply(
            lambda x: pd.Series({
                'txn_type_entropy': stats.entropy(x['type'].value_counts(normalize=True))
                if len(x) > 1 else 0
            }),
            include_groups=False
        )
        features = features.merge(type_entropy, on='nameOrig', how='left')

        # New recipient ratio (first-time transactions)
        df_sorted = df.sort_values(['nameOrig', 'step'])
        df_sorted['is_new_recipient'] = ~df_sorted.duplicated(subset=['nameOrig', 'nameDest'])
        new_recipient_ratio = df_sorted.groupby('nameOrig')['is_new_recipient'].mean().to_frame('new_recipient_ratio')
        features = features.merge(new_recipient_ratio, on='nameOrig', how='left')

        return features

    # ==================== Category 5: Temporal Patterns ====================

    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal consistency features per customer."""

        df_sorted = df.sort_values(['nameOrig', 'step'])

        # Transaction intervals
        df_sorted['time_since_last'] = df_sorted.groupby('nameOrig')['step'].diff()

        interval_features = df_sorted.groupby('nameOrig')['time_since_last'].agg([
            ('interval_mean', 'mean'),
            ('interval_std', 'std'),
            ('interval_min', 'min'),
            ('interval_max', 'max'),
        ]).reset_index()

        interval_features['interval_regularity'] = (
            1 / (1 + interval_features['interval_std'] / interval_features['interval_mean'])
        ).fillna(0)

        # Time of day patterns
        hour_features = df.groupby('nameOrig')['hour_of_day'].agg([
            ('most_active_hour', lambda x: x.mode()[0] if not x.mode().empty else 12),
            ('hour_std', 'std'),
        ]).reset_index()

        features = interval_features.merge(hour_features, on='nameOrig', how='left')

        # Day patterns
        day_features = df.groupby('nameOrig')['day'].agg([
            ('active_days', 'nunique'),
            ('first_active_day', 'min'),
            ('last_active_day', 'max'),
        ]).reset_index()

        day_features['activity_span_days'] = day_features['last_active_day'] - day_features['first_active_day'] + 1
        day_features['daily_activity_rate'] = (
            day_features['active_days'] / day_features['activity_span_days']
        ).fillna(0)

        features = features.merge(day_features[['nameOrig', 'active_days', 'daily_activity_rate']], on='nameOrig', how='left')

        return features

    # ==================== Category 6: Rolling Window Features ====================

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features per customer."""

        max_step = df['step'].max()
        features_list = []

        for window in self.rolling_windows:
            # Get transactions in the last `window` hours
            recent_df = df[df['step'] > (max_step - window)]

            if recent_df.empty:
                continue

            window_name = f'{window}h'

            window_features = recent_df.groupby('nameOrig').agg({
                'amount': [
                    (f'rolling_{window_name}_total', 'sum'),
                    (f'rolling_{window_name}_mean', 'mean'),
                    (f'rolling_{window_name}_count', 'count'),
                ],
                'isFraud': [
                    (f'rolling_{window_name}_fraud_count', 'sum'),
                ]
            })

            window_features.columns = [col[1] for col in window_features.columns]
            window_features = window_features.reset_index()

            features_list.append(window_features)

        if not features_list:
            return pd.DataFrame({'nameOrig': df['nameOrig'].unique()})

        # Merge all rolling features
        features = features_list[0]
        for feat_df in features_list[1:]:
            features = features.merge(feat_df, on='nameOrig', how='outer')

        return features

    # ==================== Category 7: Risk Indicators ====================

    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk indicator features per customer."""

        # High-value transaction ratio
        amount_threshold = df['amount'].quantile(0.9)

        risk_features = df.groupby('nameOrig', as_index=False).apply(
            lambda x: pd.Series({
                'high_value_txn_ratio': (x['amount'] >= amount_threshold).sum() / len(x),
                'total_fraud_involvement': x['isFraud'].sum(),
                'fraud_rate': x['isFraud'].mean(),
                'flagged_fraud_count': x['isFlaggedFraud'].sum(),
            }),
            include_groups=False
        )

        # Suspicious patterns: TRANSFER followed by CASH_OUT (common fraud pattern)
        df_sorted = df.sort_values(['nameOrig', 'step'])
        df_sorted['prev_type'] = df_sorted.groupby('nameOrig')['type'].shift(1)
        df_sorted['suspicious_pattern'] = (
            (df_sorted['prev_type'] == 'TRANSFER') & (df_sorted['type'] == 'CASH_OUT')
        ).astype(int)

        suspicious = df_sorted.groupby('nameOrig')['suspicious_pattern'].sum().to_frame('suspicious_pattern_count')
        risk_features = risk_features.merge(suspicious, on='nameOrig', how='left')

        # Balance anomaly: large transaction relative to balance
        df['amount_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
        balance_anomaly = df.groupby('nameOrig', as_index=False).apply(
            lambda x: pd.Series({
                'overdraw_attempt_ratio': (x['amount_balance_ratio'] > 1).sum() / len(x)
            }),
            include_groups=False
        )
        risk_features = risk_features.merge(balance_anomaly, on='nameOrig', how='left')

        return risk_features

    # ==================== Master Feature Engineering ====================

    def create_all_features(
        self,
        df: pd.DataFrame,
        include_target: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Create all features for credit risk prediction.

        Args:
            df: PaySim transaction DataFrame
            include_target: Whether to include the target variable (is_risky)

        Returns:
            Tuple of (features DataFrame, target Series if include_target else None)
        """
        print("Preparing data...")
        df = self.prepare_data(df)

        print("Creating income features...")
        income_features = self.create_income_features(df)

        print("Creating expenditure features...")
        expense_features = self.create_expenditure_features(df)

        print("Creating balance features...")
        balance_features = self.create_balance_features(df)

        print("Creating diversity features...")
        diversity_features = self.create_diversity_features(df)

        print("Creating temporal features...")
        temporal_features = self.create_temporal_features(df)

        print("Creating rolling features...")
        rolling_features = self.create_rolling_features(df)

        print("Creating risk features...")
        risk_features = self.create_risk_features(df)

        # Get all unique customers
        all_customers = pd.DataFrame({'nameOrig': df['nameOrig'].unique()})

        # Merge all features
        print("Merging all features...")
        features = all_customers
        for feat_df in [income_features, expense_features, balance_features,
                        diversity_features, temporal_features, rolling_features, risk_features]:
            features = features.merge(feat_df, on='nameOrig', how='left')

        # Fill NaN values
        features = features.fillna(0)

        # Create target variable (customer involved in any fraud)
        target = None
        if include_target:
            fraud_customers = df[df['isFraud'] == 1]['nameOrig'].unique()
            features['is_risky'] = features['nameOrig'].isin(fraud_customers).astype(int)
            target = features['is_risky']

        print(f"Created {len(features.columns) - 1} features for {len(features)} customers")

        return features, target

    def get_feature_names(self) -> List[str]:
        """Get list of feature names (excluding ID and target)."""
        exclude = ['nameOrig', 'is_risky']
        # This will be populated after create_all_features is called
        return [c for c in self.feature_names if c not in exclude]


def engineer_paysim_features(
    input_path: str,
    output_path: str,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """Convenience function to engineer features from PaySim data.

    Args:
        input_path: Path to PaySim CSV file
        output_path: Path to save engineered features
        sample_size: Optional sample size for testing

    Returns:
        DataFrame with engineered features
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    if sample_size:
        print(f"Sampling {sample_size} transactions...")
        df = df.sample(n=min(sample_size, len(df)), random_state=42)

    engineer = PaySimFeatureEngineer()
    features, target = engineer.create_all_features(df)

    print(f"Saving features to {output_path}...")
    features.to_csv(output_path, index=False)

    return features


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/PS_20174392719_1491204439457_log.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/features/paysim_features.csv"

    features = engineer_paysim_features(input_file, output_file)
    print(f"\nFeature summary:\n{features.describe()}")
