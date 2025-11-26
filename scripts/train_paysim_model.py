#!/usr/bin/env python3
"""Training script for credit risk model using PaySim data.

This script:
1. Loads PaySim mobile money transaction data
2. Engineers features per customer
3. Trains ML models to predict risky financial behavior
4. Evaluates model performance
5. Saves the best model
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
from imblearn.over_sampling import SMOTE

# Try importing optional packages
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, Exception) as e:
    HAS_XGBOOST = False
    print(f"Warning: XGBoost not available ({e}). Using RandomForest instead.")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, Exception) as e:
    HAS_LIGHTGBM = False
    print(f"Warning: LightGBM not available ({e}).")

from src.features.paysim_engineer import PaySimFeatureEngineer


def load_and_prepare_data(data_path: str, sample_size: int = None) -> pd.DataFrame:
    """Load PaySim data.

    Args:
        data_path: Path to PaySim CSV
        sample_size: Optional number of rows to load

    Returns:
        DataFrame with transactions
    """
    print(f"Loading data from {data_path}...")

    if sample_size:
        df = pd.read_csv(data_path, nrows=sample_size)
    else:
        df = pd.read_csv(data_path)

    print(f"Loaded {len(df):,} transactions")
    return df


def engineer_features(df: pd.DataFrame, remove_leaky_features: bool = True) -> tuple:
    """Engineer features from transaction data.

    Args:
        df: Transaction DataFrame
        remove_leaky_features: Remove features that leak the target

    Returns:
        Tuple of (features_df, target_series)
    """
    print("\nEngineering features...")
    engineer = PaySimFeatureEngineer()
    features, target = engineer.create_all_features(df)

    # Remove ID column and target from features
    feature_cols = [c for c in features.columns if c not in ['nameOrig', 'is_risky']]

    # Remove leaky features (directly derived from isFraud)
    if remove_leaky_features:
        leaky_features = [
            'total_fraud_involvement', 'fraud_rate', 'flagged_fraud_count',
            'rolling_24h_fraud_count', 'rolling_168h_fraud_count', 'rolling_336h_fraud_count'
        ]
        feature_cols = [c for c in feature_cols if c not in leaky_features]
        print(f"Removed {len(leaky_features)} leaky features")

    X = features[feature_cols]
    y = target

    print(f"Created {len(feature_cols)} features for {len(X):,} customers")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    return X, y, feature_cols


def train_models(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate multiple models.

    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        feature_names: List of feature names

    Returns:
        Dictionary of trained models with metrics
    """
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)

    # Handle class imbalance with SMOTE
    print("\nApplying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_resampled):,} samples")

    results = {}

    # 1. Logistic Regression
    print("\n--- Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_model.fit(X_train_resampled, y_train_resampled)
    results['logistic_regression'] = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

    # 2. Random Forest
    print("\n--- Random Forest ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train_resampled, y_train_resampled)
    results['random_forest'] = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Feature importance from Random Forest
    print("\nTop 15 Most Important Features (Random Forest):")
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances.head(15).to_string(index=False))

    # 3. XGBoost (if available)
    if HAS_XGBOOST:
        print("\n--- XGBoost ---")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)  # XGBoost handles imbalance internally
        results['xgboost'] = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # 4. LightGBM (if available)
    if HAS_LIGHTGBM:
        print("\n--- LightGBM ---")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        results['lightgbm'] = evaluate_model(lgb_model, X_test, y_test, "LightGBM")

    return results


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a trained model.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name: Name for display

    Returns:
        Dictionary with model and metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
    }

    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {'model': model, 'metrics': metrics}


def save_best_model(results, output_dir, scaler, feature_names):
    """Save the best performing model.

    Args:
        results: Dictionary of model results
        output_dir: Output directory
        scaler: Fitted scaler
        feature_names: Feature names
    """
    # Find best model by F1 score
    best_model_name = max(results, key=lambda k: results[k]['metrics']['f1'])
    best_model = results[best_model_name]['model']
    best_metrics = results[best_model_name]['metrics']

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")
    print(f"{'='*60}")

    # Save model
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "paysim_credit_model.joblib"
    scaler_path = output_dir / "paysim_scaler.joblib"
    features_path = output_dir / "paysim_feature_names.joblib"

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_names, features_path)

    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Features saved to: {features_path}")

    return best_model_name, best_model


def main():
    """Main training pipeline."""
    # Configuration
    DATA_PATH = "data/raw/PS_20174392719_1491204439457_log.csv"
    OUTPUT_DIR = "models_trained"
    SAMPLE_SIZE = 1_000_000  # Use 1M rows for faster training (set to None for full data)

    print("="*60)
    print("PAYSIM CREDIT RISK MODEL TRAINING")
    print("="*60)

    # Step 1: Load data
    df = load_and_prepare_data(DATA_PATH, sample_size=SAMPLE_SIZE)

    # Step 2: Engineer features
    X, y, feature_names = engineer_features(df)

    # Step 3: Split data
    print("\nSplitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Step 4: Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Train models
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test, feature_names)

    # Step 6: Save best model
    save_best_model(results, OUTPUT_DIR, scaler, feature_names)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
