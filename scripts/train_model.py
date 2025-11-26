"""Script to train the loan eligibility model."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.engineer import FeatureEngineer
from src.features.selector import FeatureSelector
from src.models.train import ModelTrainer
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Main training pipeline."""
    logger.info("starting_model_training_pipeline")

    # 1. Load data
    logger.info("loading_data")
    loader = DataLoader("data/raw")
    data = loader.load_all_data()

    transactions = data["mobile_money"]
    airtime = data["airtime"]
    loans = data["loans"]

    # Load target variable
    target_df = pd.read_csv("data/raw/loan_eligibility.csv")

    logger.info(
        "data_loaded",
        transactions=len(transactions),
        airtime=len(airtime),
        loans=len(loans),
    )

    # 2. Engineer features
    logger.info("engineering_features")
    engineer = FeatureEngineer(rolling_windows=[7, 30, 90])
    features = engineer.create_all_features(
        transaction_df=transactions,
        airtime_df=airtime,
        loan_df=loans,
    )

    logger.info("features_created", total_features=len(features.columns))

    # 3. Merge with target
    data_merged = features.merge(target_df, on="user_id", how="inner")
    logger.info("data_merged", total_samples=len(data_merged))

    # 4. Prepare X and y
    X = data_merged.drop(columns=["user_id", "is_eligible"])
    y = data_merged["is_eligible"]

    # Handle any remaining missing values
    preprocessor = DataPreprocessor()
    X = preprocessor.handle_missing_values(X, strategy="auto")

    logger.info("features_prepared", shape=X.shape)

    # 5. Feature selection
    logger.info("selecting_features")
    selector = FeatureSelector(target_features=45)
    selected_features = selector.select_top_features(
        X, y, remove_correlated=True, correlation_threshold=0.95
    )

    X_selected = X[selected_features]
    logger.info("features_selected", selected=len(selected_features))

    # Save feature importance plot
    selector.plot_feature_importance(
        top_n=30,
        save_path="models_trained/feature_importance.png"
    )

    # 6. Train models
    logger.info("training_models")
    trainer = ModelTrainer(test_size=0.2, val_size=0.2, random_state=42)

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        X_selected, y, apply_smote=True
    )

    # Train all models
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)

    # Print results
    print("\n" + "="*80)
    print(trainer.generate_comparison_report())
    print("="*80 + "\n")

    # 7. Evaluate on test set
    logger.info("evaluating_on_test_set")
    test_metrics = trainer._evaluate_model(trainer.best_model, X_test, y_test)

    print(f"Test Set Performance ({trainer.best_model_name}):")
    print("-" * 40)
    for metric, value in test_metrics.items():
        print(f"{metric:>15}: {value:.4f}")
    print("-" * 40 + "\n")

    # 8. Save best model
    logger.info("saving_model", model=trainer.best_model_name)
    Path("models_trained").mkdir(exist_ok=True)

    trainer.save_model(
        trainer.best_model_name,
        f"models_trained/{trainer.best_model_name}.joblib"
    )

    trainer.save_model(
        trainer.best_model_name,
        "models_trained/best_model.joblib"
    )

    # Save feature names
    import joblib
    joblib.dump(selected_features, "models_trained/feature_names.joblib")

    # Save selector for future use
    joblib.dump(selector, "models_trained/feature_selector.joblib")

    logger.info(
        "training_complete",
        best_model=trainer.best_model_name,
        test_accuracy=test_metrics["accuracy"],
        test_auc=test_metrics["roc_auc"],
    )

    print("✓ Model training completed successfully!")
    print(f"✓ Best model: {trainer.best_model_name}")
    print(f"✓ Test Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"✓ Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"✓ Models saved to models_trained/")


if __name__ == "__main__":
    main()
