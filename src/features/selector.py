"""Feature selection module for the ML Loan Eligibility Platform."""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    mutual_info_classif,
    RFE,
    RFECV,
    SelectKBest,
    chi2,
    f_classif,
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.logging import LoggerMixin


class FeatureSelector(LoggerMixin):
    """Feature selector to reduce from 120+ features to ~40-50 most predictive."""

    def __init__(self, target_features: int = 45):
        """Initialize feature selector.

        Args:
            target_features: Target number of features to select
        """
        self.target_features = target_features
        self.selected_features: List[str] = []
        self.feature_importances: pd.DataFrame = pd.DataFrame()
        self.logger.info("feature_selector_initialized", target_features=target_features)

    def calculate_correlation_matrix(
        self,
        X: pd.DataFrame,
        threshold: float = 0.95,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Calculate correlation matrix and identify highly correlated features.

        Args:
            X: Feature DataFrame
            threshold: Correlation threshold for identifying multicollinearity

        Returns:
            Tuple of (correlation matrix, list of features to remove)
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        # Identify highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]

        self.logger.info(
            "correlation_analysis_complete",
            highly_correlated_features=len(to_drop),
            threshold=threshold,
        )

        return corr_matrix, to_drop

    def calculate_mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Calculate mutual information scores for features.

        Args:
            X: Feature DataFrame
            y: Target variable

        Returns:
            DataFrame with mutual information scores
        """
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)

        # Create DataFrame with scores
        mi_df = pd.DataFrame({
            "feature": X.columns,
            "mi_score": mi_scores,
        }).sort_values("mi_score", ascending=False)

        self.logger.info(
            "mutual_information_calculated",
            features=len(mi_df),
        )

        return mi_df

    def perform_rfecv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> Tuple[RFECV, List[str]]:
        """Perform Recursive Feature Elimination with Cross-Validation.

        Args:
            X: Feature DataFrame
            y: Target variable
            cv: Number of cross-validation folds

        Returns:
            Tuple of (RFECV object, list of selected features)
        """
        # Use Random Forest as the estimator
        estimator = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )

        # Perform RFECV
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

        rfecv.fit(X, y)

        # Get selected features
        selected_features = X.columns[rfecv.support_].tolist()

        self.logger.info(
            "rfecv_complete",
            optimal_features=rfecv.n_features_,
            selected_features=len(selected_features),
        )

        return rfecv, selected_features

    def calculate_random_forest_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_estimators: int = 200,
    ) -> pd.DataFrame:
        """Calculate feature importance using Random Forest.

        Args:
            X: Feature DataFrame
            y: Target variable
            n_estimators: Number of trees in the forest

        Returns:
            DataFrame with feature importances
        """
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X, y)

        # Get feature importances
        importances = pd.DataFrame({
            "feature": X.columns,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False)

        self.logger.info(
            "random_forest_importance_calculated",
            features=len(importances),
        )

        return importances

    def rank_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> pd.DataFrame:
        """Rank features using multiple methods.

        Args:
            X: Feature DataFrame
            y: Target variable

        Returns:
            DataFrame with combined feature rankings
        """
        self.logger.info("ranking_features_with_multiple_methods")

        # Method 1: Mutual Information
        mi_scores = self.calculate_mutual_information(X, y)
        mi_scores["mi_rank"] = range(1, len(mi_scores) + 1)

        # Method 2: Random Forest Importance
        rf_scores = self.calculate_random_forest_importance(X, y)
        rf_scores["rf_rank"] = range(1, len(rf_scores) + 1)

        # Merge rankings
        rankings = mi_scores[["feature", "mi_score", "mi_rank"]].merge(
            rf_scores[["feature", "importance", "rf_rank"]],
            on="feature",
            how="outer",
        )

        # Calculate average rank
        rankings["avg_rank"] = (rankings["mi_rank"] + rankings["rf_rank"]) / 2
        rankings = rankings.sort_values("avg_rank")

        self.feature_importances = rankings

        self.logger.info(
            "feature_ranking_complete",
            features=len(rankings),
        )

        return rankings

    def select_top_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        remove_correlated: bool = True,
        correlation_threshold: float = 0.95,
    ) -> List[str]:
        """Select top features using combined approach.

        Args:
            X: Feature DataFrame
            y: Target variable
            remove_correlated: Whether to remove highly correlated features
            correlation_threshold: Threshold for removing correlated features

        Returns:
            List of selected feature names
        """
        self.logger.info("selecting_top_features")

        # Step 1: Remove highly correlated features
        if remove_correlated:
            corr_matrix, to_drop = self.calculate_correlation_matrix(X, correlation_threshold)
            X_reduced = X.drop(columns=to_drop)
            self.logger.info(
                "removed_correlated_features",
                removed=len(to_drop),
                remaining=len(X_reduced.columns),
            )
        else:
            X_reduced = X.copy()

        # Step 2: Rank features using multiple methods
        rankings = self.rank_features(X_reduced, y)

        # Step 3: Select top N features
        selected_features = rankings.head(self.target_features)["feature"].tolist()

        self.selected_features = selected_features

        self.logger.info(
            "top_features_selected",
            selected=len(selected_features),
        )

        return selected_features

    def plot_feature_importance(
        self,
        top_n: int = 30,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot feature importance.

        Args:
            top_n: Number of top features to plot
            figsize: Figure size
            save_path: Path to save the plot (if None, displays instead)
        """
        if self.feature_importances.empty:
            self.logger.warning("no_feature_importances_available")
            return

        # Get top N features
        top_features = self.feature_importances.head(top_n)

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Mutual Information
        axes[0].barh(top_features["feature"], top_features["mi_score"])
        axes[0].set_xlabel("Mutual Information Score")
        axes[0].set_title(f"Top {top_n} Features by Mutual Information")
        axes[0].invert_yaxis()

        # Plot 2: Random Forest Importance
        axes[1].barh(top_features["feature"], top_features["importance"])
        axes[1].set_xlabel("Random Forest Importance")
        axes[1].set_title(f"Top {top_n} Features by Random Forest")
        axes[1].invert_yaxis()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info("feature_importance_plot_saved", path=save_path)
        else:
            plt.show()

        plt.close()

    def plot_correlation_heatmap(
        self,
        X: pd.DataFrame,
        features: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 12),
        save_path: Optional[str] = None,
    ) -> None:
        """Plot correlation heatmap for selected features.

        Args:
            X: Feature DataFrame
            features: List of features to include (if None, uses selected features)
            figsize: Figure size
            save_path: Path to save the plot (if None, displays instead)
        """
        features = features or self.selected_features

        if not features:
            self.logger.warning("no_features_specified_for_heatmap")
            return

        # Filter to selected features
        X_subset = X[features]

        # Calculate correlation matrix
        corr_matrix = X_subset.corr()

        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
        )
        plt.title("Correlation Heatmap of Selected Features")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info("correlation_heatmap_saved", path=save_path)
        else:
            plt.show()

        plt.close()

    def generate_selection_report(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate feature selection report.

        Args:
            output_path: Path to save the report (if None, returns as string)

        Returns:
            Feature selection report
        """
        if self.feature_importances.empty:
            return "No feature selection has been performed yet."

        report_lines = [
            "Feature Selection Report",
            "=" * 60,
            f"Total Features Analyzed: {len(self.feature_importances)}",
            f"Target Features: {self.target_features}",
            f"Selected Features: {len(self.selected_features)}",
            "",
            "Top 20 Features by Average Rank:",
            "-" * 60,
        ]

        # Add top 20 features
        top_20 = self.feature_importances.head(20)
        for idx, row in top_20.iterrows():
            report_lines.append(
                f"{int(row['avg_rank']):3d}. {row['feature']:40s} "
                f"(MI: {row['mi_score']:.4f}, RF: {row['importance']:.4f})"
            )

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            self.logger.info("feature_selection_report_saved", path=output_path)

        return report
