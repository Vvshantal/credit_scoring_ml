"""Data validation module for the ML Loan Eligibility Platform."""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from ..utils.logging import LoggerMixin


class DataValidator(LoggerMixin):
    """Validator for data quality checks."""

    def __init__(self):
        """Initialize data validator."""
        self.logger.info("data_validator_initialized")

    def validate_schema(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
        optional_columns: Optional[List[str]] = None,
    ) -> Tuple[bool, List[str]]:
        """Validate DataFrame schema.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            optional_columns: List of optional column names

        Returns:
            Tuple of (is_valid, list of missing columns)
        """
        missing_columns = [col for col in required_columns if col not in df.columns]

        is_valid = len(missing_columns) == 0

        if not is_valid:
            self.logger.warning(
                "schema_validation_failed",
                missing_columns=missing_columns,
            )
        else:
            self.logger.info("schema_validation_passed")

        return is_valid, missing_columns

    def check_missing_values(
        self,
        df: pd.DataFrame,
        threshold: float = 0.3,
    ) -> Dict[str, float]:
        """Check for missing values in DataFrame.

        Args:
            df: DataFrame to check
            threshold: Threshold for warning about high missing rates

        Returns:
            Dictionary mapping column names to missing value percentages
        """
        missing_stats = {}

        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df)
            missing_stats[col] = missing_pct

            if missing_pct > threshold:
                self.logger.warning(
                    "high_missing_values",
                    column=col,
                    missing_percentage=missing_pct,
                )

        total_missing = df.isna().sum().sum()
        self.logger.info(
            "missing_values_checked",
            total_missing=total_missing,
            high_missing_columns=sum(1 for v in missing_stats.values() if v > threshold),
        )

        return missing_stats

    def check_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
    ) -> int:
        """Check for duplicate rows.

        Args:
            df: DataFrame to check
            subset: Columns to check for duplicates (if None, checks all columns)

        Returns:
            Number of duplicate rows
        """
        n_duplicates = df.duplicated(subset=subset).sum()

        if n_duplicates > 0:
            self.logger.warning(
                "duplicates_found",
                count=n_duplicates,
                percentage=n_duplicates / len(df),
            )
        else:
            self.logger.info("no_duplicates_found")

        return n_duplicates

    def check_data_types(
        self,
        df: pd.DataFrame,
        expected_types: Dict[str, str],
    ) -> Dict[str, bool]:
        """Check if columns have expected data types.

        Args:
            df: DataFrame to check
            expected_types: Dictionary mapping column names to expected types

        Returns:
            Dictionary mapping column names to validation results
        """
        type_checks = {}

        for col, expected_type in expected_types.items():
            if col not in df.columns:
                type_checks[col] = False
                self.logger.warning("column_not_found", column=col)
                continue

            actual_type = str(df[col].dtype)

            # Check type compatibility
            is_valid = self._is_type_compatible(actual_type, expected_type)
            type_checks[col] = is_valid

            if not is_valid:
                self.logger.warning(
                    "type_mismatch",
                    column=col,
                    expected=expected_type,
                    actual=actual_type,
                )

        return type_checks

    def _is_type_compatible(self, actual: str, expected: str) -> bool:
        """Check if actual type is compatible with expected type.

        Args:
            actual: Actual data type
            expected: Expected data type

        Returns:
            True if types are compatible
        """
        type_mappings = {
            "int": ["int64", "int32", "int16", "int8"],
            "float": ["float64", "float32", "float16"],
            "string": ["object", "string"],
            "datetime": ["datetime64[ns]", "datetime64"],
            "bool": ["bool"],
        }

        for key, valid_types in type_mappings.items():
            if expected == key and actual in valid_types:
                return True

        return actual == expected

    def check_value_ranges(
        self,
        df: pd.DataFrame,
        ranges: Dict[str, Tuple[float, float]],
    ) -> Dict[str, int]:
        """Check if numeric values fall within expected ranges.

        Args:
            df: DataFrame to check
            ranges: Dictionary mapping column names to (min, max) tuples

        Returns:
            Dictionary mapping column names to count of out-of-range values
        """
        out_of_range = {}

        for col, (min_val, max_val) in ranges.items():
            if col not in df.columns:
                self.logger.warning("column_not_found", column=col)
                continue

            # Count values outside range
            count = ((df[col] < min_val) | (df[col] > max_val)).sum()
            out_of_range[col] = count

            if count > 0:
                self.logger.warning(
                    "out_of_range_values",
                    column=col,
                    count=count,
                    min=min_val,
                    max=max_val,
                )

        return out_of_range

    def check_class_imbalance(
        self,
        df: pd.DataFrame,
        target_column: str,
        threshold: float = 0.2,
    ) -> Dict[str, float]:
        """Check for class imbalance in target variable.

        Args:
            df: DataFrame to check
            target_column: Name of target column
            threshold: Threshold for minority class percentage

        Returns:
            Dictionary with class distribution
        """
        if target_column not in df.columns:
            self.logger.error("target_column_not_found", column=target_column)
            return {}

        class_dist = df[target_column].value_counts(normalize=True).to_dict()

        min_class_pct = min(class_dist.values())

        if min_class_pct < threshold:
            self.logger.warning(
                "class_imbalance_detected",
                distribution=class_dist,
                min_class_percentage=min_class_pct,
            )
        else:
            self.logger.info(
                "class_balance_acceptable",
                distribution=class_dist,
            )

        return class_dist

    def validate_transaction_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """Validate mobile money transaction data.

        Args:
            df: DataFrame with transaction data

        Returns:
            Validation report
        """
        self.logger.info("validating_transaction_data", rows=len(df))

        report = {}

        # Check schema
        required_cols = ["transaction_id", "user_id", "transaction_date", "amount", "type"]
        is_valid, missing = self.validate_schema(df, required_cols)
        report["schema_valid"] = is_valid
        report["missing_columns"] = missing

        # Check missing values
        report["missing_values"] = self.check_missing_values(df)

        # Check duplicates
        report["duplicates"] = self.check_duplicates(df, subset=["transaction_id"])

        # Check data types
        expected_types = {
            "transaction_id": "string",
            "user_id": "string",
            "amount": "float",
        }
        report["type_checks"] = self.check_data_types(df, expected_types)

        # Check value ranges
        if "amount" in df.columns:
            ranges = {"amount": (0, 1000000)}  # Reasonable transaction limits
            report["range_checks"] = self.check_value_ranges(df, ranges)

        return report

    def generate_quality_report(self, df: pd.DataFrame, data_type: str) -> str:
        """Generate a comprehensive data quality report.

        Args:
            df: DataFrame to analyze
            data_type: Type of data (e.g., 'transactions', 'loans')

        Returns:
            Quality report as string
        """
        report_lines = [
            f"Data Quality Report for {data_type}",
            "=" * 50,
            f"Total Rows: {len(df)}",
            f"Total Columns: {len(df.columns)}",
            "",
            "Column Summary:",
        ]

        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
            unique_vals = df[col].nunique()

            report_lines.append(
                f"  {col}: {df[col].dtype}, "
                f"{missing_pct:.2f}% missing, "
                f"{unique_vals} unique values"
            )

        duplicates = df.duplicated().sum()
        report_lines.append("")
        report_lines.append(f"Duplicate Rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")

        return "\n".join(report_lines)
