"""Data loading module for the ML Loan Eligibility Platform."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from ..utils.logging import LoggerMixin


class DataLoader(LoggerMixin):
    """Data loader for transaction and loan data."""

    def __init__(self, data_dir: Union[str, Path] = "data/raw"):
        """Initialize data loader.

        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.logger.info("data_loader_initialized", data_dir=str(self.data_dir))

    def load_mobile_money_transactions(
        self,
        file_path: Optional[Union[str, Path]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load mobile money transaction data.

        Args:
            file_path: Path to transaction data file
            start_date: Filter transactions from this date (YYYY-MM-DD)
            end_date: Filter transactions until this date (YYYY-MM-DD)

        Returns:
            DataFrame with mobile money transactions
        """
        if file_path is None:
            file_path = self.data_dir / "mobile_money_transactions.csv"
        else:
            file_path = Path(file_path)

        self.logger.info("loading_mobile_money_transactions", file_path=str(file_path))

        try:
            df = pd.read_csv(file_path)

            # Convert date column to datetime
            if "transaction_date" in df.columns:
                df["transaction_date"] = pd.to_datetime(df["transaction_date"])

            # Filter by date range if specified
            if start_date:
                df = df[df["transaction_date"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["transaction_date"] <= pd.to_datetime(end_date)]

            self.logger.info(
                "mobile_money_transactions_loaded",
                rows=len(df),
                columns=list(df.columns),
            )

            return df

        except FileNotFoundError:
            self.logger.warning("mobile_money_transactions_file_not_found", file_path=str(file_path))
            return pd.DataFrame()

    def load_airtime_purchases(
        self,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Load airtime purchase data.

        Args:
            file_path: Path to airtime purchase data file

        Returns:
            DataFrame with airtime purchases
        """
        if file_path is None:
            file_path = self.data_dir / "airtime_purchases.csv"
        else:
            file_path = Path(file_path)

        self.logger.info("loading_airtime_purchases", file_path=str(file_path))

        try:
            df = pd.read_csv(file_path)

            # Convert date column to datetime
            if "purchase_date" in df.columns:
                df["purchase_date"] = pd.to_datetime(df["purchase_date"])

            self.logger.info(
                "airtime_purchases_loaded",
                rows=len(df),
                columns=list(df.columns),
            )

            return df

        except FileNotFoundError:
            self.logger.warning("airtime_purchases_file_not_found", file_path=str(file_path))
            return pd.DataFrame()

    def load_loan_history(
        self,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Load historical loan performance data.

        Args:
            file_path: Path to loan history data file

        Returns:
            DataFrame with loan history
        """
        if file_path is None:
            file_path = self.data_dir / "loan_history.csv"
        else:
            file_path = Path(file_path)

        self.logger.info("loading_loan_history", file_path=str(file_path))

        try:
            df = pd.read_csv(file_path)

            # Convert date columns to datetime
            date_columns = ["loan_date", "repayment_date", "default_date"]
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

            self.logger.info(
                "loan_history_loaded",
                rows=len(df),
                columns=list(df.columns),
            )

            return df

        except FileNotFoundError:
            self.logger.warning("loan_history_file_not_found", file_path=str(file_path))
            return pd.DataFrame()

    def load_utility_payments(
        self,
        file_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Load utility payment records.

        Args:
            file_path: Path to utility payment data file

        Returns:
            DataFrame with utility payments
        """
        if file_path is None:
            file_path = self.data_dir / "utility_payments.csv"
        else:
            file_path = Path(file_path)

        self.logger.info("loading_utility_payments", file_path=str(file_path))

        try:
            df = pd.read_csv(file_path)

            # Convert date column to datetime
            if "payment_date" in df.columns:
                df["payment_date"] = pd.to_datetime(df["payment_date"])

            self.logger.info(
                "utility_payments_loaded",
                rows=len(df),
                columns=list(df.columns),
            )

            return df

        except FileNotFoundError:
            self.logger.warning("utility_payments_file_not_found", file_path=str(file_path))
            return pd.DataFrame()

    def load_paysim_data(
        self,
        file_path: Optional[Union[str, Path]] = None,
        nrows: Optional[int] = None,
        sample_frac: Optional[float] = None,
    ) -> pd.DataFrame:
        """Load PaySim mobile money transaction data.

        Args:
            file_path: Path to PaySim CSV file
            nrows: Number of rows to load (for testing)
            sample_frac: Fraction of data to sample

        Returns:
            DataFrame with PaySim transactions
        """
        if file_path is None:
            file_path = self.data_dir / "PS_20174392719_1491204439457_log.csv"
        else:
            file_path = Path(file_path)

        self.logger.info("loading_paysim_data", file_path=str(file_path))

        try:
            df = pd.read_csv(file_path, nrows=nrows)

            if sample_frac and sample_frac < 1.0:
                df = df.sample(frac=sample_frac, random_state=42)
                self.logger.info("paysim_data_sampled", frac=sample_frac, rows=len(df))

            self.logger.info(
                "paysim_data_loaded",
                rows=len(df),
                columns=list(df.columns),
            )

            return df

        except FileNotFoundError:
            self.logger.warning("paysim_file_not_found", file_path=str(file_path))
            return pd.DataFrame()

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available data sources.

        Returns:
            Dictionary mapping data source names to DataFrames
        """
        self.logger.info("loading_all_data_sources")

        data = {
            "mobile_money": self.load_mobile_money_transactions(),
            "airtime": self.load_airtime_purchases(),
            "loans": self.load_loan_history(),
            "utilities": self.load_utility_payments(),
        }

        # Try loading PaySim data if available
        paysim_df = self.load_paysim_data()
        if not paysim_df.empty:
            data["paysim"] = paysim_df

        total_rows = sum(len(df) for df in data.values())
        self.logger.info(
            "all_data_sources_loaded",
            total_rows=total_rows,
            sources={k: len(v) for k, v in data.items()},
        )

        return data

    def save_processed_data(
        self,
        df: pd.DataFrame,
        file_name: str,
        output_dir: Union[str, Path] = "data/processed",
    ) -> None:
        """Save processed data to CSV file.

        Args:
            df: DataFrame to save
            file_name: Output file name
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / file_name
        df.to_csv(output_path, index=False)

        self.logger.info(
            "processed_data_saved",
            file_path=str(output_path),
            rows=len(df),
            columns=len(df.columns),
        )
