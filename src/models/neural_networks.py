"""Neural network models for the ML Loan Eligibility Platform."""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
from ..utils.logging import LoggerMixin


class FeedForwardNetwork(LoggerMixin):
    """Feedforward neural network for loan eligibility prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
    ):
        """Initialize feedforward neural network.

        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        self.logger.info(
            "feedforward_network_initialized",
            input_dim=input_dim,
            hidden_layers=hidden_layers,
        )

    def build_model(self) -> keras.Model:
        """Build the neural network model.

        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs

        # Hidden layers with batch normalization and dropout
        for units in self.hidden_layers:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Output layer
        outputs = layers.Dense(1, activation="sigmoid")(x)

        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.AUC(name="auc"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
            ],
        )

        self.model = model

        self.logger.info("feedforward_model_built", total_params=model.count_params())

        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ) -> keras.callbacks.History:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Maximum number of epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor="val_auc",
                patience=early_stopping_patience,
                restore_best_weights=True,
                mode="max",
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ]

        # Train model
        self.logger.info(
            "training_feedforward_network",
            epochs=epochs,
            batch_size=batch_size,
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=0,
        )

        self.history = history

        self.logger.info("feedforward_network_trained")

        return history

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features

        Returns:
            Predicted probabilities
        """
        return self.model.predict(X, verbose=0).flatten()

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """Evaluate model performance.

        Args:
            X: Features
            y: True labels

        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred_proba = self.predict(X)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calculate metrics
        metrics = {
            "roc_auc": roc_auc_score(y, y_pred_proba),
            "pr_auc": average_precision_score(y, y_pred_proba),
            "accuracy": (y_pred == y).mean(),
        }

        self.logger.info("model_evaluated", metrics=metrics)

        return metrics

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        self.model.save(path)
        self.logger.info("model_saved", path=path)

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to saved model
        """
        self.model = keras.models.load_model(path)
        self.logger.info("model_loaded", path=path)


class LSTMNetwork(LoggerMixin):
    """LSTM neural network for sequential transaction data."""

    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 30,
        lstm_units: List[int] = [64, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
    ):
        """Initialize LSTM network.

        Args:
            input_dim: Number of features per timestep
            sequence_length: Length of input sequences
            lstm_units: List of LSTM layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        self.logger.info(
            "lstm_network_initialized",
            input_dim=input_dim,
            sequence_length=sequence_length,
            lstm_units=lstm_units,
        )

    def build_model(self) -> keras.Model:
        """Build the LSTM model.

        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.input_dim))
        x = inputs

        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = layers.LSTM(units, return_sequences=return_sequences)(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Dense layers
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(self.dropout_rate)(x)

        # Output layer
        outputs = layers.Dense(1, activation="sigmoid")(x)

        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.AUC(name="auc"),
            ],
        )

        self.model = model

        self.logger.info("lstm_model_built", total_params=model.count_params())

        return model

    def prepare_sequences(
        self,
        transactions_df: pd.DataFrame,
        user_id_col: str = "user_id",
        date_col: str = "transaction_date",
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequential data for LSTM.

        Args:
            transactions_df: Transaction DataFrame
            user_id_col: User ID column
            date_col: Date column
            feature_cols: List of feature columns (if None, uses all numeric)

        Returns:
            Tuple of (sequences array, user IDs)
        """
        # Sort by user and date
        df = transactions_df.sort_values([user_id_col, date_col])

        # Select feature columns
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in feature_cols if col != user_id_col]

        # Create sequences for each user
        sequences = []
        user_ids = []

        for user_id in df[user_id_col].unique():
            user_data = df[df[user_id_col] == user_id][feature_cols].values

            if len(user_data) >= self.sequence_length:
                # Take the most recent sequence
                seq = user_data[-self.sequence_length:]
                sequences.append(seq)
                user_ids.append(user_id)
            elif len(user_data) > 0:
                # Pad sequence if too short
                pad_length = self.sequence_length - len(user_data)
                seq = np.vstack([
                    np.zeros((pad_length, len(feature_cols))),
                    user_data
                ])
                sequences.append(seq)
                user_ids.append(user_id)

        sequences = np.array(sequences)
        user_ids = np.array(user_ids)

        self.logger.info(
            "sequences_prepared",
            num_sequences=len(sequences),
            sequence_shape=sequences.shape,
        )

        return sequences, user_ids

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ) -> keras.callbacks.History:
        """Train the LSTM model.

        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Maximum number of epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor="val_auc",
                patience=early_stopping_patience,
                restore_best_weights=True,
                mode="max",
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ]

        # Train model
        self.logger.info(
            "training_lstm_network",
            epochs=epochs,
            batch_size=batch_size,
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=0,
        )

        self.history = history

        self.logger.info("lstm_network_trained")

        return history

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Make predictions.

        Args:
            X: Sequences

        Returns:
            Predicted probabilities
        """
        return self.model.predict(X, verbose=0).flatten()

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict:
        """Evaluate model performance.

        Args:
            X: Sequences
            y: True labels

        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred_proba = self.predict(X)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Calculate metrics
        metrics = {
            "roc_auc": roc_auc_score(y, y_pred_proba),
            "pr_auc": average_precision_score(y, y_pred_proba),
            "accuracy": (y_pred == y).mean(),
        }

        self.logger.info("lstm_model_evaluated", metrics=metrics)

        return metrics

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        self.model.save(path)
        self.logger.info("lstm_model_saved", path=path)

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to saved model
        """
        self.model = keras.models.load_model(path)
        self.logger.info("lstm_model_loaded", path=path)
