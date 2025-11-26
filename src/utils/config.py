"""Configuration management for the ML Loan Eligibility Platform."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    name: str = "loan_eligibility_db"
    user: str = "postgres"
    password: str = Field(default="", env="DB_PASSWORD")
    pool_size: int = 20
    max_overflow: int = 10

    @property
    def url(self) -> str:
        """Get database URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisConfig(BaseSettings):
    """Redis configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = Field(default="", env="REDIS_PASSWORD")
    cache_ttl: int = 3600


class APIConfig(BaseSettings):
    """API configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    jwt_secret: str = Field(default="", env="JWT_SECRET")
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 86400


class ModelConfig(BaseSettings):
    """Model configuration."""

    version: str = "1.0.0"
    model_path: str = "models_trained/"
    feature_threshold: int = 40
    prediction_threshold: float = 0.5
    retrain_schedule: str = "monthly"


class FeatureConfig(BaseSettings):
    """Feature engineering configuration."""

    rolling_windows: List[int] = [7, 30, 90]
    aggregation_periods: List[str] = ["daily", "weekly", "monthly", "quarterly"]
    min_transaction_history_days: int = 180


class DecisionEngineConfig(BaseSettings):
    """Decision engine configuration."""

    auto_approve_threshold: float = 0.85
    auto_reject_threshold: float = 0.35
    manual_review_range: List[float] = [0.35, 0.85]
    max_loan_amount: float = 50000
    min_loan_amount: float = 1000


class MonitoringConfig(BaseSettings):
    """Monitoring configuration."""

    prometheus_port: int = 9090
    metrics_interval: int = 60
    alert_threshold_accuracy: float = 0.85
    alert_threshold_processing_time: int = 180


class FairnessConfig(BaseSettings):
    """Fairness configuration."""

    enable_fairness_checks: bool = True
    disparate_impact_threshold: float = 0.8
    enable_bias_mitigation: bool = True


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"
    output: str = "stdout"
    file_path: str = "logs/application.log"


class ProcessingConfig(BaseSettings):
    """Processing configuration."""

    max_concurrent_applications: int = 100
    timeout_seconds: int = 180
    retry_attempts: int = 3


class SecurityConfig(BaseSettings):
    """Security configuration."""

    enable_encryption_at_rest: bool = True
    enable_mfa: bool = True
    password_min_length: int = 12
    session_timeout: int = 3600


class Config:
    """Main configuration class."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        self._config_data = self._load_config(config_path)

        # Initialize all configuration sections
        self.database = DatabaseConfig(**self._config_data.get("database", {}))
        self.redis = RedisConfig(**self._config_data.get("redis", {}))
        self.api = APIConfig(**self._config_data.get("api", {}))
        self.model = ModelConfig(**self._config_data.get("model", {}))
        self.features = FeatureConfig(**self._config_data.get("features", {}))
        self.decision_engine = DecisionEngineConfig(**self._config_data.get("decision_engine", {}))
        self.monitoring = MonitoringConfig(**self._config_data.get("monitoring", {}))
        self.fairness = FairnessConfig(**self._config_data.get("fairness", {}))
        self.logging = LoggingConfig(**self._config_data.get("logging", {}))
        self.processing = ProcessingConfig(**self._config_data.get("processing", {}))
        self.security = SecurityConfig(**self._config_data.get("security", {}))

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        if not config_path.exists():
            return {}

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Expand environment variables
        config = self._expand_env_vars(config)

        return config

    def _expand_env_vars(self, config: Any) -> Any:
        """Recursively expand environment variables in configuration.

        Args:
            config: Configuration value (dict, list, or string)

        Returns:
            Configuration with expanded environment variables
        """
        if isinstance(config, dict):
            return {k: self._expand_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._expand_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, "")
        else:
            return config


# Global configuration instance
config = Config()
