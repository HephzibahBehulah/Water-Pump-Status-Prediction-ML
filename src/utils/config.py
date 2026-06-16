"""
Configuration loader for the project.
Loads YAML configuration files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv
from .logger import setup_logger

logger = setup_logger(__name__)

# Load .env file
load_dotenv()


class Config:
    """Load and access configuration from YAML files and environment variables."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to config YAML file. If None, uses ENVIRONMENT variable
                        to determine which config to load (development/production)
        """
        self.environment = os.getenv('ENVIRONMENT', 'development').lower()

        if config_path is None:
            config_path = f"config/{self.environment}.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_yaml()
        logger.info(f"✅ Configuration loaded from: {self.config_path} ({self.environment})")

    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'models.random_forest.n_estimators')
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> cfg = Config()
            >>> rf_estimators = cfg.get('models.random_forest.n_estimators')
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_dict(self, key: str) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self.get(key, {})

    def get_list(self, key: str) -> list:
        """Get configuration as list."""
        return self.get(key, [])

    def get_int(self, key: str, default: int = 0) -> int:
        """Get configuration as integer."""
        value = self.get(key, default)
        return int(value) if value is not None else default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get configuration as float."""
        value = self.get(key, default)
        return float(value) if value is not None else default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get configuration as boolean."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).lower() in ('true', '1', 'yes')

    def print_config(self):
        """Print entire configuration (for debugging)."""
        print("\n" + "="*50)
        print(f"Configuration ({self.environment})")
        print("="*50)
        print(yaml.dump(self.config, default_flow_style=False))
        print("="*50 + "\n")


# Global config instance
config = None


def get_config(force_reload: bool = False) -> Config:
    """
    Get global config instance. Loads once on first call.

    Args:
        force_reload: Force reload configuration

    Returns:
        Config instance
    """
    global config
    if config is None or force_reload:
        config = Config()
    return config
