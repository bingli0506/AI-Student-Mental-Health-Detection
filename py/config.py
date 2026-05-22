"""
config.py

Global configuration management module for the HybridDLMLMentalHealthDetection
research framework. This module centralizes all configuration parameters used
across the project including dataset paths, preprocessing parameters, model
hyperparameters, training settings, experiment configuration, environment
variables, and visualization options.

Design Philosophy
-----------------
This module follows several academic and engineering best practices:

1. Reproducibility
   All hyperparameters, dataset paths, and training settings are stored in a
   centralized configuration object to ensure that experiments can be easily
   reproduced and audited.

2. Structured Configuration
   Configuration parameters are organized into logical groups:
   - Dataset configuration
   - Preprocessing configuration
   - Model configuration
   - Training configuration
   - Cross-validation configuration
   - Visualization configuration
   - Environment configuration

3. Validation and Safety
   All configuration parameters are validated before use to prevent silent
   misconfiguration errors that could invalidate experimental results.

4. File-Based Configuration
   The configuration system supports loading external configuration files
   (JSON or YAML) to enable experiment-specific overrides.

5. Environment Variable Support
   Sensitive or environment-specific parameters can be defined using system
   environment variables.

6. Extensibility
   The architecture is intentionally modular so that additional configuration
   groups can be added without breaking existing components.

Usage
-----
Typical usage within the project:

    from config import ProjectConfig

    config = ProjectConfig()
    config_dict = config.as_dict()

External configuration files may override default values:

    config = ProjectConfig(config_file="experiment.yaml")

Supported file formats:
    - JSON (.json)
    - YAML (.yaml / .yml)

Author
------
HybridDLMLMentalHealthDetection Research Framework
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

try:
    import yaml
except Exception:
    yaml = None


DATASET_URL: str = "https://www.kaggle.com/datasets/shariful07/student-mental-health"


@dataclass
class DatasetConfig:
    """
    Configuration parameters for dataset loading and preparation.
    """

    dataset_path: str = "data/student_mental_health.csv"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    target_column: str = "mental_health_issue_count"

    categorical_features: list = field(
        default_factory=lambda: [
            "Gender",
            "Marital status",
            "Course",
            "Year of Study",
        ]
    )

    numerical_features: list = field(
        default_factory=lambda: [
            "Age",
            "CGPA",
        ]
    )

    label_range: tuple = (0, 3)


@dataclass
class PreprocessingConfig:
    """
    Configuration for feature preprocessing and engineering.
    """

    normalize_numeric: bool = True
    normalization_method: str = "minmax"

    one_hot_encode: bool = True
    drop_timestamp: bool = True

    create_composite_target: bool = True

    depression_column: str = "Do you have Depression?"
    anxiety_column: str = "Do you have Anxiety?"
    panic_column: str = "Do you have Panic attack?"


@dataclass
class ModelConfig:
    """
    Configuration for base learners and meta learner used in the hybrid
    ensemble architecture.
    """

    rf_estimators: int = 200
    rf_max_depth: Optional[int] = None
    rf_criterion: str = "gini"

    xgb_estimators: int = 200
    xgb_learning_rate: float = 0.1
    xgb_max_depth: int = 6
    xgb_objective: str = "multi:softprob"

    mlp_hidden_layers: tuple = (128, 64)
    mlp_learning_rate: float = 0.001
    mlp_batch_size: int = 32
    mlp_epochs: int = 100
    mlp_activation: str = "relu"
    mlp_optimizer: str = "adam"

    cnn_filters: int = 64
    cnn_kernel_size: int = 3
    cnn_pool_size: int = 2
    cnn_dense_units: int = 64
    cnn_epochs: int = 100
    cnn_batch_size: int = 32

    meta_solver: str = "lbfgs"
    meta_multi_class: str = "multinomial"
    meta_max_iter: int = 200


@dataclass
class TrainingConfig:
    """
    Configuration parameters controlling model training procedures.
    """

    random_seed: int = 42
    batch_size: int = 32
    epochs: int = 100

    learning_rate: float = 0.001

    device: str = "cpu"

    save_models: bool = True
    model_output_dir: str = "artifacts/models"


@dataclass
class CrossValidationConfig:
    """
    Configuration parameters for cross-validation experiments.
    """

    n_folds: int = 5
    stratified: bool = True
    shuffle: bool = True
    random_state: int = 42


@dataclass
class VisualizationConfig:
    """
    Configuration parameters for result visualization and plotting.
    """

    output_dir: str = "artifacts/plots"

    plot_format: str = "png"
    dpi: int = 300

    enable_confusion_matrix: bool = True
    enable_model_comparison: bool = True
    enable_distribution_plots: bool = True


@dataclass
class EnvironmentConfig:
    """
    Configuration parameters for environment variables and runtime behavior.
    """

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    num_threads: int = field(default_factory=lambda: int(os.getenv("NUM_THREADS", "4")))
    experiment_name: str = field(
        default_factory=lambda: os.getenv("EXPERIMENT_NAME", "mental_health_detection")
    )


class ProjectConfig:
    """
    Central configuration object for the entire research project.

    This class aggregates all configuration sections and provides utilities
    for loading configurations from external files, validating parameters,
    and exporting configuration values for use across project modules.
    """

    def __init__(self, config_file: Optional[str] = None) -> None:
        """
        Initialize the configuration object.

        Parameters
        ----------
        config_file : Optional[str]
            Optional path to a JSON or YAML configuration file used to
            override default settings.
        """

        self.dataset = DatasetConfig()
        self.preprocessing = PreprocessingConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.cross_validation = CrossValidationConfig()
        self.visualization = VisualizationConfig()
        self.environment = EnvironmentConfig()

        if config_file:
            self._load_from_file(config_file)

        self._validate()

    def _load_from_file(self, path: str) -> None:
        """
        Load configuration overrides from JSON or YAML file.

        Parameters
        ----------
        path : str
            Path to configuration file.
        """

        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        elif path.endswith(".yaml") or path.endswith(".yml"):
            if yaml is None:
                raise ImportError("PyYAML must be installed to load YAML configuration files.")
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

        else:
            raise ValueError("Unsupported configuration format. Use JSON or YAML.")

        self._apply_overrides(data)

    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """
        Apply configuration overrides from a dictionary.

        Parameters
        ----------
        overrides : Dict[str, Any]
            Dictionary containing configuration overrides.
        """

        for section_name, section_values in overrides.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)

                for key, value in section_values.items():
                    if hasattr(section, key):
                        setattr(section, key, value)

    def _validate(self) -> None:
        """
        Validate configuration parameters to prevent invalid experimental setups.
        """

        if self.training.epochs <= 0:
            raise ValueError("Training epochs must be positive.")

        if self.model.rf_estimators <= 0:
            raise ValueError("RandomForest estimators must be positive.")

        if self.model.xgb_estimators <= 0:
            raise ValueError("XGBoost estimators must be positive.")

        if self.cross_validation.n_folds < 2:
            raise ValueError("Cross-validation folds must be >= 2.")

        if self.training.batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        if self.model.mlp_learning_rate <= 0:
            raise ValueError("MLP learning rate must be positive.")

    def as_dict(self) -> Dict[str, Any]:
        """
        Return configuration values as a dictionary.

        Returns
        -------
        Dict[str, Any]
            Nested dictionary representation of all configuration sections.
        """

        return {
            "dataset": asdict(self.dataset),
            "preprocessing": asdict(self.preprocessing),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "cross_validation": asdict(self.cross_validation),
            "visualization": asdict(self.visualization),
            "environment": asdict(self.environment),
        }