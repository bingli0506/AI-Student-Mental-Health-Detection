"""
Utility Module for HybridDLMLMentalHealthDetection

This module provides a centralized collection of utility functions used across
the HybridDL-ML mental health detection project. The utilities implemented here
support deterministic experimentation, logging infrastructure, file and path
management, model persistence, input validation, and evaluation metric
computation.

Design Philosophy
-----------------
The utilities are implemented with research reproducibility and engineering
reliability as primary objectives. Many academic machine learning experiments
suffer from non-determinism, insufficient logging, or inconsistent evaluation
metrics. This module addresses these concerns by providing standardized helpers
that enforce reproducible behavior and consistent computation.

Core Functional Areas
---------------------
1. Random Seed Management
   Ensures deterministic behavior across Python, NumPy, and environment-level
   randomness sources.

2. Logging Utilities
   Provides configurable logging setup suitable for both development and
   large-scale experiment tracking.

3. File and Path Operations
   Handles directory creation, file existence checks, and safe file writing.

4. Model Persistence
   Functions to save and load trained models using serialization mechanisms.

5. Evaluation Metrics
   Implements common classification metrics including accuracy, precision,
   recall, and F1-score.

6. Input Validation
   Provides defensive programming utilities to validate data before model
   training or evaluation.

These utilities are intentionally dependency-light to avoid circular imports
and maintain modular architecture compliance.

All utilities are grouped under the ProjectUtils class to provide a clean
namespace for project-wide helper functionality.
"""

import os
import sys
import json
import random
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import numpy as np


class ProjectUtils:
    """
    Collection of static utility functions used throughout the project.

    The class acts as a namespace container for general-purpose helper
    utilities. No internal state is stored in this class; all methods are
    implemented as static methods to simplify usage and avoid unnecessary
    object instantiation.

    Example
    -------
    ProjectUtils.set_seed(42)
    logger = ProjectUtils.setup_logging("logs/experiment.log")
    metrics = ProjectUtils.compute_metrics(y_true, y_pred)
    """

    @staticmethod
    def set_seed(seed: int) -> None:
        """
        Set deterministic random seeds for reproducible experiments.

        This method configures random seeds across Python's random module,
        NumPy, and environment-level hash randomization to ensure consistent
        results across repeated runs.

        Parameters
        ----------
        seed : int
            The seed value used for deterministic random number generation.

        Notes
        -----
        Deterministic seed initialization is critical in machine learning
        research to ensure experimental reproducibility, especially when
        models rely on stochastic optimization or random sampling.

        The function affects:
        - Python random module
        - NumPy random generator
        - Python hash seed
        """
        if not isinstance(seed, int):
            raise TypeError("Seed must be an integer.")

        if seed < 0:
            raise ValueError("Seed must be non-negative.")

        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    @staticmethod
    def setup_logging(log_path: Optional[str] = None,
                      level: int = logging.INFO) -> logging.Logger:
        """
        Configure and return a project-wide logger.

        Parameters
        ----------
        log_path : Optional[str]
            Optional path where logs will be written.
        level : int
            Logging verbosity level.

        Returns
        -------
        logging.Logger
            Configured logger instance.
        """
        logger = logging.getLogger("HybridDLMLMentalHealthDetection")

        if logger.handlers:
            return logger

        logger.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_path:
            ProjectUtils.ensure_dir(Path(log_path).parent)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    @staticmethod
    def ensure_dir(path: Path) -> None:
        """
        Ensure that a directory exists.

        Parameters
        ----------
        path : Path
            Directory path that should exist.

        Notes
        -----
        If the directory does not exist, it will be created recursively.
        """
        if not isinstance(path, Path):
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_model(model: Any, path: str) -> None:
        """
        Persist a trained model to disk using pickle serialization.

        Parameters
        ----------
        model : Any
            Trained model object to be saved.
        path : str
            Destination file path.

        Raises
        ------
        IOError
            If the model cannot be written to disk.
        """
        if model is None:
            raise ValueError("Model cannot be None.")

        if not path:
            raise ValueError("Invalid model save path.")

        directory = Path(path).parent
        ProjectUtils.ensure_dir(directory)

        try:
            with open(path, "wb") as f:
                pickle.dump(model, f)
        except Exception as exc:
            raise IOError(f"Failed to save model: {exc}")

    @staticmethod
    def load_model(path: str) -> Any:
        """
        Load a serialized model from disk.

        Parameters
        ----------
        path : str
            Path to the serialized model file.

        Returns
        -------
        Any
            Deserialized model object.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
        except Exception as exc:
            raise IOError(f"Failed to load model: {exc}")

        return model

    @staticmethod
    def calculate_accuracy(y_true: Iterable, y_pred: Iterable) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        y_true : Iterable
            Ground truth labels.
        y_pred : Iterable
            Predicted labels.

        Returns
        -------
        float
            Accuracy score.
        """
        y_true = np.array(list(y_true))
        y_pred = np.array(list(y_pred))

        if len(y_true) == 0:
            raise ValueError("Empty ground truth labels.")

        correct = np.sum(y_true == y_pred)
        return float(correct) / float(len(y_true))

    @staticmethod
    def calculate_precision(y_true: Iterable, y_pred: Iterable) -> float:
        """
        Compute macro-averaged precision.

        Parameters
        ----------
        y_true : Iterable
            True labels.
        y_pred : Iterable
            Predicted labels.

        Returns
        -------
        float
            Precision score.
        """
        y_true = np.array(list(y_true))
        y_pred = np.array(list(y_pred))

        classes = np.unique(y_true)
        precisions: List[float] = []

        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fp = np.sum((y_pred == cls) & (y_true != cls))

            if tp + fp == 0:
                precisions.append(0.0)
            else:
                precisions.append(tp / (tp + fp))

        return float(np.mean(precisions))

    @staticmethod
    def calculate_recall(y_true: Iterable, y_pred: Iterable) -> float:
        """
        Compute macro-averaged recall.

        Parameters
        ----------
        y_true : Iterable
            Ground truth labels.
        y_pred : Iterable
            Predicted labels.

        Returns
        -------
        float
            Recall score.
        """
        y_true = np.array(list(y_true))
        y_pred = np.array(list(y_pred))

        classes = np.unique(y_true)
        recalls: List[float] = []

        for cls in classes:
            tp = np.sum((y_pred == cls) & (y_true == cls))
            fn = np.sum((y_pred != cls) & (y_true == cls))

            if tp + fn == 0:
                recalls.append(0.0)
            else:
                recalls.append(tp / (tp + fn))

        return float(np.mean(recalls))

    @staticmethod
    def calculate_f1_score(y_true: Iterable, y_pred: Iterable) -> float:
        """
        Compute macro-averaged F1 score.

        Parameters
        ----------
        y_true : Iterable
            Ground truth labels.
        y_pred : Iterable
            Predicted labels.

        Returns
        -------
        float
            F1 score.
        """
        precision = ProjectUtils.calculate_precision(y_true, y_pred)
        recall = ProjectUtils.calculate_recall(y_true, y_pred)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def compute_metrics(y_true: Iterable, y_pred: Iterable) -> Dict[str, float]:
        """
        Compute a dictionary of standard classification metrics.

        Parameters
        ----------
        y_true : Iterable
            Ground truth labels.
        y_pred : Iterable
            Predicted labels.

        Returns
        -------
        Dict[str, float]
            Dictionary containing accuracy, precision, recall, and F1-score.
        """
        accuracy = ProjectUtils.calculate_accuracy(y_true, y_pred)
        precision = ProjectUtils.calculate_precision(y_true, y_pred)
        recall = ProjectUtils.calculate_recall(y_true, y_pred)
        f1 = ProjectUtils.calculate_f1_score(y_true, y_pred)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    @staticmethod
    def validate_inputs(features: Any, labels: Optional[Any] = None) -> None:
        """
        Validate feature and label inputs before model operations.

        Parameters
        ----------
        features : Any
            Feature matrix or iterable of samples.
        labels : Optional[Any]
            Target labels associated with features.

        Raises
        ------
        ValueError
            If features or labels are invalid.
        """
        if features is None:
            raise ValueError("Feature data cannot be None.")

        if isinstance(features, (list, tuple, np.ndarray)) and len(features) == 0:
            raise ValueError("Feature data cannot be empty.")

        if labels is not None:
            if isinstance(labels, (list, tuple, np.ndarray)) and len(labels) == 0:
                raise ValueError("Label data cannot be empty.")

            if len(features) != len(labels):
                raise ValueError("Feature and label lengths must match.")

    @staticmethod
    def save_json(data: Dict[str, Any], path: str) -> None:
        """
        Save dictionary data as a JSON file.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary to serialize.
        path : str
            Output file path.
        """
        ProjectUtils.ensure_dir(Path(path).parent)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as exc:
            raise IOError(f"Failed to write JSON file: {exc}")

    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        """
        Load dictionary data from a JSON file.

        Parameters
        ----------
        path : str
            Path to JSON file.

        Returns
        -------
        Dict[str, Any]
            Loaded dictionary.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSON file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            raise IOError(f"Failed to read JSON file: {exc}")