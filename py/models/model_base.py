"""
models.model_base

This module defines the foundational model abstraction used throughout the
HybridDL-ML-Ensemble architecture for early detection of mental health risks
among university students. The design follows academic research standards
with a clear abstraction layer enabling heterogeneous model implementations
such as Random Forest, XGBoost, Multilayer Perceptron (MLP), CNN-1D, and
meta-learning classifiers.

The BaseMentalHealthModel serves as the unified interface for all predictive
models used in the system. It enforces consistent behavior across classical
machine learning models and deep learning architectures. This abstraction
facilitates reproducible experimentation, model comparison, ensemble learning,
and integration within the broader project pipeline.

Design Principles
-----------------
1. Reproducibility
   The class supports deterministic training, model serialization, and
   parameter inspection to facilitate academic reproducibility.

2. Extensibility
   New models can be implemented by inheriting from BaseMentalHealthModel
   and implementing required methods without modifying upstream code.

3. Framework Agnostic Interface
   Although this class inherits from torch.nn.Module for deep learning
   compatibility, it can also wrap traditional machine learning models.

4. Robustness and Validation
   Built-in validation ensures correct inputs during training and inference.

5. Experiment Transparency
   The module includes utilities for parameter inspection, model statistics,
   and checkpoint persistence.

The base class is intended to be subclassed by concrete implementations such as:
    - RandomForestModel
    - XGBoostModel
    - MLPModel
    - CNN1DModel
    - LogisticRegressionMetaLearner

All derived models should follow the same training and prediction interface,
allowing them to be seamlessly integrated within stacking and ensemble
strategies.

Author: HybridDLMLMentalHealthDetection Research Framework
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import os
import json
import torch
import torch.nn as nn


class BaseMentalHealthModel(nn.Module, ABC):
    """
    Abstract base class for all predictive models used in the mental health
    early detection framework.

    This class defines the minimal interface that any model must implement in
    order to be compatible with the ensemble architecture and experimental
    evaluation pipeline.

    The class extends torch.nn.Module so that neural network architectures can
    be implemented directly while still allowing classical machine learning
    models to be wrapped inside subclasses.

    Attributes
    ----------
    model_name : str
        Human-readable identifier of the model.

    device : torch.device
        Device used for computation (CPU or GPU).

    is_trained : bool
        Indicates whether the model has been fitted.

    metadata : Dict[str, Any]
        Dictionary storing training metadata and configuration details.
    """

    def __init__(self, model_name: str = "BaseMentalHealthModel",
                 device: Optional[str] = None) -> None:
        super().__init__()

        self.model_name: str = model_name
        self.device: torch.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.is_trained: bool = False
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation for neural network based models.

        Subclasses implementing deep learning architectures must override this
        method. Classical machine learning wrappers may leave this unused.

        Parameters
        ----------
        features : torch.Tensor
            Input feature tensor representing encoded student attributes.

        Returns
        -------
        torch.Tensor
            Model predictions or probability logits.
        """
        raise NotImplementedError("Forward method must be implemented.")

    @abstractmethod
    def fit(self, features, labels) -> None:
        """
        Train the model on the provided feature matrix and labels.

        This method must implement the training logic appropriate for the
        specific model type. For neural models this may involve gradient-based
        optimization; for classical models it may delegate to external libraries.

        Parameters
        ----------
        features : Any
            Feature matrix representing encoded student data.

        labels : Any
            Target labels representing mental health risk classes.

        Returns
        -------
        None
        """
        raise NotImplementedError("Subclasses must implement the fit method.")

    @abstractmethod
    def predict(self, features):
        """
        Generate predictions for the given feature matrix.

        Parameters
        ----------
        features : Any
            Input feature matrix.

        Returns
        -------
        Any
            Predicted class labels or probabilities depending on model design.
        """
        raise NotImplementedError("Subclasses must implement the predict method.")

    def save(self, path: str) -> None:
        """
        Persist model parameters and metadata to disk.

        The model is saved in a reproducible format containing both
        learned parameters and supplementary metadata.

        Parameters
        ----------
        path : str
            Destination file path.

        Returns
        -------
        None
        """
        if not path:
            raise ValueError("A valid file path must be provided.")

        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        checkpoint = {
            "model_name": self.model_name,
            "state_dict": self.state_dict(),
            "metadata": self.metadata,
            "is_trained": self.is_trained
        }

        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """
        Load model parameters from a previously saved checkpoint.

        Parameters
        ----------
        path : str
            Path to the saved model file.

        Returns
        -------
        None
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.load_state_dict(checkpoint["state_dict"])
        self.metadata = checkpoint.get("metadata", {})
        self.is_trained = checkpoint.get("is_trained", False)

    def get_parameters(self) -> Dict[str, Any]:
        """
        Retrieve detailed statistics about model parameters.

        This method is useful for experiment tracking, model auditing,
        and reproducibility documentation in research environments.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing parameter statistics including total
            parameters, trainable parameters, and model name.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": self.model_name,
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "device": str(self.device),
            "is_trained": self.is_trained
        }

    def validate_inputs(self, features, labels=None) -> None:
        """
        Validate training or inference inputs.

        Ensures that feature matrices and labels satisfy minimal structural
        requirements before being passed to the model.

        Parameters
        ----------
        features : Any
            Feature matrix.

        labels : Any, optional
            Target labels.

        Returns
        -------
        None
        """
        if features is None:
            raise ValueError("Features must not be None.")

        if hasattr(features, "__len__") and len(features) == 0:
            raise ValueError("Feature matrix is empty.")

        if labels is not None:
            if hasattr(labels, "__len__") and len(labels) == 0:
                raise ValueError("Label vector is empty.")

            if hasattr(features, "__len__") and hasattr(labels, "__len__"):
                if len(features) != len(labels):
                    raise ValueError("Feature and label lengths do not match.")

    def summary(self) -> str:
        """
        Produce a structured textual summary of the model.

        This method is useful for experiment logging and debugging.

        Returns
        -------
        str
            JSON formatted summary describing the model.
        """
        info = self.get_parameters()
        info["metadata"] = self.metadata

        return json.dumps(info, indent=2)

    def __repr__(self) -> str:
        """
        Human-readable representation of the model instance.
        """
        params = self.get_parameters()
        return f"{self.__class__.__name__}(name={params['model_name']}, parameters={params['total_parameters']})"