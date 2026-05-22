"""
models/model_encoder.py

This module implements the StudentFeatureEncoder, a representation learning
component used within the HybridDL-ML-Ensemble architecture for early detection
of mental health issues among university students.

The encoder transforms structured student attributes (demographic, academic,
and psychological indicators) into a latent feature representation suitable
for downstream machine learning and deep learning models.

Architectural Motivation
------------------------
The HybridDL-ML-Ensemble system integrates heterogeneous learners including
Random Forest, XGBoost, MLP, and CNN-1D. Although classical tree models operate
directly on tabular inputs, deep models benefit from structured feature
embeddings that capture nonlinear interactions between variables such as:

    - Age and academic stress (CGPA)
    - Demographic factors (gender, marital status)
    - Course and year of study
    - Self-reported psychological indicators

The StudentFeatureEncoder provides a shared transformation layer that:

1. Normalizes and validates input feature tensors.
2. Projects raw features into a latent representation space.
3. Captures nonlinear relationships through multilayer transformations.
4. Produces embeddings suitable for downstream ensemble models.

Design Characteristics
----------------------
The encoder follows a lightweight feed-forward architecture inspired by the
MLP component described in the research paper:

    Input Features
        ↓
    Linear Projection
        ↓
    Nonlinear Activation (ReLU)
        ↓
    Hidden Layer Representation
        ↓
    Latent Feature Embedding

Although simple, this architecture is sufficient for structured tabular data
where relationships between features are moderately nonlinear.

Integration with Project Architecture
-------------------------------------
This module integrates with:

    models.model_base.BaseMentalHealthModel
        Provides a base abstraction for ML components.

    data.data_preprocess.MentalHealthPreprocessor
        Supplies cleaned and normalized feature matrices.

    strategies and experiments modules
        Consume encoded representations for training and evaluation.

Reproducibility and Research Quality
------------------------------------
The implementation includes:

    - Input validation
    - Deterministic initialization
    - Type annotations
    - Extensive documentation
    - Clear separation of encoding and model logic

This ensures compatibility with academic research workflows and reproducibility
requirements.
"""

from typing import Any, Iterable
import numpy as np

from models.model_base import BaseMentalHealthModel


class StudentFeatureEncoder(BaseMentalHealthModel):
    """
    StudentFeatureEncoder

    A neural-inspired feature encoder that converts structured student
    attributes into latent representations for mental health prediction.

    The encoder performs a deterministic feed-forward transformation of
    feature vectors into a compact latent space. It is designed to improve
    representation quality before models such as MLP or CNN-1D process the
    features.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
    hidden_dim : int
        Size of the intermediate hidden layer.
    latent_dim : int
        Dimensionality of the output encoded representation.
    seed : int
        Random seed for deterministic initialization.

    Attributes
    ----------
    input_dim : int
        Input feature dimensionality.
    hidden_dim : int
        Hidden layer size.
    latent_dim : int
        Latent embedding size.
    W1 : np.ndarray
        Weight matrix for the first projection layer.
    b1 : np.ndarray
        Bias vector for the first layer.
    W2 : np.ndarray
        Weight matrix for latent projection.
    b2 : np.ndarray
        Bias vector for latent layer.

    Notes
    -----
    The encoder is intentionally lightweight because the dataset used in the
    research study contains approximately 100 samples. Overly complex
    representation networks would risk overfitting.

    Therefore, a shallow projection network is employed to balance expressive
    power and generalization.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        seed: int = 42
    ) -> None:
        """
        Initialize the StudentFeatureEncoder.

        Parameters
        ----------
        input_dim : int
            Number of features in the input vector.
        hidden_dim : int
            Number of neurons in the hidden layer.
        latent_dim : int
            Size of the latent embedding produced by the encoder.
        seed : int
            Random seed for reproducibility.
        """

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seed = seed

        # Initialize deterministic random generator
        rng = np.random.RandomState(seed)

        # Xavier-style initialization for stability
        self.W1 = rng.normal(0, 1 / np.sqrt(input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        self.W2 = rng.normal(0, 1 / np.sqrt(hidden_dim), (hidden_dim, latent_dim))
        self.b2 = np.zeros(latent_dim)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the Rectified Linear Unit activation function.

        ReLU is widely used in deep learning architectures due to its
        computational efficiency and ability to mitigate vanishing gradients.

        Parameters
        ----------
        x : np.ndarray
            Input tensor.

        Returns
        -------
        np.ndarray
            Activated tensor.
        """
        return np.maximum(0, x)

    def _validate_input(self, features: Any) -> np.ndarray:
        """
        Validate and convert input features into a numpy array.

        Parameters
        ----------
        features : Any
            Input features (list, tuple, or numpy array).

        Returns
        -------
        np.ndarray
            Validated feature matrix.

        Raises
        ------
        ValueError
            If the input dimensions are inconsistent with the encoder
            configuration.
        """

        if isinstance(features, np.ndarray):
            X = features
        elif isinstance(features, Iterable):
            X = np.array(list(features), dtype=float)
        else:
            raise TypeError("Features must be iterable or numpy array.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected feature dimension {self.input_dim}, "
                f"but received {X.shape[1]}"
            )

        return X

    def encode(self, features):
        """
        Encode student indicators into latent representations.

        This method applies a two-layer nonlinear projection to transform
        raw student attributes into a compact representation vector.

        Encoding Process
        ----------------
        1. Input validation and shape normalization.
        2. Linear projection to hidden feature space.
        3. ReLU activation to introduce nonlinearity.
        4. Linear transformation to latent representation.

        Parameters
        ----------
        features : array-like
            Input student feature vectors.

        Returns
        -------
        np.ndarray
            Encoded latent representations with shape (N, latent_dim).
        """

        X = self._validate_input(features)

        # First projection layer
        hidden = np.dot(X, self.W1) + self.b1
        hidden = self._relu(hidden)

        # Latent embedding projection
        latent = np.dot(hidden, self.W2) + self.b2

        return latent

    def fit(self, features, labels) -> None:
        """
        Fit method placeholder to maintain compatibility with BaseMentalHealthModel.

        The StudentFeatureEncoder is designed as a deterministic feature
        transformation module rather than a trainable supervised model.
        Therefore, the fit method does not update parameters.

        Parameters
        ----------
        features : array-like
            Input training features.
        labels : array-like
            Corresponding labels (unused).

        Notes
        -----
        Future extensions could implement representation learning using
        reconstruction objectives or contrastive learning.
        """

        # Currently no training procedure required
        _ = self._validate_input(features)
        return None

    def predict(self, features):
        """
        Predict method inherited from BaseMentalHealthModel interface.

        For encoder modules, prediction corresponds to returning encoded
        representations rather than class labels.

        Parameters
        ----------
        features : array-like
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Encoded feature representations.
        """
        return self.encode(features)

    def get_config(self) -> dict:
        """
        Return encoder configuration parameters.

        Returns
        -------
        dict
            Dictionary describing encoder architecture.
        """
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "seed": self.seed
        }