"""
models.model_retrieval
----------------------

This module implements the SimilarCaseRetriever component used in the
HybridDL-ML-Ensemble mental health early detection system.

The retriever is responsible for identifying historically similar student
cases given a query instance. This capability provides interpretability
and decision support by allowing clinicians, researchers, or institutional
support systems to examine previously observed student profiles that most
closely resemble the current case.

Architectural Role
------------------
Within the overall project architecture, this module provides a retrieval
mechanism operating alongside predictive models. While base learners
(Random Forest, XGBoost, MLP, CNN-1D) produce predictions, the retrieval
module enables case-based reasoning by locating similar instances in a
historical feature space.

This approach improves interpretability in mental health prediction tasks,
where model transparency is important. Rather than presenting only a
predicted class label, the system can reference comparable historical
cases and their associated outcomes.

Theoretical Background
----------------------
The retrieval process is based on similarity search in a high-dimensional
feature space. Each student record is represented as a numerical vector
after preprocessing and feature encoding.

Given:
    X = {x1, x2, ..., xN}
where xi ∈ R^d represents a student feature vector.

For a query vector q, similarity is computed using cosine similarity:

    sim(q, xi) = (q · xi) / (||q|| * ||xi||)

Cosine similarity is chosen because:
    - It is robust to magnitude differences
    - It works well in high-dimensional spaces
    - It captures directional similarity between feature vectors

The retriever returns the top-k most similar historical cases based on
this similarity score.

Design Considerations
---------------------
1. Efficient vectorized similarity computation using NumPy
2. Support for dynamic case base updates
3. Robust input validation
4. Defensive programming for reproducibility
5. Clear separation between indexing and querying operations

Usage Workflow
--------------
1. Initialize retriever
2. Build case index from historical features and labels
3. Query similar cases for interpretability

Example:

    retriever = SimilarCaseRetriever()
    retriever.build_index(features, labels)

    similar_cases = retriever.retrieve(query_vector, top_k=5)

The returned cases contain similarity scores and associated metadata.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class SimilarCaseRetriever:
    """
    Case-based retrieval system for mental health prediction interpretability.

    This class stores historical student records in a feature space and
    retrieves the most similar cases to a query instance. It supports
    cosine similarity-based retrieval and can be integrated with the
    ensemble prediction pipeline.

    Attributes
    ----------
    case_features : Optional[np.ndarray]
        Matrix of stored feature vectors (N x D).
    case_labels : Optional[np.ndarray]
        Corresponding labels or risk categories.
    metadata : List[Dict[str, Any]]
        Optional metadata associated with each case.
    fitted : bool
        Indicates whether the retriever index has been built.
    """

    def __init__(self) -> None:
        """
        Initialize an empty retrieval system.
        """
        self.case_features: Optional[np.ndarray] = None
        self.case_labels: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []
        self.fitted: bool = False

    def build_index(
        self,
        features: np.ndarray,
        labels: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Build the internal case index used for similarity retrieval.

        Parameters
        ----------
        features : np.ndarray
            Matrix of student feature representations with shape (N, D).
        labels : Optional[np.ndarray]
            Optional label array representing mental health risk levels.
        metadata : Optional[List[Dict[str, Any]]]
            Optional metadata dictionaries describing each case.

        Raises
        ------
        ValueError
            If input features are invalid.
        """

        if features is None:
            raise ValueError("Features cannot be None.")

        if not isinstance(features, np.ndarray):
            raise TypeError("Features must be a NumPy array.")

        if features.ndim != 2:
            raise ValueError("Features must be a 2D matrix.")

        self.case_features = features.astype(np.float32)

        if labels is not None:
            if len(labels) != len(features):
                raise ValueError("Labels must match feature count.")
            self.case_labels = np.asarray(labels)

        if metadata is not None:
            if len(metadata) != len(features):
                raise ValueError("Metadata length must match feature count.")
            self.metadata = metadata
        else:
            self.metadata = [{} for _ in range(len(features))]

        self.fitted = True

    def _cosine_similarity(
        self,
        query: np.ndarray,
        matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between a query vector and matrix.

        Parameters
        ----------
        query : np.ndarray
            Query feature vector (D,).
        matrix : np.ndarray
            Feature matrix (N x D).

        Returns
        -------
        np.ndarray
            Similarity scores (N,).
        """

        query_norm = np.linalg.norm(query)
        matrix_norm = np.linalg.norm(matrix, axis=1)

        if query_norm == 0:
            raise ValueError("Query vector has zero magnitude.")

        dot_products = matrix @ query
        similarity = dot_products / (matrix_norm * query_norm + 1e-12)

        return similarity

    def retrieve(self, query, top_k: int = 5) -> list:
        """
        Retrieve the top-k most similar historical student cases.

        This function performs similarity search over the stored feature
        index and returns the most relevant cases based on cosine
        similarity.

        Parameters
        ----------
        query : array-like
            Feature representation of the query student case.
        top_k : int
            Number of similar cases to return.

        Returns
        -------
        list
            List of dictionaries describing similar cases. Each entry
            contains:
                - index: case index
                - similarity: similarity score
                - label: associated mental health label (if available)
                - metadata: optional descriptive information

        Raises
        ------
        RuntimeError
            If the retriever index has not been built.
        """

        if not self.fitted:
            raise RuntimeError(
                "Retriever index has not been built. "
                "Call build_index() before retrieval."
            )

        if query is None:
            raise ValueError("Query cannot be None.")

        query_vector = np.asarray(query).astype(np.float32)

        if query_vector.ndim != 1:
            raise ValueError("Query must be a 1D feature vector.")

        similarities = self._cosine_similarity(query_vector, self.case_features)

        top_k = max(1, min(top_k, len(similarities)))

        ranked_indices = np.argsort(similarities)[::-1][:top_k]

        results: List[Dict[str, Any]] = []

        for idx in ranked_indices:
            entry: Dict[str, Any] = {
                "index": int(idx),
                "similarity": float(similarities[idx]),
                "metadata": self.metadata[idx]
            }

            if self.case_labels is not None:
                entry["label"] = int(self.case_labels[idx])

            results.append(entry)

        return results

    def add_case(
        self,
        feature_vector: np.ndarray,
        label: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a new case to the retrieval index.

        Parameters
        ----------
        feature_vector : np.ndarray
            Feature vector representing a student case.
        label : Optional[int]
            Mental health risk category.
        metadata : Optional[Dict[str, Any]]
            Optional descriptive metadata.
        """

        vector = np.asarray(feature_vector).astype(np.float32)

        if vector.ndim != 1:
            raise ValueError("Feature vector must be 1D.")

        if self.case_features is None:
            self.case_features = vector.reshape(1, -1)
            self.case_labels = (
                np.array([label]) if label is not None else None
            )
            self.metadata = [metadata or {}]
        else:
            self.case_features = np.vstack([self.case_features, vector])

            if label is not None:
                if self.case_labels is None:
                    self.case_labels = np.array([label])
                else:
                    self.case_labels = np.append(self.case_labels, label)

            self.metadata.append(metadata or {})

        self.fitted = True

    def get_case_count(self) -> int:
        """
        Return the number of indexed cases.

        Returns
        -------
        int
            Total number of stored cases.
        """

        if self.case_features is None:
            return 0

        return len(self.case_features)