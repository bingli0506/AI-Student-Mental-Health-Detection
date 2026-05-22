"""
strategy_relevance.py

Relevance Strategy Module for HybridDL-ML-Ensemble Mental Health Detection System.

This module implements a research-oriented relevance scoring strategy used to
estimate the importance or early-warning relevance of student features within
the mental health detection framework. The strategy is motivated by the
methodology described in the HybridDL-ML-Ensemble architecture, where
demographic attributes, academic performance indicators, and psychological
signals jointly contribute to mental health risk prediction.

The purpose of this module is to compute a normalized relevance score for
incoming feature vectors representing student records. This score reflects
the degree to which the observed indicators correspond to patterns associated
with higher mental health risk levels.

Conceptual Background
---------------------
The relevance scoring mechanism is inspired by the ensemble methodology
outlined in the referenced research paper:

1. Multiple heterogeneous models (Random Forest, XGBoost, MLP, CNN-1D)
   generate probabilistic predictions of mental health risk categories.

2. Feature interactions across demographic, academic, and psychological
   indicators influence model outputs.

3. Certain indicators (e.g., CGPA decline, psychological symptoms,
   academic stress markers) carry stronger predictive significance.

This module operationalizes these insights by constructing a weighted
relevance model that measures how strongly a given student record aligns
with known high-risk patterns.

Relevance scoring is computed using a hybrid approach:

    R(x) = Σ w_i * f_i(x)

Where:
    - x represents the student feature vector
    - f_i(x) represents normalized feature contributions
    - w_i represents empirically motivated importance weights

The resulting score is normalized to the range [0, 1], enabling consistent
interpretation across different datasets and experimental configurations.

Design Goals
------------
The implementation emphasizes:

• Reproducibility for academic research
• Interpretability of relevance scores
• Compatibility with ensemble prediction pipelines
• Extensibility for future feature integration
• Robust error handling and input validation

This module is typically used by higher-level strategy components
during early-warning analysis or model interpretation workflows.
"""

from typing import Dict, List, Union, Optional
import math


Numeric = Union[int, float]


class RelevanceStrategy:
    """
    Relevance scoring strategy for student mental health risk indicators.

    This class computes a normalized relevance score for a given set of
    features representing a student's demographic, academic, and mental
    health indicators.

    The relevance score reflects the strength of association between the
    student's indicators and patterns historically correlated with mental
    health distress.

    Parameters
    ----------
    feature_weights : Optional[Dict[str, float]]
        Optional custom feature weight configuration. If not provided,
        default empirically motivated weights are used.

    normalization_bounds : Optional[Dict[str, tuple]]
        Feature normalization ranges used for min-max normalization.

    Notes
    -----
    The strategy prioritizes several categories of indicators:

    • Psychological indicators (depression, anxiety, panic attacks)
    • Academic stress indicators (CGPA changes)
    • Demographic vulnerability markers (age, study year)
    • Behavioral markers (treatment seeking)

    Scores are computed using weighted aggregation and then normalized.
    """

    def __init__(
        self,
        feature_weights: Optional[Dict[str, float]] = None,
        normalization_bounds: Optional[Dict[str, tuple]] = None
    ) -> None:
        """
        Initialize the relevance strategy.

        Parameters
        ----------
        feature_weights : dict, optional
            Custom feature importance weights.

        normalization_bounds : dict, optional
            Feature normalization boundaries for min-max scaling.
        """

        # Default feature importance weights inspired by feature importance
        # analysis described in the research paper.
        self.feature_weights: Dict[str, float] = feature_weights or {
            "cgpa": 0.24,
            "age": 0.17,
            "mental_health_total": 0.18,
            "year_of_study": 0.10,
            "gender_female": 0.08,
            "gender_male": 0.04,
            "marital_status_single": 0.05,
            "treatment_sought": 0.07,
            "anxiety": 0.03,
            "depression": 0.02,
            "panic_attack": 0.02
        }

        # Normalization boundaries for continuous variables
        self.normalization_bounds: Dict[str, tuple] = normalization_bounds or {
            "age": (17, 30),
            "cgpa": (0.0, 4.0),
            "mental_health_total": (0, 3),
            "year_of_study": (1, 5)
        }

    def score(self, features) -> float:
        """
        Compute the relevance score for a student feature vector.

        Parameters
        ----------
        features : dict or list
            Feature representation of the student record. A dictionary
            is preferred for interpretability.

        Returns
        -------
        float
            Normalized relevance score in the range [0, 1].

        Raises
        ------
        ValueError
            If feature input is invalid or incomplete.

        Notes
        -----
        The scoring process follows three steps:

        1. Validate and normalize incoming features.
        2. Compute weighted feature contributions.
        3. Normalize aggregated score to [0,1].
        """

        feature_dict = self._validate_features(features)

        normalized_features = self._normalize_features(feature_dict)

        raw_score = self._compute_weighted_score(normalized_features)

        normalized_score = self._normalize_score(raw_score)

        return normalized_score

    def _validate_features(self, features) -> Dict[str, Numeric]:
        """
        Validate and standardize feature input.

        Parameters
        ----------
        features : dict or list

        Returns
        -------
        dict
            Standardized feature dictionary.
        """

        if features is None:
            raise ValueError("Feature input cannot be None.")

        if isinstance(features, dict):
            return features

        raise ValueError(
            "Features must be provided as a dictionary mapping feature names to values."
        )

    def _normalize_features(self, features: Dict[str, Numeric]) -> Dict[str, float]:
        """
        Apply min-max normalization to continuous features.

        Parameters
        ----------
        features : dict

        Returns
        -------
        dict
            Normalized feature values.
        """

        normalized: Dict[str, float] = {}

        for key, value in features.items():

            if key in self.normalization_bounds:
                min_val, max_val = self.normalization_bounds[key]

                if max_val == min_val:
                    normalized[key] = 0.0
                else:
                    normalized[key] = (float(value) - min_val) / (max_val - min_val)

                normalized[key] = max(0.0, min(1.0, normalized[key]))

            else:
                # Binary or already normalized feature
                normalized[key] = float(value)

        return normalized

    def _compute_weighted_score(self, features: Dict[str, float]) -> float:
        """
        Compute weighted relevance score.

        Parameters
        ----------
        features : dict

        Returns
        -------
        float
            Raw weighted score.
        """

        score = 0.0

        for feature_name, weight in self.feature_weights.items():

            value = features.get(feature_name, 0.0)

            score += weight * value

        return score

    def _normalize_score(self, score: float) -> float:
        """
        Normalize aggregated relevance score to [0, 1].

        Parameters
        ----------
        score : float

        Returns
        -------
        float
            Normalized score.
        """

        max_possible_score = sum(self.feature_weights.values())

        if max_possible_score == 0:
            return 0.0

        normalized = score / max_possible_score

        return max(0.0, min(1.0, normalized))

    def rank(self, records: List[Dict[str, Numeric]]) -> List[Dict[str, Union[float, Dict]]]:
        """
        Rank student records by relevance score.

        Parameters
        ----------
        records : list of dict
            List of student feature dictionaries.

        Returns
        -------
        list
            Sorted list of records with relevance scores attached.
        """

        scored_records = []

        for record in records:
            relevance = self.score(record)

            scored_records.append({
                "features": record,
                "relevance_score": relevance
            })

        scored_records.sort(
            key=lambda item: item["relevance_score"],
            reverse=True
        )

        return scored_records

    def explain(self, features: Dict[str, Numeric]) -> Dict[str, float]:
        """
        Provide feature-level contribution explanation.

        Parameters
        ----------
        features : dict

        Returns
        -------
        dict
            Feature contribution breakdown.
        """

        normalized = self._normalize_features(features)

        contributions: Dict[str, float] = {}

        for feature_name, weight in self.feature_weights.items():

            value = normalized.get(feature_name, 0.0)

            contributions[feature_name] = weight * value

        return contributions

    def batch_score(self, records: List[Dict[str, Numeric]]) -> List[float]:
        """
        Compute relevance scores for multiple student records.

        Parameters
        ----------
        records : list of dict

        Returns
        -------
        list
            List of normalized relevance scores.
        """

        scores: List[float] = []

        for record in records:
            scores.append(self.score(record))

        return scores