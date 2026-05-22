"""
strategies/strategy_sequence.py

Sequence Strategy Module for HybridDLMLMentalHealthDetection

This module implements the SequenceStrategy component responsible for
processing temporal behavioral or academic indicators related to student
mental health risk. Although the primary dataset in the referenced research
is cross-sectional, real-world deployments of early mental health detection
systems often involve sequences of events collected over time. These may
include repeated survey responses, academic performance updates, behavioral
logs, or support-service interactions.

The purpose of the sequence strategy is to transform a list of chronological
events into interpretable temporal indicators that can be used by downstream
models or decision-support systems.

The design of this module follows principles derived from the methodology
described in the research paper "The Application of AI-Based Early Detection
of Mental Health Issues Among University Students in Academic Settings".
While the paper focuses on a hybrid ensemble architecture (Random Forest,
XGBoost, MLP, CNN-1D with Logistic Regression stacking), temporal analysis
can complement the model by identifying trends in behavioral indicators.

Core responsibilities of this module include:

1. Temporal summarization of student behavioral events
2. Detection of trends in academic or psychological indicators
3. Computation of aggregate statistics across time
4. Identification of potential early-warning signals
5. Construction of interpretable sequence-level features

Sequence modeling approach implemented here:

Given a sequence of events:

    E = [e1, e2, ..., et]

Each event is expected to contain numerical or categorical attributes such as:

    - timestamp
    - CGPA
    - stress_score
    - anxiety_flag
    - depression_flag
    - panic_flag

The strategy computes:

    - Temporal statistics (mean, variance, min, max)
    - Trend estimation using linear regression
    - Recent vs historical change signals
    - Frequency of risk indicators
    - Composite temporal risk score

These features can be used by:

    - Machine learning classifiers
    - Ensemble meta-learning layers
    - Risk monitoring dashboards
    - Early warning systems

The implementation emphasizes:
    * Deterministic processing
    * Robust handling of incomplete data
    * Transparent statistical calculations
    * Reproducibility for research purposes
    * Extensibility for future multimodal inputs

Author: HybridDLMLMentalHealthDetection System
"""

from typing import Any, Dict, List, Optional
import statistics
import math


class SequenceStatistics:
    """
    Utility class for computing statistical properties of temporal signals.

    This class encapsulates standard statistical operations commonly used
    in time-series summarization including mean, variance, min/max detection,
    and trend estimation.

    The methods are intentionally implemented without heavy external
    dependencies to maintain portability and reproducibility.
    """

    @staticmethod
    def mean(values: List[float]) -> float:
        """Compute the mean of numeric values."""
        if not values:
            return 0.0
        return float(statistics.mean(values))

    @staticmethod
    def variance(values: List[float]) -> float:
        """Compute variance with safe handling of small samples."""
        if len(values) <= 1:
            return 0.0
        return float(statistics.variance(values))

    @staticmethod
    def minimum(values: List[float]) -> float:
        """Return minimum value."""
        if not values:
            return 0.0
        return float(min(values))

    @staticmethod
    def maximum(values: List[float]) -> float:
        """Return maximum value."""
        if not values:
            return 0.0
        return float(max(values))

    @staticmethod
    def trend(values: List[float]) -> float:
        """
        Estimate a linear trend slope using simple least squares.

        A positive slope indicates increasing signal intensity
        (potential worsening mental health indicator).

        A negative slope indicates decreasing signal intensity.
        """
        n = len(values)

        if n < 2:
            return 0.0

        x_vals = list(range(n))
        mean_x = statistics.mean(x_vals)
        mean_y = statistics.mean(values)

        numerator = 0.0
        denominator = 0.0

        for x, y in zip(x_vals, values):
            numerator += (x - mean_x) * (y - mean_y)
            denominator += (x - mean_x) ** 2

        if denominator == 0:
            return 0.0

        return numerator / denominator


class TemporalRiskAnalyzer:
    """
    Analyze temporal behavioral indicators and compute composite risk scores.

    This component translates sequences of psychological indicators into
    interpretable risk metrics used by early warning systems.
    """

    def __init__(self) -> None:
        """Initialize analyzer with configurable weights."""
        self.weights = {
            "depression": 1.0,
            "anxiety": 1.0,
            "panic": 1.0
        }

    def compute_risk_frequency(self, events: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute frequency of mental health indicators across the sequence.

        Returns normalized frequencies representing prevalence of
        psychological distress signals.
        """
        if not events:
            return {"depression": 0.0, "anxiety": 0.0, "panic": 0.0}

        counts = {"depression": 0, "anxiety": 0, "panic": 0}

        for event in events:
            if event.get("depression", 0):
                counts["depression"] += 1
            if event.get("anxiety", 0):
                counts["anxiety"] += 1
            if event.get("panic", 0):
                counts["panic"] += 1

        total = len(events)

        return {
            key: counts[key] / total
            for key in counts
        }

    def compute_composite_risk(self, freq: Dict[str, float]) -> float:
        """
        Compute a weighted composite mental health risk score.

        The formulation aggregates normalized frequencies of
        depression, anxiety, and panic indicators.
        """
        risk_score = 0.0

        for indicator, weight in self.weights.items():
            risk_score += freq.get(indicator, 0.0) * weight

        return risk_score / max(len(self.weights), 1)


class SequenceStrategy:
    """
    SequenceStrategy

    Responsible for summarizing temporal behavioral or academic indicators
    associated with student mental health monitoring.

    The strategy converts a sequence of student events into a structured
    summary containing statistical descriptors, trend signals, and
    composite risk indicators.

    Expected input format:

        events = [
            {
                "timestamp": "...",
                "cgpa": 3.2,
                "stress_score": 0.5,
                "depression": 0,
                "anxiety": 1,
                "panic": 0
            },
            ...
        ]

    Output summary example:

        {
            "event_count": 12,
            "cgpa_mean": 3.1,
            "cgpa_trend": -0.04,
            "stress_mean": 0.48,
            "stress_trend": 0.07,
            "risk_frequency": {...},
            "composite_risk": 0.42
        }

    These summaries can be consumed by machine learning pipelines or
    monitoring dashboards to detect emerging mental health concerns.
    """

    def __init__(self) -> None:
        """Initialize internal statistical utilities and analyzers."""
        self.stats = SequenceStatistics()
        self.risk_analyzer = TemporalRiskAnalyzer()

    def _extract_series(self, events: List[Dict[str, Any]], key: str) -> List[float]:
        """
        Extract a numerical time-series from event dictionaries.

        Missing values are safely ignored.
        """
        values: List[float] = []

        for event in events:
            value = event.get(key)

            if value is None:
                continue

            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue

        return values

    def _safe_recent_change(self, values: List[float]) -> float:
        """
        Compute difference between most recent and historical average.

        This metric captures abrupt behavioral changes.
        """
        if len(values) < 2:
            return 0.0

        historical = statistics.mean(values[:-1])
        recent = values[-1]

        return recent - historical

    def summarize(self, events: list) -> dict:
        """
        Summarize temporal behavioral or academic indicators.

        Parameters
        ----------
        events : list
            Chronologically ordered student events.

        Returns
        -------
        dict
            A structured summary describing statistical properties,
            trends, and risk indicators across the sequence.
        """

        if not isinstance(events, list):
            raise TypeError("events must be a list of event dictionaries")

        if len(events) == 0:
            return {
                "event_count": 0,
                "cgpa_mean": 0.0,
                "cgpa_trend": 0.0,
                "stress_mean": 0.0,
                "stress_trend": 0.0,
                "recent_stress_change": 0.0,
                "risk_frequency": {"depression": 0.0, "anxiety": 0.0, "panic": 0.0},
                "composite_risk": 0.0
            }

        cgpa_series = self._extract_series(events, "cgpa")
        stress_series = self._extract_series(events, "stress_score")

        cgpa_mean = self.stats.mean(cgpa_series)
        cgpa_trend = self.stats.trend(cgpa_series)

        stress_mean = self.stats.mean(stress_series)
        stress_trend = self.stats.trend(stress_series)

        stress_change = self._safe_recent_change(stress_series)

        risk_frequency = self.risk_analyzer.compute_risk_frequency(events)

        composite_risk = self.risk_analyzer.compute_composite_risk(
            risk_frequency
        )

        summary = {
            "event_count": len(events),
            "cgpa_mean": cgpa_mean,
            "cgpa_trend": cgpa_trend,
            "stress_mean": stress_mean,
            "stress_trend": stress_trend,
            "recent_stress_change": stress_change,
            "risk_frequency": risk_frequency,
            "composite_risk": composite_risk
        }

        return summary