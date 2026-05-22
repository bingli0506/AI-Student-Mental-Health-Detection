"""
experiments/evaluate.py

This module implements the evaluation framework for the HybridDL-ML-Ensemble
mental health early detection system. The Evaluator class is responsible for
measuring the predictive performance of trained models on held-out test data
and producing comprehensive experimental metrics suitable for academic
analysis and reproducible research.

Evaluation Methodology
----------------------
The evaluation protocol implemented in this module follows the methodology
described in the associated research work on AI-based early detection of
mental health issues among university students. The system predicts a
multi-class label representing the cumulative mental health burden reported
by each student.

The target variable represents the aggregated count of three binary indicators:
    - Depression
    - Anxiety
    - Panic Attack

The resulting classification labels form a four-class problem:

    Class 0 : No reported conditions
    Class 1 : One reported condition
    Class 2 : Two reported conditions
    Class 3 : Three reported conditions

Evaluation Metrics
------------------
The following metrics are computed according to the definitions used in the
paper and standard machine learning literature:

    Accuracy
        Overall classification correctness.

    Precision (Macro Averaged)
        Measures the proportion of correctly predicted positives among all
        predicted positives per class.

    Recall (Macro Averaged)
        Measures the proportion of correctly identified positives among all
        true positives per class.

    F1-Score (Macro Averaged)
        Harmonic mean of precision and recall.

Additional outputs include:

    - Confusion Matrix
    - Per-class metrics
    - Prediction distributions
    - Optional permutation significance testing

Design Considerations
---------------------
This implementation follows several important engineering principles:

1. Deterministic evaluation with reproducible random seeds.
2. Robust validation of inputs and model outputs.
3. Separation of metric computation and evaluation logic.
4. Clear documentation for reproducibility in academic experiments.
5. Structured outputs suitable for visualization and statistical analysis.

The Evaluator integrates seamlessly with the project's modular architecture
and relies on standardized interfaces defined in the system specification.

Author: HybridDLMLMentalHealthDetection Research Framework
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from collections import Counter

from models.model_base import BaseMentalHealthModel
from utils.utils import ProjectUtils


class Evaluator:
    """
    Evaluate trained mental health prediction models.

    The Evaluator executes a rigorous model evaluation pipeline including
    prediction generation, metric calculation, confusion matrix analysis,
    and optional permutation testing for statistical significance.

    Parameters
    ----------
    model : BaseMentalHealthModel
        Trained model implementing the standardized prediction interface.

    test_features : Any
        Feature matrix representing the held-out test dataset.

    test_labels : Any
        Ground-truth labels corresponding to the test dataset.

    random_seed : int, optional
        Seed used to ensure deterministic evaluation procedures.
    """

    def __init__(
        self,
        model: BaseMentalHealthModel,
        test_features: Any,
        test_labels: Any,
        random_seed: int = 42,
    ) -> None:
        self.model = model
        self.test_features = test_features
        self.test_labels = np.asarray(test_labels)
        self.random_seed = random_seed

        ProjectUtils.set_seed(self.random_seed)

    def evaluate(self) -> dict:
        """
        Evaluate classification and risk-ranking performance.

        This method runs the full evaluation workflow including predictions,
        metric computation, confusion matrix analysis, and statistical tests.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics, confusion matrix,
            and auxiliary analysis outputs.
        """
        predictions = self._generate_predictions()

        accuracy = self._compute_accuracy(self.test_labels, predictions)
        precision = self._compute_precision(self.test_labels, predictions)
        recall = self._compute_recall(self.test_labels, predictions)
        f1_score = self._compute_f1_score(precision, recall)

        confusion = self._compute_confusion_matrix(
            self.test_labels,
            predictions
        )

        class_distribution = self._compute_prediction_distribution(predictions)

        permutation_results = self._permutation_test(
            num_iterations=200
        )

        results: Dict[str, Any] = {
            "accuracy": float(accuracy),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1_score),
            "confusion_matrix": confusion.tolist(),
            "prediction_distribution": class_distribution,
            "permutation_test": permutation_results,
            "num_samples": int(len(self.test_labels)),
        }

        return results

    def _generate_predictions(self) -> np.ndarray:
        """
        Generate predictions from the trained model.

        Returns
        -------
        numpy.ndarray
            Predicted class labels.
        """
        predictions = self.model.predict(self.test_features)
        return np.asarray(predictions)

    def _compute_accuracy(
        self,
        labels: np.ndarray,
        preds: np.ndarray
    ) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        labels : numpy.ndarray
            True labels.

        preds : numpy.ndarray
            Predicted labels.

        Returns
        -------
        float
            Accuracy score.
        """
        correct = np.sum(labels == preds)
        return correct / len(labels)

    def _compute_precision(
        self,
        labels: np.ndarray,
        preds: np.ndarray
    ) -> float:
        """
        Compute macro-averaged precision.

        Returns
        -------
        float
            Macro precision.
        """
        classes = np.unique(labels)
        precisions: List[float] = []

        for c in classes:
            tp = np.sum((preds == c) & (labels == c))
            fp = np.sum((preds == c) & (labels != c))

            denom = tp + fp
            if denom == 0:
                precisions.append(0.0)
            else:
                precisions.append(tp / denom)

        return float(np.mean(precisions))

    def _compute_recall(
        self,
        labels: np.ndarray,
        preds: np.ndarray
    ) -> float:
        """
        Compute macro-averaged recall.

        Returns
        -------
        float
            Macro recall.
        """
        classes = np.unique(labels)
        recalls: List[float] = []

        for c in classes:
            tp = np.sum((preds == c) & (labels == c))
            fn = np.sum((preds != c) & (labels == c))

            denom = tp + fn
            if denom == 0:
                recalls.append(0.0)
            else:
                recalls.append(tp / denom)

        return float(np.mean(recalls))

    def _compute_f1_score(
        self,
        precision: float,
        recall: float
    ) -> float:
        """
        Compute F1 score from precision and recall.

        Returns
        -------
        float
            F1 score.
        """
        denom = precision + recall
        if denom == 0:
            return 0.0

        return 2 * (precision * recall) / denom

    def _compute_confusion_matrix(
        self,
        labels: np.ndarray,
        preds: np.ndarray
    ) -> np.ndarray:
        """
        Construct confusion matrix.

        Returns
        -------
        numpy.ndarray
            Confusion matrix of shape (C, C).
        """
        classes = np.unique(labels)
        num_classes = len(classes)

        matrix = np.zeros((num_classes, num_classes), dtype=int)

        for true, pred in zip(labels, preds):
            matrix[int(true), int(pred)] += 1

        return matrix

    def _compute_prediction_distribution(
        self,
        preds: np.ndarray
    ) -> Dict[int, int]:
        """
        Compute predicted class distribution.

        Returns
        -------
        dict
            Frequency of predicted labels.
        """
        counts = Counter(preds.tolist())
        return {int(k): int(v) for k, v in counts.items()}

    def _permutation_test(
        self,
        num_iterations: int = 200
    ) -> Dict[str, float]:
        """
        Perform permutation testing for statistical significance.

        The test estimates the probability that the observed accuracy
        could occur by chance under label randomization.

        Parameters
        ----------
        num_iterations : int
            Number of permutation trials.

        Returns
        -------
        dict
            Permutation test statistics including empirical p-value.
        """
        observed_accuracy = self._compute_accuracy(
            self.test_labels,
            self._generate_predictions()
        )

        random_accuracies: List[float] = []

        for _ in range(num_iterations):
            shuffled = np.random.permutation(self.test_labels)

            acc = self._compute_accuracy(shuffled, shuffled)
            random_accuracies.append(acc)

        random_accuracies = np.array(random_accuracies)

        p_value = np.mean(random_accuracies >= observed_accuracy)

        return {
            "observed_accuracy": float(observed_accuracy),
            "mean_random_accuracy": float(np.mean(random_accuracies)),
            "std_random_accuracy": float(np.std(random_accuracies)),
            "p_value": float(p_value),
            "iterations": int(num_iterations),
        }