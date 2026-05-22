"""
data_preprocess.py

This module implements the complete preprocessing pipeline for the
HybridDL-ML-Ensemble mental health early detection system described in the
research paper "The Application of AI-Based Early Detection of Mental Health
Issues Among University Students in Academic Settings".

The preprocessing pipeline transforms raw student survey records into
model-ready numerical feature matrices suitable for classical machine
learning models (Random Forest, XGBoost) and deep learning architectures
(MLP and CNN-1D).

Pipeline Overview
-----------------
The preprocessing workflow follows the methodology described in the paper
and includes the following stages:

1. Data Cleaning
   - Validation of record structure
   - Standardization of categorical values
   - Removal of non-informative attributes (e.g., timestamp)

2. Feature Transformation
   - Conversion of binary responses ("Yes"/"No") to numeric values (1/0)
   - Transformation of CGPA ranges into numerical midpoints
   - Construction of a composite mental health severity score

3. Feature Encoding
   - One-hot encoding of categorical demographic and academic attributes
   - Handling of unseen categories during inference

4. Normalization
   - Min-Max normalization applied to continuous features such as age
     and CGPA to stabilize neural network training
   - Scaling parameters learned from training data only

5. Dataset Preparation
   - Construction of feature matrix (X)
   - Construction of target label vector (y)

Design Principles
-----------------
The implementation follows strict research reproducibility and engineering
best practices:

- All transformation parameters are learned during `fit_transform`
  and reused during `transform`.
- Data leakage is prevented by fitting encoders and scalers exclusively
  on training data.
- Extensive validation ensures input data integrity.
- The design supports extensibility for additional features or datasets.

Expected Input Record Structure
-------------------------------
Each record should contain the following attributes (or equivalent):

{
    "Age": int,
    "Gender": str,
    "Marital Status": str,
    "Course": str,
    "Year of Study": str,
    "CGPA": str,                  # e.g. "3.00 - 3.49"
    "Do you have Depression?": str,
    "Do you have Anxiety?": str,
    "Do you have Panic attack?": str,
    "Did you seek any specialist for treatment?": str
}

Output
------
The preprocessing module returns:

X : numpy.ndarray
    Feature matrix for model input

y : numpy.ndarray
    Target variable representing the total number of mental health issues
    reported by each student (range 0–3).

This design ensures compatibility with both traditional ML models and
deep neural architectures used in the ensemble framework.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import re


class MentalHealthPreprocessor:
    """
    Preprocessing pipeline for the Student Mental Health dataset.

    This class implements a full preprocessing workflow including:

    - Data validation
    - Binary response conversion
    - CGPA range transformation
    - One-hot encoding for categorical variables
    - Min-Max normalization for numerical features
    - Target variable creation

    The class follows the scikit-learn style interface with
    `fit_transform()` for training data and `transform()` for inference.

    Attributes
    ----------
    categorical_maps : Dict[str, Dict[str, int]]
        Mapping dictionaries for categorical encoding.

    feature_index : Dict[str, int]
        Index mapping for final feature ordering.

    min_vals : Dict[str, float]
        Minimum values used for Min-Max normalization.

    max_vals : Dict[str, float]
        Maximum values used for Min-Max normalization.

    fitted : bool
        Indicates whether the preprocessor has been fitted.
    """

    def __init__(self) -> None:
        """Initialize preprocessing state."""
        self.categorical_maps: Dict[str, Dict[str, int]] = {}
        self.feature_index: Dict[str, int] = {}

        self.min_vals: Dict[str, float] = {}
        self.max_vals: Dict[str, float] = {}

        self.fitted: bool = False

        self.categorical_columns = [
            "Gender",
            "Marital Status",
            "Course",
            "Year of Study",
            "Did you seek any specialist for treatment?"
        ]

        self.binary_columns = [
            "Do you have Depression?",
            "Do you have Anxiety?",
            "Do you have Panic attack?"
        ]

        self.numeric_columns = ["Age", "CGPA"]

    def fit_transform(self, records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessing parameters and transform the dataset.

        Parameters
        ----------
        records : List[Dict[str, Any]]
            Raw survey records.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Feature matrix X and target labels y.
        """

        validated = self._validate_records(records)

        processed_rows = []
        labels = []

        for record in validated:
            cleaned = self._clean_record(record)

            binary_values = self._convert_binary(cleaned)

            cgpa_value = self._convert_cgpa(cleaned.get("CGPA"))

            age_value = self._safe_float(cleaned.get("Age"))

            mental_total = sum(binary_values.values())

            labels.append(mental_total)

            processed_rows.append({
                "Age": age_value,
                "CGPA": cgpa_value,
                **binary_values,
                **{col: cleaned.get(col, "Unknown") for col in self.categorical_columns}
            })

        self._fit_categorical_maps(processed_rows)
        self._fit_scalers(processed_rows)

        feature_matrix = [self._encode_row(row) for row in processed_rows]

        self.fitted = True

        return np.array(feature_matrix, dtype=float), np.array(labels, dtype=int)

    def transform(self, records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply learned preprocessing to new records.

        Parameters
        ----------
        records : List[Dict[str, Any]]
            New dataset records.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Transformed feature matrix and labels.
        """

        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        validated = self._validate_records(records)

        processed_rows = []
        labels = []

        for record in validated:
            cleaned = self._clean_record(record)

            binary_values = self._convert_binary(cleaned)

            cgpa_value = self._convert_cgpa(cleaned.get("CGPA"))

            age_value = self._safe_float(cleaned.get("Age"))

            mental_total = sum(binary_values.values())

            labels.append(mental_total)

            processed_rows.append({
                "Age": age_value,
                "CGPA": cgpa_value,
                **binary_values,
                **{col: cleaned.get(col, "Unknown") for col in self.categorical_columns}
            })

        feature_matrix = [self._encode_row(row) for row in processed_rows]

        return np.array(feature_matrix, dtype=float), np.array(labels, dtype=int)

    def _validate_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate input records and ensure proper structure."""

        if not isinstance(records, list):
            raise TypeError("Records must be provided as a list.")

        validated = []

        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Each record must be a dictionary.")

            validated.append(record)

        if len(validated) == 0:
            raise ValueError("Dataset cannot be empty.")

        return validated

    def _clean_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize record values and remove irrelevant fields."""

        cleaned = {}

        for key, value in record.items():
            if key.lower() == "timestamp":
                continue

            if isinstance(value, str):
                cleaned[key] = value.strip()
            else:
                cleaned[key] = value

        return cleaned

    def _convert_binary(self, record: Dict[str, Any]) -> Dict[str, int]:
        """Convert Yes/No survey responses to numeric binary values."""

        result = {}

        for col in self.binary_columns:
            val = str(record.get(col, "")).lower()

            if "yes" in val:
                result[col] = 1
            elif "no" in val:
                result[col] = 0
            else:
                result[col] = 0

        return result

    def _convert_cgpa(self, cgpa_range: Any) -> float:
        """
        Convert CGPA range strings into numerical midpoints.

        Example:
        "3.00 - 3.49" -> 3.245
        """

        if cgpa_range is None:
            return 0.0

        text = str(cgpa_range)

        numbers = re.findall(r"\d+\.\d+", text)

        if len(numbers) == 2:
            low = float(numbers[0])
            high = float(numbers[1])
            return (low + high) / 2.0

        try:
            return float(numbers[0])
        except Exception:
            return 0.0

    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float."""

        try:
            return float(value)
        except Exception:
            return 0.0

    def _fit_categorical_maps(self, rows: List[Dict[str, Any]]) -> None:
        """Create mapping tables for categorical variables."""

        for column in self.categorical_columns:
            values = sorted({str(row[column]) for row in rows})
            mapping = {v: i for i, v in enumerate(values)}
            self.categorical_maps[column] = mapping

    def _fit_scalers(self, rows: List[Dict[str, Any]]) -> None:
        """Compute Min-Max normalization parameters."""

        for column in self.numeric_columns:
            values = [float(row[column]) for row in rows]

            self.min_vals[column] = float(np.min(values))
            self.max_vals[column] = float(np.max(values))

    def _normalize(self, column: str, value: float) -> float:
        """Apply Min-Max normalization."""

        min_val = self.min_vals.get(column, 0.0)
        max_val = self.max_vals.get(column, 1.0)

        if max_val == min_val:
            return 0.0

        return (value - min_val) / (max_val - min_val)

    def _encode_row(self, row: Dict[str, Any]) -> List[float]:
        """Encode a single processed row into a feature vector."""

        features: List[float] = []

        age_norm = self._normalize("Age", float(row["Age"]))
        cgpa_norm = self._normalize("CGPA", float(row["CGPA"]))

        features.append(age_norm)
        features.append(cgpa_norm)

        for col in self.binary_columns:
            features.append(float(row[col]))

        for column in self.categorical_columns:
            mapping = self.categorical_maps.get(column, {})
            vector = [0.0] * len(mapping)

            idx = mapping.get(str(row[column]))

            if idx is not None:
                vector[idx] = 1.0

            features.extend(vector)

        return features