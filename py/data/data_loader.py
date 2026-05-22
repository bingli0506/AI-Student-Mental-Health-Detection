"""
data_loader.py

This module implements data ingestion utilities for the HybridDL-ML-Ensemble
mental health early detection framework. It provides mechanisms to load,
validate, and prepare the Student Mental Health dataset used for predicting
early indicators of psychological distress among university students.

The dataset referenced in the research paper originates from a publicly
available Kaggle dataset titled "Student Mental Health". It contains survey
responses from university students and includes demographic attributes,
academic information, and self-reported psychological indicators.

Dataset Attributes (as described in the research paper):

Demographic:
- Age
- Gender
- Marital Status

Academic:
- Course
- Year of Study
- CGPA

Mental Health Indicators:
- Do you have Depression?
- Do you have Anxiety?
- Do you have Panic Attack?

Treatment:
- Did you seek any specialist for treatment?

The data loader implements the following capabilities:

1. Robust file loading with validation
2. Flexible CSV parsing and schema validation
3. Record standardization
4. Optional in-memory caching for repeated experiments
5. Train-test splitting utilities
6. PyTorch-compatible Dataset abstraction for model training
7. DataLoader construction for batch-based deep learning pipelines

The design prioritizes:
- Reproducibility
- Robust error handling
- Extensibility for future datasets
- Compatibility with PyTorch training workflows
- Research-grade documentation

Author: HybridDLMLMentalHealthDetection Project
"""

import os
import csv
import random
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader


class StudentMentalHealthDataset(Dataset):
    """
    PyTorch Dataset implementation for the Student Mental Health dataset.

    This class wraps structured student mental health records and provides
    an interface compatible with PyTorch's data loading pipeline.

    Each dataset item consists of:
        - feature vector (list or tensor)
        - label (mental health issue count or class)

    The dataset assumes preprocessing has already converted raw categorical
    features into numerical representations.

    Parameters
    ----------
    features : List[List[float]]
        Preprocessed feature vectors.
    labels : List[int]
        Target labels corresponding to each feature vector.
    device : Optional[str]
        Device identifier ("cpu" or "cuda") for tensor allocation.

    Notes
    -----
    The dataset object is intentionally lightweight and does not perform
    heavy preprocessing internally. All feature engineering steps are
    delegated to the MentalHealthPreprocessor module.
    """

    def __init__(
        self,
        features: List[List[float]],
        labels: List[int],
        device: Optional[str] = None
    ) -> None:

        if len(features) != len(labels):
            raise ValueError(
                "Features and labels must have the same number of samples."
            )

        self.features = features
        self.labels = labels
        self.device = device if device is not None else "cpu"

    def __len__(self) -> int:
        """
        Return the total number of samples.

        Returns
        -------
        int
            Number of records in the dataset.
        """
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample from the dataset.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Feature tensor and label tensor.
        """
        feature = torch.tensor(self.features[index], dtype=torch.float32).to(self.device)
        label = torch.tensor(self.labels[index], dtype=torch.long).to(self.device)

        return feature, label


class MentalHealthDataLoader:
    """
    Data loader for student mental health datasets.

    This class is responsible for reading survey records from disk,
    validating their structure, converting them into a consistent
    internal representation, and preparing them for downstream
    preprocessing and modeling stages.

    Core Responsibilities
    ---------------------
    - Load CSV survey data
    - Validate schema integrity
    - Normalize record structure
    - Provide reproducible dataset splitting
    - Support optional caching for faster repeated loading

    Example
    -------
    loader = MentalHealthDataLoader()

    records = loader.load("data/student_mental_health.csv")

    train_records, test_records = loader.split(records, test_size=0.2)
    """

    REQUIRED_COLUMNS = [
        "Age",
        "Gender",
        "Course",
        "Year of Study",
        "CGPA",
        "Marital Status",
        "Depression",
        "Anxiety",
        "Panic Attack"
    ]

    def __init__(self, enable_cache: bool = True) -> None:
        """
        Initialize the data loader.

        Parameters
        ----------
        enable_cache : bool
            If True, loaded datasets are cached in memory to avoid repeated
            disk reads during experiments.
        """
        self.enable_cache = enable_cache
        self._cache: Dict[str, List[Dict]] = {}

    def load(self, path: str) -> List[Dict]:
        """
        Load survey records from a CSV dataset.

        Parameters
        ----------
        path : str
            Path to the dataset file.

        Returns
        -------
        List[Dict]
            List of standardized student records.

        Raises
        ------
        FileNotFoundError
            If the dataset file does not exist.
        ValueError
            If required dataset columns are missing.
        """

        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        if self.enable_cache and path in self._cache:
            return self._cache[path]

        records: List[Dict] = []

        with open(path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            self._validate_columns(reader.fieldnames)

            for row in reader:
                record = self._normalize_record(row)
                records.append(record)

        if self.enable_cache:
            self._cache[path] = records

        return records

    def split(self, records: List[Dict], test_size: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
        """
        Split records into training and testing partitions.

        Parameters
        ----------
        records : list
            Loaded dataset records.
        test_size : float
            Fraction of data allocated to the test set.

        Returns
        -------
        tuple
            (train_records, test_records)
        """

        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        if len(records) == 0:
            raise ValueError("Cannot split an empty dataset.")

        shuffled = records.copy()
        random.shuffle(shuffled)

        split_index = int(len(shuffled) * (1 - test_size))

        train_records = shuffled[:split_index]
        test_records = shuffled[split_index:]

        return train_records, test_records

    def _validate_columns(self, columns: Optional[List[str]]) -> None:
        """
        Validate that required dataset columns are present.

        Parameters
        ----------
        columns : list
            List of column names from the dataset header.

        Raises
        ------
        ValueError
            If required columns are missing.
        """

        if columns is None:
            raise ValueError("Dataset appears to have no header.")

        missing = [col for col in self.REQUIRED_COLUMNS if col not in columns]

        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {missing}"
            )

    def _normalize_record(self, row: Dict) -> Dict:
        """
        Normalize a raw CSV row into a standardized record.

        Parameters
        ----------
        row : dict
            Raw CSV dictionary.

        Returns
        -------
        dict
            Cleaned record dictionary.
        """

        def safe_int(value):
            try:
                return int(value)
            except Exception:
                return None

        def safe_float(value):
            try:
                return float(value)
            except Exception:
                return None

        record = {
            "age": safe_int(row.get("Age")),
            "gender": row.get("Gender"),
            "course": row.get("Course"),
            "year_of_study": row.get("Year of Study"),
            "cgpa": row.get("CGPA"),
            "marital_status": row.get("Marital Status"),
            "depression": row.get("Depression"),
            "anxiety": row.get("Anxiety"),
            "panic_attack": row.get("Panic Attack")
        }

        return record


def create_pytorch_dataloader(
    features: List[List[float]],
    labels: List[int],
    batch_size: int = 32,
    shuffle: bool = True,
    device: Optional[str] = None
) -> DataLoader:
    """
    Utility function to create a PyTorch DataLoader.

    Parameters
    ----------
    features : list
        Feature matrix.
    labels : list
        Label vector.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle samples each epoch.
    device : str
        Target computation device.

    Returns
    -------
    torch.utils.data.DataLoader
        Configured DataLoader instance.
    """

    dataset = StudentMentalHealthDataset(
        features=features,
        labels=labels,
        device=device
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False
    )

    return loader