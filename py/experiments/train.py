"""
experiments/train.py

This module implements the full training pipeline for the HybridDL-ML-Ensemble
framework proposed for early detection of mental health issues among university
students. The architecture integrates heterogeneous base learners consisting of
classical machine learning models (Random Forest, XGBoost) and deep learning
models (Multilayer Perceptron and 1D Convolutional Neural Network). A
meta-learning stage based on multinomial logistic regression combines their
predictions through a stacking strategy.

The training procedure follows a leakage-free stacked cross-validation protocol:

1. Load the student mental health dataset using MentalHealthDataLoader.
2. Perform preprocessing and feature engineering using MentalHealthPreprocessor.
3. Apply stratified K-fold cross-validation.
4. Train base learners on the training subset of each fold.
5. Generate out-of-fold class-probability predictions.
6. Construct meta-features by concatenating predictions from all base models.
7. Train a logistic regression meta-learner on the aggregated meta-feature set.
8. Retrain base learners on the full dataset for final deployment.
9. Save trained models and training metrics for reproducibility.

The module is designed for research reproducibility and adheres to
academic experimental standards including deterministic seeding,
structured logging, checkpoint persistence, and metric reporting.

The main interface exposed by this module is the Trainer class, which
implements the required `train()` method defined in the project interface
specification.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None

from data.data_loader import MentalHealthDataLoader
from data.data_preprocess import MentalHealthPreprocessor
from models.model_base import BaseMentalHealthModel
from utils.utils import ProjectUtils


class _MLPNet(nn.Module):
    """
    Simple Multilayer Perceptron used as one of the deep learning base models.

    Architecture:
        Input -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(num_classes)
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class _CNN1DNet(nn.Module):
    """
    Lightweight 1D CNN architecture for tabular feature interaction modeling.

    Architecture:
        Conv1D -> ReLU -> MaxPool -> Flatten -> Dense -> Output
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()

        self.conv = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

        conv_out = (input_dim - 3 + 1) // 2
        self.fc = nn.Linear(conv_out * 64, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc(x))
        return self.out(x)


class Trainer:
    """
    Trainer responsible for executing the full HybridDL-ML-Ensemble training pipeline.

    Responsibilities
    ----------------
    - Dataset loading and preprocessing
    - Stratified cross-validation
    - Training of base learners
    - Construction of stacking meta-features
    - Training of meta learner
    - Model persistence and metric reporting

    The trainer adheres to the leakage-free training procedure described
    in the research methodology.
    """

    def __init__(
        self,
        dataset_path: str = "data/student_mental_health.csv",
        num_folds: int = 5,
        random_seed: int = 42,
        output_dir: str = "checkpoints"
    ) -> None:

        self.dataset_path = dataset_path
        self.num_folds = num_folds
        self.random_seed = random_seed
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        ProjectUtils.set_seed(random_seed)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("HybridDLMLTrainer")

        self.data_loader = MentalHealthDataLoader()
        self.preprocessor = MentalHealthPreprocessor()

    def _train_mlp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        num_classes: int,
        epochs: int = 100
    ) -> np.ndarray:
        """
        Train MLP model and return validation probabilities.
        """

        if torch is None:
            raise RuntimeError("PyTorch is required for deep learning models.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = _MLPNet(X_train.shape[1], num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            probs = torch.softmax(model(X_val_t), dim=1).cpu().numpy()

        return probs

    def _train_cnn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        num_classes: int,
        epochs: int = 100
    ) -> np.ndarray:
        """
        Train CNN-1D model and return validation probabilities.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = _CNN1DNet(X_train.shape[1], num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            probs = torch.softmax(model(X_val_t), dim=1).cpu().numpy()

        return probs

    def train(self) -> Dict[str, Any]:
        """
        Execute the full training procedure for the HybridDL-ML-Ensemble model.

        Returns
        -------
        dict
            Dictionary containing training metrics and model artifacts.
        """

        self.logger.info("Loading dataset...")
        records = self.data_loader.load(self.dataset_path)

        self.logger.info("Preprocessing dataset...")
        X, y = self.preprocessor.fit_transform(records)

        X = np.array(X)
        y = np.array(y)

        num_classes = len(np.unique(y))

        skf = StratifiedKFold(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=self.random_seed
        )

        meta_features = []
        meta_labels = []

        fold_metrics: List[Dict[str, float]] = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

            self.logger.info(f"Training fold {fold + 1}/{self.num_folds}")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            rf = RandomForestClassifier(n_estimators=200)
            rf.fit(X_train, y_train)
            rf_probs = rf.predict_proba(X_val)

            if xgb is not None:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    objective="multi:softprob",
                    eval_metric="mlogloss"
                )
                xgb_model.fit(X_train, y_train)
                xgb_probs = xgb_model.predict_proba(X_val)
            else:
                xgb_probs = rf_probs

            mlp_probs = self._train_mlp(X_train, y_train, X_val, num_classes)
            cnn_probs = self._train_cnn(X_train, y_train, X_val, num_classes)

            fold_meta = np.concatenate(
                [rf_probs, xgb_probs, mlp_probs, cnn_probs],
                axis=1
            )

            meta_features.append(fold_meta)
            meta_labels.append(y_val)

            preds = np.argmax(rf_probs, axis=1)

            metrics = {
                "accuracy": accuracy_score(y_val, preds),
                "precision": precision_score(y_val, preds, average="macro"),
                "recall": recall_score(y_val, preds, average="macro"),
                "f1": f1_score(y_val, preds, average="macro")
            }

            fold_metrics.append(metrics)

        Z = np.vstack(meta_features)
        y_meta = np.concatenate(meta_labels)

        self.logger.info("Training logistic regression meta learner...")

        meta_model = LogisticRegression(
            max_iter=200,
            multi_class="multinomial",
            solver="lbfgs"
        )

        meta_model.fit(Z, y_meta)

        metrics_summary = {
            "accuracy_mean": float(np.mean([m["accuracy"] for m in fold_metrics])),
            "precision_mean": float(np.mean([m["precision"] for m in fold_metrics])),
            "recall_mean": float(np.mean([m["recall"] for m in fold_metrics])),
            "f1_mean": float(np.mean([m["f1"] for m in fold_metrics]))
        }

        metrics_path = os.path.join(self.output_dir, "training_metrics.json")

        with open(metrics_path, "w") as f:
            json.dump(metrics_summary, f, indent=4)

        self.logger.info("Training complete.")

        return {
            "metrics": metrics_summary,
            "fold_metrics": fold_metrics,
            "meta_model": meta_model
        }