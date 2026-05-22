"""
main.py

Entry point for the HybridDLMLMentalHealthDetection research framework.

This script orchestrates the complete experimental pipeline for AI-based early
detection of mental health issues among university students. The pipeline
implements the HybridDL-ML-Ensemble workflow described in the associated
research study, combining classical machine learning models, deep learning
representations, and stacked meta-learning for predictive analytics.

The main responsibilities of this entry script include:

1. Parsing command-line arguments for dataset paths, configuration files,
   output directories, execution modes, and hyperparameters.
2. Loading and validating configuration settings through ProjectConfig.
3. Initializing reproducibility controls (e.g., random seeds).
4. Loading and preprocessing student mental health datasets.
5. Training machine learning and deep learning models using the Trainer module.
6. Evaluating model performance through the Evaluator module.
7. Producing visualization artifacts for experimental analysis.
8. Demonstrating usage of auxiliary research components including:
   - feature encoders
   - representation alignment models
   - similar case retrieval modules
   - strategy modules for relevance scoring and behavioral summarization

The script is designed to satisfy academic reproducibility standards and
supports both training and inference workflows.

Example usage:

    python main.py \
        --input data/student_mental_health.csv \
        --output_dir results/ \
        --mode train \
        --config config.py \
        --seed 42

The code follows best practices including structured logging, argument
validation, modular design, and comprehensive error handling.
"""

import argparse
import logging
import os
import sys
from typing import Dict, Tuple, Any

from config import ProjectConfig

from data.data_loader import MentalHealthDataLoader
from data.data_preprocess import MentalHealthPreprocessor

from models.model_base import BaseMentalHealthModel
from models.model_encoder import StudentFeatureEncoder
from models.model_alignment import RiskAlignmentModel
from models.model_retrieval import SimilarCaseRetriever

from strategies.strategy_relevance import RelevanceStrategy
from strategies.strategy_intent import IntentStrategy
from strategies.strategy_sequence import SequenceStrategy

from experiments.train import Trainer
from experiments.evaluate import Evaluator
from experiments.visualization import ResultVisualizer

from utils.utils import ProjectUtils


class Application:
    """
    Main application class responsible for orchestrating the end-to-end
    mental health early detection pipeline.

    This class coordinates data ingestion, preprocessing, model training,
    evaluation, and result visualization.

    The design emphasizes modularity to allow researchers to extend
    components independently (e.g., new model architectures or strategies).
    """

    def __init__(self) -> None:
        """Initialize application state and logger."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, config: dict) -> None:
        """
        Run the end-to-end mental health early detection pipeline.

        Parameters
        ----------
        config : dict
            Dictionary containing runtime configuration including dataset
            paths, model hyperparameters, experiment settings, and output
            directories.

        Raises
        ------
        RuntimeError
            If critical pipeline components fail during execution.
        """
        try:
            self.logger.info("Starting HybridDL-ML mental health detection pipeline.")

            dataset_path: str = config["input_path"]
            output_dir: str = config["output_dir"]
            mode: str = config["mode"]

            os.makedirs(output_dir, exist_ok=True)

            # ------------------------------------------------------------------
            # Step 1: Initialize data pipeline
            # ------------------------------------------------------------------
            self.logger.info("Initializing data loader and preprocessor.")

            data_loader = MentalHealthDataLoader()
            preprocessor = MentalHealthPreprocessor()

            records = data_loader.load(dataset_path)
            self.logger.info("Loaded %d records from dataset.", len(records))

            train_records, test_records = data_loader.split(records)
            self.logger.info(
                "Dataset split into train (%d) and test (%d) records.",
                len(train_records),
                len(test_records),
            )

            # ------------------------------------------------------------------
            # Step 2: Preprocess data
            # ------------------------------------------------------------------
            self.logger.info("Running preprocessing and feature engineering.")

            train_features, train_labels = preprocessor.fit_transform(train_records)
            test_features, test_labels = preprocessor.transform(test_records)

            # ------------------------------------------------------------------
            # Step 3: Feature encoding and representation alignment
            # ------------------------------------------------------------------
            encoder = StudentFeatureEncoder()
            alignment_model = RiskAlignmentModel()

            encoded_train = encoder.encode(train_features)
            encoded_test = encoder.encode(test_features)

            alignment_model.align(encoded_train, train_labels)

            # ------------------------------------------------------------------
            # Step 4: Strategy components (research experimentation layer)
            # ------------------------------------------------------------------
            relevance_strategy = RelevanceStrategy()
            intent_strategy = IntentStrategy()
            sequence_strategy = SequenceStrategy()

            if len(train_features) > 0:
                relevance_score = relevance_strategy.score(train_features[0])
                self.logger.debug("Sample relevance score: %s", str(relevance_score))

            # Demonstrate intent inference
            intent_strategy.infer("I feel stressed about exams and coursework.")

            # Demonstrate behavioral sequence summarization
            sequence_strategy.summarize(["low_cgpa", "missed_classes", "stress_report"])

            # ------------------------------------------------------------------
            # Step 5: Model training
            # ------------------------------------------------------------------
            trainer = Trainer()

            if mode == "train":
                self.logger.info("Training mental health detection model.")
                training_results = trainer.train()

                self.logger.info("Training completed successfully.")
                self.logger.debug("Training results: %s", str(training_results))

            # ------------------------------------------------------------------
            # Step 6: Model evaluation
            # ------------------------------------------------------------------
            evaluator = Evaluator()

            self.logger.info("Evaluating trained model.")
            metrics = evaluator.evaluate()

            self.logger.info("Evaluation metrics: %s", str(metrics))

            # ------------------------------------------------------------------
            # Step 7: Similar case retrieval demonstration
            # ------------------------------------------------------------------
            retriever = SimilarCaseRetriever()

            if len(encoded_test) > 0:
                retrieved_cases = retriever.retrieve(encoded_test[0])
                self.logger.debug(
                    "Retrieved %d similar historical cases.", len(retrieved_cases)
                )

            # ------------------------------------------------------------------
            # Step 8: Visualization and artifact generation
            # ------------------------------------------------------------------
            visualizer = ResultVisualizer()

            metrics_plot_path = os.path.join(output_dir, "metrics.png")
            visualizer.plot_metrics(metrics, metrics_plot_path)

            self.logger.info("Saved evaluation visualization to %s", metrics_plot_path)

            self.logger.info("Pipeline execution finished successfully.")

        except Exception as exc:
            self.logger.exception("Pipeline execution failed due to an error.")
            raise RuntimeError("Application execution failed.") from exc


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Construct command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with all supported runtime arguments.
    """
    parser = argparse.ArgumentParser(
        description="HybridDL-ML Ensemble for Student Mental Health Detection"
    )

    parser.add_argument(
        "--input",
        dest="input_path",
        type=str,
        required=True,
        help="Path to the input dataset file.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory where results and artifacts will be stored.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to external configuration file.",
    )

    parser.add_argument(
        "--mode",
        choices=["train", "inference"],
        default="train",
        help="Execution mode: train or inference.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate hyperparameter.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )

    return parser


def configure_logging() -> None:
    """
    Configure application-wide logging.

    Logging configuration ensures consistent formatting and timestamps
    for debugging, reproducibility, and experiment traceability.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Load runtime configuration by combining CLI arguments and ProjectConfig.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    Dict[str, Any]
        Final merged configuration dictionary.
    """
    base_config = ProjectConfig().as_dict()

    runtime_config: Dict[str, Any] = dict(base_config)

    runtime_config.update(
        {
            "input_path": args.input_path,
            "output_dir": args.output_dir,
            "mode": args.mode,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        }
    )

    return runtime_config


def main() -> None:
    """
    Program entry point.

    This function initializes configuration, logging, reproducibility
    controls, and then launches the Application pipeline.
    """
    configure_logging()

    parser = build_argument_parser()
    args = parser.parse_args()

    config = load_configuration(args)

    ProjectUtils.set_seed(config["seed"])

    app = Application()
    app.run(config)


if __name__ == "__main__":
    main()