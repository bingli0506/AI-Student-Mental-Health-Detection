"""
strategy_intent.py

Intent Strategy Module for HybridDL-ML Mental Health Detection System.

This module implements the IntentStrategy class responsible for inferring
psychological intent and distress signals from textual student expressions.
Within the overall architecture, this strategy provides a lightweight natural
language inference layer that extracts interpretable mental health indicators
from free-form student text such as survey comments, counseling notes, or
behavioral self-reports.

The intent inference mechanism is designed to complement structured academic
and demographic signals used by the HybridDL-ML-Ensemble model described in
the research framework. While the primary predictive models operate on tabular
features, this module extracts auxiliary semantic cues from text that may
reflect latent emotional states.

Methodological Overview
-----------------------

The implemented intent recognition strategy follows a rule-enhanced lexical
inference approach inspired by early-stage clinical text mining and mental
health monitoring literature. The approach includes the following stages:

1. Text Normalization
   Raw student text is standardized through lowercasing, punctuation removal,
   and whitespace normalization to reduce noise.

2. Distress Cue Detection
   Lexical dictionaries capture key expressions associated with depression,
   anxiety, panic attacks, and help-seeking behavior. These dictionaries are
   constructed from common self-reported phrases observed in student mental
   health surveys.

3. Weighted Intent Scoring
   Each detected cue contributes to an intent-specific score. Scores are
   normalized to approximate probability-like confidence values.

4. Sentiment Approximation
   A lightweight sentiment polarity estimator evaluates positive versus
   negative emotional tone.

5. Final Intent Classification
   The dominant intent is selected based on the highest aggregated distress
   score.

Design Rationale
----------------

The design intentionally avoids heavy NLP dependencies to ensure reproducibility
and deployability within academic environments where computational resources
may be limited. Although neural language models could provide richer semantic
representations, lexicon-based inference offers:

- High interpretability
- Deterministic reproducibility
- Minimal computational cost
- Easy extensibility for domain-specific expressions

The output structure is designed to support downstream integration with
relevance scoring, risk alignment models, and similar-case retrieval modules.

Expected Output Format
----------------------

The infer() method returns a structured dictionary:

{
    "intent": str,
    "confidence": float,
    "scores": dict,
    "sentiment": float,
    "distress_cues": list
}

Where:
- intent: predicted psychological intent category
- confidence: normalized score for predicted intent
- scores: scores for each mental health category
- sentiment: approximate sentiment polarity
- distress_cues: detected textual indicators

This design supports interpretability and auditability, which are essential
for ethical mental health monitoring systems.
"""

import re
from typing import Dict, List, Tuple


class IntentStrategy:
    """
    Strategy for inferring psychological intent and distress cues from text.

    The strategy implements a deterministic rule-based classifier that maps
    textual expressions to mental health risk categories. The system detects
    linguistic signals associated with psychological distress and computes
    interpretable intent scores.

    The model focuses on four primary categories aligned with the mental
    health prediction task:

    - depression
    - anxiety
    - panic
    - help_seeking
    - neutral

    The classifier returns a structured representation containing scores,
    sentiment polarity, and matched distress cues.
    """

    def __init__(self) -> None:
        """
        Initialize lexical dictionaries and scoring parameters.

        Dictionaries are curated to capture typical language used by
        university students when describing mental health challenges.
        """

        self.depression_keywords: List[str] = [
            "sad", "hopeless", "empty", "tired", "worthless",
            "no motivation", "lonely", "depressed", "exhausted",
            "can't focus", "no energy", "unhappy"
        ]

        self.anxiety_keywords: List[str] = [
            "anxious", "worried", "overthinking", "stress",
            "nervous", "tense", "restless", "panic",
            "fear", "pressure", "overwhelmed"
        ]

        self.panic_keywords: List[str] = [
            "panic attack", "heart racing", "can't breathe",
            "shaking", "sweating", "dizzy", "losing control"
        ]

        self.help_keywords: List[str] = [
            "need help", "talk to someone", "counseling",
            "therapy", "support", "doctor", "psychologist",
            "mental health service"
        ]

        self.positive_keywords: List[str] = [
            "happy", "good", "motivated", "confident",
            "excited", "relaxed", "hopeful"
        ]

        self.negative_keywords: List[str] = [
            "bad", "terrible", "awful", "stressed",
            "tired", "sad", "hopeless"
        ]

    def _normalize_text(self, text: str) -> str:
        """
        Normalize input text by removing punctuation and standardizing format.

        Parameters
        ----------
        text : str
            Raw textual input.

        Returns
        -------
        str
            Cleaned text suitable for lexical matching.
        """

        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _count_matches(self, text: str, keywords: List[str]) -> Tuple[int, List[str]]:
        """
        Count keyword matches within text.

        Parameters
        ----------
        text : str
            Normalized input text.
        keywords : List[str]
            Lexical cues associated with a specific mental health condition.

        Returns
        -------
        Tuple[int, List[str]]
            Number of matches and list of detected cues.
        """

        matches = []
        for keyword in keywords:
            if keyword in text:
                matches.append(keyword)

        return len(matches), matches

    def _compute_sentiment(self, text: str) -> float:
        """
        Estimate simple sentiment polarity score.

        Sentiment is approximated using positive and negative lexical counts.

        Parameters
        ----------
        text : str
            Normalized text.

        Returns
        -------
        float
            Sentiment score in range [-1, 1].
        """

        pos_count, _ = self._count_matches(text, self.positive_keywords)
        neg_count, _ = self._count_matches(text, self.negative_keywords)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return (pos_count - neg_count) / float(total)

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize scores to produce probability-like values.

        Parameters
        ----------
        scores : Dict[str, float]
            Raw intent scores.

        Returns
        -------
        Dict[str, float]
            Normalized scores.
        """

        total = sum(scores.values())

        if total == 0:
            return {k: 0.0 for k in scores}

        return {k: v / total for k, v in scores.items()}

    def infer(self, text: str) -> Dict[str, object]:
        """
        Infer psychological intent and distress signals from student text.

        Parameters
        ----------
        text : str
            Free-form textual expression from a student (e.g., survey comment).

        Returns
        -------
        dict
            Structured inference result containing predicted intent,
            normalized scores, sentiment estimate, and detected cues.
        """

        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Input text must be a non-empty string.")

        normalized_text = self._normalize_text(text)

        depression_count, dep_cues = self._count_matches(
            normalized_text, self.depression_keywords
        )

        anxiety_count, anx_cues = self._count_matches(
            normalized_text, self.anxiety_keywords
        )

        panic_count, panic_cues = self._count_matches(
            normalized_text, self.panic_keywords
        )

        help_count, help_cues = self._count_matches(
            normalized_text, self.help_keywords
        )

        raw_scores: Dict[str, float] = {
            "depression": float(depression_count),
            "anxiety": float(anxiety_count),
            "panic": float(panic_count),
            "help_seeking": float(help_count),
            "neutral": 1.0
        }

        normalized_scores = self._normalize_scores(raw_scores)

        predicted_intent = max(normalized_scores, key=normalized_scores.get)
        confidence = normalized_scores[predicted_intent]

        sentiment_score = self._compute_sentiment(normalized_text)

        distress_cues = list(set(dep_cues + anx_cues + panic_cues + help_cues))

        result: Dict[str, object] = {
            "intent": predicted_intent,
            "confidence": float(confidence),
            "scores": normalized_scores,
            "sentiment": float(sentiment_score),
            "distress_cues": distress_cues
        }

        return result