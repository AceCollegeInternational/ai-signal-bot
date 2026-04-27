import os
import json
import joblib
from typing import Dict, Any, Optional
from utils.logger import get_logger

log = get_logger(__name__)


class MLValidator:
    """
    Second-opinion gatekeeper using pre-trained ML models.
    Filters out low-probability trades based on historical performance.
    """

    def __init__(self, model_path: str = "models/rf_v1.joblib"):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the pre-trained model if it exists."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                log.info(f"ML Model loaded from {self.model_path}")
            except Exception as e:
                log.error(f"Failed to load ML model: {e}")
        else:
            log.debug(
                f"No ML model found at {self.model_path}. ML validation disabled."
            )

    def predict_success_prob(self, features: Dict[str, float]) -> float:
        """
        Return the probability of a trade being successful (0.0 to 1.0).
        Returns 1.0 (pass-through) if no model is loaded.
        """
        if self.model is None or not features:
            return 1.0

        try:
            # Note: This is a placeholder for actual feature engineering/alignment
            # In production, we'd ensure feature names match the model training set
            import pandas as pd

            df = pd.DataFrame([features])

            # Placeholder for model.predict_proba
            # prob = self.model.predict_proba(df)[0][1] # Probability of Class 1 (Win)

            # For prototype, we skip if model format is unknown
            return 1.0
        except Exception as e:
            log.error(f"ML Prediction error: {e}")
            return 1.0

    def is_approved(self, features: Dict[str, float], threshold: float = 0.6) -> bool:
        """Check if signal passes the ML win-probability threshold."""
        prob = self.predict_success_prob(features)
        if prob < threshold:
            log.info(f"ML REJECTED: Win probability {prob:.2f} < {threshold}")
            return False
        return True
