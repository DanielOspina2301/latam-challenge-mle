from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from typing import Tuple, Union, List

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from challenge.infrastructure.preprocessing import Preprocessor

class DelayModel:

    def __init__(
        self,
        top_features: List[str] = None,
        threshold_in_minutes: int = 15,
    ):
        self._model = None
        self.preprocessor = Preprocessor()
        self.top_10_features = top_features
        self.threshold_in_minutes = threshold_in_minutes

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        try:
            data["period_day"] = data["Fecha-I"].apply(self.preprocessor.get_period_day)
            data["high_season"] = data["Fecha-I"].apply(self.preprocessor.is_high_season)
            data["min_diff"] = data.apply(self.preprocessor.get_min_diff, axis=1)
            data["delay"] = np.where(data["min_diff"] > self.threshold_in_minutes, 1, 0)

            features = pd.concat([
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES")],
                axis=1
            )
        except Exception as e:
            raise RuntimeError(f"Error in preprocessing: {e}")
        
        if target_column:
            target = data[target_column]
            return features[self.top_10_features], target
        
        return features[self.top_10_features]

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> Tuple[dict, LogisticRegression]:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)

        model = LogisticRegression(
            class_weight={
                1: len(y_train[y_train == 1]) / len(y_train),
                0: len(y_train[y_train == 0]) / len(y_train)
            }
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        self._model = model
        metrics = classification_report(y_test, y_pred, output_dict=True)
        return metrics, model

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            raise ValueError("Model is not trained or loaded")
        return self._model.predict(features).tolist()
    
    def save(self, path: Union[str, Path]):
        """Save the trained model to disk."""
        if self._model is None:
            raise ValueError("No model to save")
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: Union[str, Path]):
        """Load a model from disk."""
        with open(path, "rb") as f:
            self._model = pickle.load(f)
