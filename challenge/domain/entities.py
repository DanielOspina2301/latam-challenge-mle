from dataclasses import dataclass
from typing import List


@dataclass
class PredictionInput:
    features: List[float]


@dataclass
class PredictionOutput:
    prediction: int
    probability: float
