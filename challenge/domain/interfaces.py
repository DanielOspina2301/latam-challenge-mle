from abc import ABC, abstractmethod

from challenge.domain.entities import PredictionInput, PredictionOutput


class ModelLoaderPort(ABC):

    @abstractmethod
    def load_model(self):
        pass


class PredictorPort(ABC):

    @abstractmethod
    def predict(self, data: PredictionInput) -> PredictionOutput:
        pass
