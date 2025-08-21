from pathlib import Path
import pickle
from challenge.domain.interfaces import ModelLoaderPort


class PickleModelLoader(ModelLoaderPort):
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)

    
    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        with open(self.model_path, "rb") as f:
            return pickle.load(f)
