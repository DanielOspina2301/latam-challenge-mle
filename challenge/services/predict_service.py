from fastapi.params import Depends
import pandas as pd
from challenge.core.dependencies import get_model
from challenge.infrastructure.bigquery import BigQueryClient
from challenge.infrastructure.gcp_storage import GCSClient
from challenge.model import DelayModel


class PredictService:
    def __init__(self, model: DelayModel):
        self.model = model
        self.gcs_client = GCSClient()
        self.bigquery_client = BigQueryClient()

    def predict(self, data: list) -> list:
        data = [flight.__dict__ for flight in data]
        features = pd.DataFrame(data)
        features = self.model.preprocess(data=features)
        return self.model.predict(features=features)