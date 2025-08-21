import pickle
import uuid

from fastapi.params import Depends
from challenge.core.dependencies import get_model
from challenge.core.settings import settings
from challenge.infrastructure.bigquery import BigQueryClient
from challenge.infrastructure.gcp_storage import GCSClient
from challenge.model import DelayModel
from challenge.utils.utils import load_data_from_csv


class TrainingService:
    def __init__(self, model: DelayModel):
        self.model = model
        self.gcs_client = GCSClient()
        self.bigquery_client = BigQueryClient()


    def train_model(self) -> str:
        last_file = self.gcs_client.get_training_data()
        if not last_file:
            raise FileNotFoundError("File not found in GCS")
        
        data = load_data_from_csv(csv_data=last_file)

        features, target = self.model.preprocess(data=data, target_column="delay")
        metrics, model = self.model.fit(features=features, target=target)
        model_id = f"{uuid.uuid4()}.pkl"
        
        dest_blob_name = f"models/{model_id}"
        gcs_uri = self.gcs_client.upload_model(model, dest_blob_name=dest_blob_name)
        
        self.bigquery_client.insert_metrics({
            **metrics
        }, model_id=model_id)

        return gcs_uri, metrics
    
    def update_model(self, model_name: str = None):
        if model_name:
            trained_model = self.gcs_client.get_file(file_name=model_name, prefix="models/")

            if trained_model:
                self.model.load(model=trained_model)
            else:
                raise FileNotFoundError(f"Model {model_name} does not exist in the bucket.")
            
        last_model = self.gcs_client.get_trained_model()
        if last_model:
            self.model.load(model=last_model)
        else:
            raise FileNotFoundError("There are no models in the bucket.")
