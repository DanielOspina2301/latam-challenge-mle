import uuid
from challenge.core.settings import settings
from challenge.infrastructure.bigquery import BigQueryClient
from challenge.infrastructure.gcp_storage import GCSClient
from challenge.model import DelayModel
from challenge.utils.utils import load_data_from_csv


class TrainingService:
    def __init__(self):
        self.model = DelayModel(
            top_features=[
                "OPERA_Latin American Wings",
                "MES_7",
                "MES_10",
                "OPERA_Grupo LATAM",
                "MES_12",
                "TIPOVUELO_I",
                "MES_4",
                "MES_11",
                "OPERA_Sky Airline",
                "OPERA_Copa Air"
            ]
        )
        self.gcs_client = GCSClient()
        self.bigquery_client = BigQueryClient()


    def train_model(self) -> str:
        last_file = self.gcs_client.get_last_training_file()
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
