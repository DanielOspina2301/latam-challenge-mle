import pickle
from google.cloud import storage

from challenge.core.settings import settings


class GCSClient:
    def __init__(self):
        self.client = storage.Client(project=settings.project_id)
        self.bucket = self.client.bucket(settings.gcs_bucket)

    
    def upload_model(self, model, dest_blob_name: str):
        model_pickle = pickle.dumps(model)
        blob = self.bucket.blob(dest_blob_name)
        blob.upload_from_string(model_pickle, content_type="application/octet-stream")
        return f"gs://{settings.gcs_bucket}/{dest_blob_name}"
    

    def download_file(self, blob_name: str, local_path: str):
        blob = self.bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        return local_path
    

    def list_files(self, prefix: str = None):
        return list(self.client.list_blobs(self.bucket, prefix=prefix))
    

    def get_last_training_file(self, prefix: str = "training/"):
        files = self.list_files(prefix)
        if not files:
            return None
        
        files.sort(key=lambda blob: blob.time_created, reverse=True)
        latest_blob = files[0]
        return latest_blob.download_as_text()
