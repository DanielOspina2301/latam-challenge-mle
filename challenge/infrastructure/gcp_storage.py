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
    

    def get_last_file(self, prefix: str):
        files = self.list_files(prefix)
        if not files:
            return None
        
        files.sort(key=lambda blob: blob.time_created, reverse=True)
        
        return files[0]
    

    def get_training_data(self, prefix: str = "training/"):
        latest_blob = self.get_last_file(prefix=prefix)
        if latest_blob:
            return latest_blob.download_as_text()
        return None
    

    def get_trained_model(self, prefix: str = "models/"):
        latest_blob = self.get_last_file(prefix=prefix)
        if latest_blob:
            return latest_blob.download_as_string()
        return None


    def get_file(self, file_name: str, prefix: str):
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            if not blobs:
                return None
            
            for blob in blobs:
                if blob.name == f"{prefix}{file_name}":
                    return blob.download_as_string()
                
            return None
        except Exception as e:
            raise RuntimeError(f"Get file from storage failed: {e}")
