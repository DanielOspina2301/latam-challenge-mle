from datetime import datetime
from google.cloud import bigquery

from challenge.core.settings import settings


class BigQueryClient:
    def __init__(self):
        self.client = bigquery.Client(project=settings.project_id)
        self.dataset = settings.dataset_id
        self.table = settings.table_id


    def insert_metrics(self, metrics: dict, model_id: str):
        row_to_insert = {
            "model_id": model_id,
            "precision_0": metrics["0"]["precision"],
            "recall_0": metrics["0"]["recall"],
            "f1_score_0": metrics["0"]["f1-score"],
            "precision_1": metrics["1"]["precision"],
            "recall_1": metrics["1"]["recall"],
            "f1_score_1": metrics["1"]["f1-score"],
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["macro avg"]["precision"],
            "recall_macro": metrics["macro avg"]["recall"],
            "f1_score_macro": metrics["macro avg"]["f1-score"],
            "precision_weighted": metrics["weighted avg"]["precision"],
            "recall_weighted": metrics["weighted avg"]["recall"],
            "f1_score_weighted": metrics["weighted avg"]["f1-score"],
            "training_date": datetime.now().isoformat()
        }

        table_ref = f"{self.client.project}.{self.dataset}.{self.table}"
        errors = self.client.insert_rows_json(table_ref, [row_to_insert])
        if errors:
            raise RuntimeError(f"Bigquery insert failed: {errors}")
        return True
