from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "flight-delay"
    APP_VERSION: str = "1.0.0"
    BASE_PATH: str = f"/api/{APP_NAME}"

    MODELS_BUCKET_NAME: str = ""
    project_id: str = ""
    dataset_id: str = ""
    table_id: str = ""
    gcs_bucket: str = ""


    class Config:
        env_file = ".env"


settings = Settings()
