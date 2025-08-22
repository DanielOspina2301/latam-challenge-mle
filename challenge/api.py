from contextlib import asynccontextmanager
import fastapi
from fastapi.params import Depends
import uvicorn

from challenge.core.dependencies import get_model
from challenge.core.settings import settings
from challenge.model import DelayModel
from challenge.schemas.prediction import RequestTemplate
from challenge.services.predict_service import PredictService
from challenge.services.training_service import TrainingService


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    app.state.model = DelayModel()
    service = TrainingService(app.state.model)
    service.update_model()
    yield


app = fastapi.FastAPI(
    title="Flight Delay Model",
    version=settings.APP_VERSION,
    description="API to calculate probability of flight delay",
    lifespan=lifespan
)

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: RequestTemplate, model: DelayModel = Depends(get_model)) -> dict:
    try:
        service = PredictService(model)
        predictions = service.predict(data=data.flights)
        return {"predict": predictions}
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))
    
@app.get("/fit", status_code=200)
async def get_fit(cloud_data:bool, model: DelayModel = Depends(get_model)) -> dict:
    try:
        service = TrainingService(model)
        trained_model_uri, metrics = service.train_model(cloud_data)
        return {
            "message": "Model trained successfully",
            "model_path": trained_model_uri,
            "metrics": metrics
        }
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=f"Error during training: {str(e)}")
    
@app.get("/update-model", status_code=200)
async def force_update_model(model_id: str, cloud: bool, model: DelayModel = Depends(get_model)) -> dict:
    if model_id.endswith('.pkl'):
        raise fastapi.HTTPException(status_code=400, detail='Model id should not have extension')
    try:
        service = TrainingService(model)
        service.update_model(model_name=f'{model_id}.pkl', cloud=cloud)
        return {'updated_model': model_id}
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=f'An error occurred during updating model: {str(e)}')


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, loop="asyncio")
