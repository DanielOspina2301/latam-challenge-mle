import fastapi
import uvicorn

from challenge.core.settings import settings
from challenge.schemas.prediction import RequestTemplate
from challenge.services.training_service import TrainingService


app = fastapi.FastAPI(
    title="Flight Delay Model",
    version=settings.APP_VERSION,
    description="API to calculate probability of flight delay"
)

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: RequestTemplate) -> dict:
    try:
        predictions = [0]
        return {"predict": predictions}
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e))
    
@app.get("/fit", status_code=200)
async def get_fit() -> dict:
    try:
        service = TrainingService()
        trained_model_uri, metrics = service.train_model()
        return {
            "message": "Model trained successfully",
            "model_path": trained_model_uri,
            "metrics": metrics
        }
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=f"Error during training: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, loop="asyncio")
