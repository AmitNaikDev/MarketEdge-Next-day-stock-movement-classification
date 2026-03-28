from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.dependencies import load_models ,clear_models
from api.routers import prediction  

@asynccontextmanager
async def lifespan(app:FastAPI):
    load_models()
    yield
    clear_models()

app = FastAPI(
    title = "Stock Movement Classifier API",
    description = "Predicts next-day stock price direction using RF and LSTM.",
    version = "1.0.0",
    lifespan = lifespan 
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

app.include_router(prediction.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
