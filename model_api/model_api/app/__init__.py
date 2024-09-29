from fastapi import FastAPI
from .routes import router

app: FastAPI = FastAPI()

app.include_router(router)
