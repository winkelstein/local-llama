from fastapi import APIRouter
from ..llama import llama
from pydantic import BaseModel

router = APIRouter()


class HomeResponse(BaseModel):
    api_version: str
    model: str
    use_gpu: str
    context_window: str


@router.get("/")
async def home() -> HomeResponse:
    return {
        "api_version": "0.1.0",
        "model": llama.model_name,
        "use_gpu": str(llama.use_gpu),
        "context_window": str(llama.context_window),
    }
