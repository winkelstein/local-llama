from fastapi import APIRouter
from .home import router as home_router
from .chat_completion import router as chat_completion_router

router: APIRouter = APIRouter()

router.include_router(home_router)
router.include_router(chat_completion_router)
