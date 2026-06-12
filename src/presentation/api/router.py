from fastapi import APIRouter
from src.presentation.api.v1 import endpoints

api_router = APIRouter()

# Include prediction endpoints at the root level (e.g. /predict)
# so they match the evaluation server's expectations exactly.
api_router.include_router(endpoints.router, tags=["prediction"])
