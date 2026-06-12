from fastapi import APIRouter
from app.api.v1.endpoints.predict import router as predict_router

routers = APIRouter()
router_list = [predict_router]

for router in router_list:
    routers.include_router(router)
