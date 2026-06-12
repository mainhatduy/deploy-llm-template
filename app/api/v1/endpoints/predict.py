from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, HTTPException, status
from app.core.container import Container
from app.schema.predict_schema import PredictRequest, PredictResponse
from app.services.predict_service import PredictService

router = APIRouter(
    prefix="/predict",
    tags=["predict"],
)

@router.post("", response_model=PredictResponse, status_code=status.HTTP_200_OK)
@inject
async def predict(
    request_data: PredictRequest,
    service: PredictService = Depends(Provide[Container.predict_service]),
) -> PredictResponse:
    """
    Main prediction endpoint.
    Routes queries to Model Instance 1 (for Type 1 logic queries) or
    Model Instance 2 (for Type 2 math/physics queries).
    """
    try:
        response = await service.execute(request_data)
        return response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction execution failed: {str(e)}"
        )
