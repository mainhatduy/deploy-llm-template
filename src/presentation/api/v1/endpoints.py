from fastapi import APIRouter, Depends, HTTPException, Request, status
from src.domain.models.predict import PredictRequest, PredictResponse
from src.domain.services.llm_service import ILLMService
from src.use_cases.predict_use_case import PredictUseCase

router = APIRouter()

def get_llm_service(request: Request) -> ILLMService:
    """Retrieve the LLM service singleton from application state."""
    if not hasattr(request.app.state, "llm_service"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM Service not initialized in application state"
        )
    return request.app.state.llm_service

def get_predict_use_case(llm_service: ILLMService = Depends(get_llm_service)) -> PredictUseCase:
    """Dependency factory for PredictUseCase."""
    return PredictUseCase(llm_service)

@router.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
async def predict(
    request_data: PredictRequest,
    use_case: PredictUseCase = Depends(get_predict_use_case)
) -> PredictResponse:
    """
    Main prediction endpoint.
    Routes queries to Model Instance 1 (for Type 1 logic queries) or
    Model Instance 2 (for Type 2 math/physics queries).
    """
    try:
        response = await use_case.execute(request_data)
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
