import logging
from src.domain.models.predict import PredictRequest, PredictResponse
from src.domain.services.llm_service import ILLMService

logger = logging.getLogger(__name__)

class PredictUseCase:
    def __init__(self, llm_service: ILLMService):
        self.llm_service = llm_service

    async def execute(self, request: PredictRequest) -> PredictResponse:
        logger.info(f"Executing prediction for query_id={request.query_id}, type={request.type}")
        
        if request.type == "type1":
            response = await self.llm_service.predict_type1(request)
        elif request.type == "type2":
            response = await self.llm_service.predict_type2(request)
        else:
            raise ValueError(f"Unsupported query type: {request.type}")
            
        logger.info(f"Prediction result for query_id={request.query_id}: answer='{response.answer}', premises_used={response.premises_used}")
        return response
