import logging
from app.schema.predict_schema import PredictRequest, PredictResponse
from app.repository.vllm_repository import VLLMRepository

logger = logging.getLogger(__name__)

class PredictService:
    def __init__(self, vllm_repository: VLLMRepository):
        self.vllm_repository = vllm_repository

    async def execute(self, request: PredictRequest) -> PredictResponse:
        logger.info(f"Executing prediction for query_id={request.query_id}, type={request.type}")
        
        if request.type == "type1":
            response = await self.vllm_repository.predict_type1(request)
        elif request.type == "type2":
            response = await self.vllm_repository.predict_type2(request)
        else:
            raise ValueError(f"Unsupported query type: {request.type}")
            
        logger.info(f"Prediction result for query_id={request.query_id}: answer='{response.answer}', premises_used={response.premises_used}")
        return response
