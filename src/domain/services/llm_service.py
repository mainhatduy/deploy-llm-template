from abc import ABC, abstractmethod
from src.domain.models.predict import PredictRequest, PredictResponse

class ILLMService(ABC):
    @abstractmethod
    async def predict_type1(self, request: PredictRequest) -> PredictResponse:
        """
        Process a Type 1 query (logic reasoning over premises, optional choices).
        Uses Model Instance 1.
        """
        pass

    @abstractmethod
    async def predict_type2(self, request: PredictRequest) -> PredictResponse:
        """
        Process a Type 2 query (physics/math problem, no premises/options).
        Uses Model Instance 2.
        """
        pass
    
    @abstractmethod
    async def check_health(self) -> bool:
        """
        Checks health of the underlying vLLM model instances.
        """
        pass
