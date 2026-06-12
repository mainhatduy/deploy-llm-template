from dependency_injector import containers, providers

from app.core.config import configs
from app.repository.vllm_repository import VLLMRepository
from app.services.predict_service import PredictService
from app.util.vllm_process_manager import VLLMProcessManager

class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(
        modules=[
            "app.api.v1.endpoints.predict",
        ]
    )

    vllm_process_manager = providers.Singleton(VLLMProcessManager)

    vllm_repository = providers.Singleton(VLLMRepository)

    predict_service = providers.Factory(
        PredictService,
        vllm_repository=vllm_repository,
    )
