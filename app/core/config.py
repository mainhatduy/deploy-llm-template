import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Configs(BaseSettings):
    # API configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "FastAPI LLM Deployment template"

    # vLLM common configuration (defaults/fallbacks)
    GPU_MEMORY_UTILIZATION: float = 0.4
    MAX_MODEL_LEN: int = 4096
    
    # Instance 1 configuration (for Type 1 queries)
    MODEL_NAME_TYPE1: str = "Qwen/Qwen3-0.6B"
    GPU_MEMORY_UTILIZATION_TYPE1: float | None = None
    MAX_MODEL_LEN_TYPE1: int | None = None
    VLLM_PORT_TYPE1: int = 8001
    VLLM_HOST_TYPE1: str = "127.0.0.1"

    # Instance 2 configuration (for Type 2 queries)
    MODEL_NAME_TYPE2: str = "Qwen/Qwen3-0.6B"
    GPU_MEMORY_UTILIZATION_TYPE2: float | None = None
    MAX_MODEL_LEN_TYPE2: int | None = None
    VLLM_PORT_TYPE2: int = 8002
    VLLM_HOST_TYPE2: str = "127.0.0.1"

    # Process Management Configuration
    START_VLLM_ON_STARTUP: bool = True
    VLLM_STARTUP_TIMEOUT_SECONDS: int = 300  # vLLM model download & load might take time
    VLLM_POLLING_INTERVAL: float = 2.0
    MOCK_VLLM: bool = False

    # Hugging Face configuration
    HF_TOKEN: str | None = None

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Resolved properties for Type 1 Instance
    @property
    def model_name_type1(self) -> str:
        return self.MODEL_NAME_TYPE1

    @property
    def gpu_memory_utilization_type1(self) -> float:
        return self.GPU_MEMORY_UTILIZATION_TYPE1 if self.GPU_MEMORY_UTILIZATION_TYPE1 is not None else self.GPU_MEMORY_UTILIZATION

    @property
    def max_model_len_type1(self) -> int:
        return self.MAX_MODEL_LEN_TYPE1 if self.MAX_MODEL_LEN_TYPE1 is not None else self.MAX_MODEL_LEN

    # Resolved properties for Type 2 Instance
    @property
    def model_name_type2(self) -> str:
        return self.MODEL_NAME_TYPE2

    @property
    def gpu_memory_utilization_type2(self) -> float:
        return self.GPU_MEMORY_UTILIZATION_TYPE2 if self.GPU_MEMORY_UTILIZATION_TYPE2 is not None else self.GPU_MEMORY_UTILIZATION

    @property
    def max_model_len_type2(self) -> int:
        return self.MAX_MODEL_LEN_TYPE2 if self.MAX_MODEL_LEN_TYPE2 is not None else self.MAX_MODEL_LEN

configs = Configs()
