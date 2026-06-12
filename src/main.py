import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from src.infrastructure.config.settings import settings
from src.infrastructure.llm.vllm_service_impl import VLLMServiceImpl
from src.infrastructure.process.vllm_process_manager import VLLMProcessManager
from src.presentation.api.router import api_router

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Instantiate process manager
vllm_manager = VLLMProcessManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events manager for FastAPI.
    Handles startup of vLLM background servers and initialization of clients,
    and ensures clean termination of child processes on shutdown.
    """
    logger.info("Starting up FastAPI application...")
    
    # 1. Start vLLM background instances
    try:
        vllm_manager.start_all()
    except Exception as e:
        logger.error(f"Failed to start vllm servers during startup: {e}", exc_info=True)
        # We don't crash the API server, but we log the error so debugging is easier.

    # 2. Initialize LLM Service client
    llm_service = VLLMServiceImpl()
    app.state.llm_service = llm_service
    
    logger.info("Application startup completed. Ready to serve requests.")
    
    yield
    
    # 3. Shutdown process manager and child processes
    logger.info("Shutting down FastAPI application...")
    try:
        vllm_manager.stop_all()
    except Exception as e:
        logger.error(f"Error during vLLM server shutdown: {e}", exc_info=True)
    logger.info("Application shutdown completed.")

# Initialize FastAPI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Deploying 2 Qwen models in 1 GPU using vLLM and FastAPI Clean Architecture",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include main API router
app.include_router(api_router)

@app.get("/health", status_code=status.HTTP_200_OK, tags=["monitoring"])
async def health_check():
    """
    Monitoring endpoint.
    Checks if the API is up and checks the health status of both vLLM models.
    """
    vllm_healthy = False
    if hasattr(app.state, "llm_service"):
        vllm_healthy = await app.state.llm_service.check_health()
        
    return {
        "status": "healthy" if vllm_healthy else "degraded",
        "api_server": "running",
        "vllm_servers": "healthy" if vllm_healthy else "unhealthy"
    }
