import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import configs
from app.core.container import Container
from app.api.v1.routes import routers as v1_routers

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize Dependency Injection Container
container = Container()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events manager for FastAPI.
    Handles startup of vLLM background servers,
    and ensures clean termination of child processes on shutdown.
    """
    logger.info("Starting up FastAPI application...")
    
    # Resolve process manager from DI container
    vllm_manager = container.vllm_process_manager()
    
    # 1. Start vLLM background instances
    try:
        vllm_manager.start_all()
    except Exception as e:
        logger.error(f"Failed to start vllm servers during startup: {e}", exc_info=True)

    logger.info("Application startup completed. Ready to serve requests.")
    
    yield
    
    # 2. Shutdown process manager and child processes
    logger.info("Shutting down FastAPI application...")
    try:
        vllm_manager.stop_all()
    except Exception as e:
        logger.error(f"Error during vLLM server shutdown: {e}", exc_info=True)
    logger.info("Application shutdown completed.")

# Initialize FastAPI App
app = FastAPI(
    title=configs.PROJECT_NAME,
    description="Deploying 2 Qwen models in 1 GPU using vLLM and FastAPI Clean Architecture (following jujumilk3)",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in configs.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include v1 API routers at the root level (so /predict is directly under /)
app.include_router(v1_routers)

# Also support prefix /api/v1 for compatibility
app.include_router(v1_routers, prefix=configs.API_V1_STR)

@app.get("/health", status_code=status.HTTP_200_OK, tags=["monitoring"])
async def health_check():
    """
    Monitoring endpoint.
    Checks if the API is up and checks the health status of both vLLM models.
    """
    vllm_repo = container.vllm_repository()
    vllm_healthy = await vllm_repo.check_health()
        
    return {
        "status": "healthy" if vllm_healthy else "degraded",
        "api_server": "running",
        "vllm_servers": "healthy" if vllm_healthy else "unhealthy"
    }
