import os
import subprocess
import time
import httpx
import logging
import signal
from pathlib import Path
from src.infrastructure.config.settings import settings

logger = logging.getLogger(__name__)

class VLLMProcessManager:
    def __init__(self):
        self.process_type1 = None
        self.process_type2 = None
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

    def is_port_in_use(self, host: str, port: int) -> bool:
        """Check if a port is already being listened to."""
        try:
            with httpx.Client() as client:
                # Try to hit the health endpoint of vLLM or just a basic check
                response = client.get(f"http://{host}:{port}/health")
                return response.status_code == 200
        except httpx.RequestError:
            # If request fails, let's try a simple connection test by attempting to hit it
            try:
                import socket
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)
                    s.connect((host, port))
                    return True
            except (socket.timeout, ConnectionRefusedError):
                return False

    def start_vllm_instance(
        self,
        name: str,
        host: str,
        port: int,
        model_name: str,
        gpu_memory_utilization: float,
        max_model_len: int,
        log_file: Path
    ) -> subprocess.Popen:
        """Start a single vLLM instance as a subprocess."""
        if self.is_port_in_use(host, port):
            logger.info(f"vLLM instance '{name}' appears to be already running on http://{host}:{port}. Skipping startup.")
            return None

        # Build command
        # Using sys.executable to run inside the same virtual environment
        import sys
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--host", host,
            "--port", str(port),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-model-len", str(max_model_len),
            "--disable-log-requests"
        ]

        logger.info(f"Starting vLLM instance '{name}' on port {port}...")
        logger.info(f"Command: {' '.join(cmd)}")

        # Open log file
        log_fp = open(log_file, "w")
        
        # Build environment
        env = os.environ.copy()
        if settings.HF_TOKEN:
            env["HF_TOKEN"] = settings.HF_TOKEN

        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid  # Create a process group so we can terminate it cleanly
        )
        return process

    def wait_for_healthy(self, host: str, port: int, timeout: int = 300) -> bool:
        """Poll the health/v1/models endpoint until healthy or timeout."""
        start_time = time.time()
        url = f"http://{host}:{port}/v1/models"
        logger.info(f"Waiting for vLLM server on {url} to become healthy...")
        
        while time.time() - start_time < timeout:
            try:
                # We can also check /health, but /v1/models ensures model is actually loaded
                with httpx.Client() as client:
                    response = client.get(url, timeout=2.0)
                    if response.status_code == 200:
                        logger.info(f"vLLM server on port {port} is healthy and model is loaded!")
                        return True
            except httpx.RequestError:
                pass
            time.sleep(settings.VLLM_POLLING_INTERVAL)
            
        logger.error(f"Timeout waiting for vLLM server on port {port} to become healthy.")
        return False

    def start_all(self):
        """Start both vLLM instances and wait for them to become healthy."""
        if settings.MOCK_VLLM:
            logger.info("MOCK_VLLM is set to True. Skipping starting actual vLLM subprocesses.")
            return

        if not settings.START_VLLM_ON_STARTUP:
            logger.info("START_VLLM_ON_STARTUP is set to False. Skipping vLLM startup.")
            return

        logger.info("Initializing vLLM instances...")
        
        # Instance 1 (Type 1)
        self.process_type1 = self.start_vllm_instance(
            name="Type 1 Server",
            host=settings.VLLM_HOST_TYPE1,
            port=settings.VLLM_PORT_TYPE1,
            model_name=settings.model_name_type1,
            gpu_memory_utilization=settings.gpu_memory_utilization_type1,
            max_model_len=settings.max_model_len_type1,
            log_file=self.log_dir / "vllm_type1.log"
        )
        
        # Instance 2 (Type 2)
        self.process_type2 = self.start_vllm_instance(
            name="Type 2 Server",
            host=settings.VLLM_HOST_TYPE2,
            port=settings.VLLM_PORT_TYPE2,
            model_name=settings.model_name_type2,
            gpu_memory_utilization=settings.gpu_memory_utilization_type2,
            max_model_len=settings.max_model_len_type2,
            log_file=self.log_dir / "vllm_type2.log"
        )

        # Wait for them to load the models and become healthy
        h1 = True
        h2 = True
        if self.process_type1 is not None or self.is_port_in_use(settings.VLLM_HOST_TYPE1, settings.VLLM_PORT_TYPE1):
            h1 = self.wait_for_healthy(settings.VLLM_HOST_TYPE1, settings.VLLM_PORT_TYPE1, settings.VLLM_STARTUP_TIMEOUT_SECONDS)
        
        if self.process_type2 is not None or self.is_port_in_use(settings.VLLM_HOST_TYPE2, settings.VLLM_PORT_TYPE2):
            h2 = self.wait_for_healthy(settings.VLLM_HOST_TYPE2, settings.VLLM_PORT_TYPE2, settings.VLLM_STARTUP_TIMEOUT_SECONDS)

        if not (h1 and h2):
            logger.warning("One or both vLLM instances failed to start or load models successfully.")

    def stop_process(self, process: subprocess.Popen, name: str):
        """Stop a running subprocess gracefully."""
        if process is None:
            return
            
        logger.info(f"Stopping vLLM process '{name}' (PID: {process.pid})...")
        try:
            # Send SIGTERM to process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            # Wait up to 10 seconds for termination
            for _ in range(20):
                if process.poll() is not None:
                    logger.info(f"vLLM process '{name}' stopped gracefully.")
                    return
                time.sleep(0.5)
                
            # If still running, force kill
            logger.warning(f"vLLM process '{name}' did not stop in time. Sending SIGKILL...")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()
            logger.info(f"vLLM process '{name}' force killed.")
        except Exception as e:
            logger.error(f"Error stopping vLLM process '{name}': {e}")

    def stop_all(self):
        """Stop both managed vLLM instances."""
        if self.process_type1:
            self.stop_process(self.process_type1, "Type 1 Server")
            self.process_type1 = None
        if self.process_type2:
            self.stop_process(self.process_type2, "Type 2 Server")
            self.process_type2 = None
