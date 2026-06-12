# FastAPI Multi-Model vLLM Deployment Template

This project provides a clean template to deploy **two Large Language Models (LLMs)** on a **single GPU** using **vLLM** and **FastAPI**, structured under **Clean Architecture** principles.

The setup is optimized to handle two different types of tasks:
- **Type 1 (Logic/Reasoning)**: Routes requests to the first vLLM instance (e.g., Qwen3-0.6B).
- **Type 2 (Math/Physics/Calculation)**: Routes requests to the second vLLM instance.

---

## 🚀 Key Features

1. **Automatic Lifespan Management**: FastAPI automatically starts and stops background vLLM subprocesses matching the application lifecycle.
2. **GPU Memory Sharing**: Easy allocation of GPU resources (e.g., set `GPU_MEMORY_UTILIZATION=0.4` per instance to run two models concurrently on a single GPU).
3. **Clean Architecture Structure**:
   - `domain`: Contains business models and service interfaces.
   - `use_cases`: Houses the application use cases (`PredictUseCase`).
   - `infrastructure`: Implements details for configuration, vLLM subprocesses, and HTTP clients.
   - `presentation`: Exposes RESTful APIs via FastAPI.
4. **Mocking Mode (`MOCK_VLLM`)**: Support for development/testing on machines without a GPU (Local/CPU) by mimicking model outputs.

---

## 🛠️ Environment Setup

### Prerequisites
- Operating System: Linux (Ubuntu recommended).
- Python: Version `3.9` or higher (`3.10` / `3.11` recommended).
- CUDA Toolkit (if deploying with actual vLLM instances on GPU).

### Installation Steps

1. **Create a Virtual Environment**
   ```bash
   # Create a virtual environment named .venv
   python3 -m venv .venv
   
   # Activate the virtual environment
   source .venv/bin/activate
   ```

2. **Upgrade pip and Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**
   Copy the example environment configuration file:
   ```bash
   cp .env.example .env
   ```
   Open `.env` and adjust the variables to fit your environment:
   - **`MOCK_VLLM`**: Set to `true` to run on machines without GPU/vLLM. Set to `false` in actual production deployments.
   - **`GPU_MEMORY_UTILIZATION`**: The fraction of GPU memory allocated per instance (e.g., `0.4` allocates 40% memory to each instance). Ensure the total memory usage of all active instances does not exceed your GPU capacity.
   - **`MODEL_NAME_TYPE1` & `MODEL_NAME_TYPE2`**: The target models hosted on Hugging Face (e.g., `Qwen/Qwen3-0.6B`).
   - **`HF_TOKEN`**: Provide a Hugging Face token if you use gated models.

---

## 🏃 Running the Application

### Step 1: Start the FastAPI Server

Once the virtual environment is activated, run:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

> [!NOTE]
> - If `START_VLLM_ON_STARTUP=true` and `MOCK_VLLM=false`, FastAPI will start downloading the models from Hugging Face (if not cached) and launch the vLLM subprocesses on ports `8001` and `8002`. This process might take several minutes.
> - If you want the server to auto-reload on code changes, add the `--reload` flag. Keep in mind that every reload will restart the underlying vLLM processes.

### Step 2: Check System Health

Verify that the API server and both vLLM instances are online using the `/health` endpoint:

```bash
curl http://localhost:8000/health
```

**Success Response:**
```json
{
  "status": "healthy",
  "api_server": "running",
  "vllm_servers": "healthy"
}
```

### Step 3: Interactive API Docs (Swagger UI)
Visit the API docs page in your browser:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)** to explore endpoints and test requests interactively.

---

## 🧪 Testing Predictions

A test script `test_predict.py` is included to easily send requests to the `/predict` endpoint.

Open another terminal session (activate `.venv`) and run:

```bash
python test_predict.py
```

The script automatically sends three payloads:
1. **Type 1: Multiple Choice Question** - Testing logical reasoning.
2. **Type 1: Free-form Question** - Testing text completion.
3. **Type 2: Physics Problem** - Testing calculations.

---

## 📁 Directory Structure

```text
.
├── src/
│   ├── domain/               # Domain models and service interfaces
│   ├── use_cases/            # Core business logic (PredictUseCase)
│   ├── infrastructure/       # vLLM integration, configuration, and subprocess manager
│   │   ├── config/           # Pydantic settings parsing the .env file
│   │   ├── llm/              # Async OpenAI client integration with vLLM
│   │   └── process/          # Subprocess manager starting/stopping vLLM instances
│   ├── presentation/         # FastAPI router and endpoints definition
│   └── main.py               # Application entrypoint & lifespan lifecycle hooks
├── logs/                     # Stdout and stderr logs for vLLM subprocesses
├── .env.example              # Sample environment configuration template
├── requirements.txt          # Python dependencies
├── test_predict.py           # Verification script for local API testing
└── README.md                 # Project documentation (This file)
```
