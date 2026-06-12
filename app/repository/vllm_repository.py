import json
import logging
import re
import httpx
from openai import AsyncOpenAI
from app.schema.predict_schema import PredictRequest, PredictResponse
from app.core.config import configs

logger = logging.getLogger(__name__)

class VLLMRepository:
    def __init__(self):
        # Initialize two clients pointing to different ports
        self.client_type1 = AsyncOpenAI(
            base_url=f"http://{configs.VLLM_HOST_TYPE1}:{configs.VLLM_PORT_TYPE1}/v1",
            api_key="empty-key-for-vllm"
        )
        self.client_type2 = AsyncOpenAI(
            base_url=f"http://{configs.VLLM_HOST_TYPE2}:{configs.VLLM_PORT_TYPE2}/v1",
            api_key="empty-key-for-vllm"
        )
        self._model_name_type1 = None
        self._model_name_type2 = None

    async def _resolve_model_name(self, client: AsyncOpenAI, is_type1: bool) -> str:
        """Dynamically fetch the model name from the running vLLM server."""
        cached_name = self._model_name_type1 if is_type1 else self._model_name_type2
        if cached_name:
            return cached_name

        fallback_name = configs.model_name_type1 if is_type1 else configs.model_name_type2
        try:
            models = await client.models.list()
            if models.data:
                resolved_name = models.data[0].id
                if is_type1:
                    self._model_name_type1 = resolved_name
                else:
                    self._model_name_type2 = resolved_name
                logger.info(f"Resolved vLLM model name for {'Type 1' if is_type1 else 'Type 2'}: {resolved_name}")
                return resolved_name
        except Exception as e:
            logger.warning(f"Failed to fetch model name from vLLM server: {e}. Falling back to default: {fallback_name}")
        
        return fallback_name

    def _clean_json_response(self, content: str) -> str:
        # Remove thinking block if present
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        
        # Strip markdown code blocks
        content_clean = content.strip()
        if content_clean.startswith("```json"):
            content_clean = content_clean[7:]
        elif content_clean.startswith("```"):
            content_clean = content_clean[3:]
        if content_clean.endswith("```"):
            content_clean = content_clean[:-3]
        content_clean = content_clean.strip()
        
        # Extract the JSON object using regex if possible
        match = re.search(r"(\{.*\})", content_clean, re.DOTALL)
        if match:
            content_clean = match.group(1)
            
        return content_clean

    async def predict_type1(self, request: PredictRequest) -> PredictResponse:
        if configs.MOCK_VLLM:
            logger.info(f"MOCK_VLLM is enabled. Mocking Type 1 prediction for query_id={request.query_id}...")
            query_lower = request.query.lower()
            if "eligible for graduation" in query_lower:
                return PredictResponse(
                    query_id=request.query_id,
                    answer="No",
                    premises_used=[0, 1]
                )
            elif "more credits" in query_lower:
                return PredictResponse(
                    query_id=request.query_id,
                    answer="2",
                    premises_used=[0, 1]
                )
            fallback_ans = request.options[0] if request.options else "mock_answer"
            mock_premises = list(range(len(request.premises)))
            return PredictResponse(
                query_id=request.query_id,
                answer=fallback_ans,
                premises_used=mock_premises
            )

        model = await self._resolve_model_name(self.client_type1, is_type1=True)

        system_prompt = (
            "You are a logical reasoning assistant. Analyze the query using the provided premises.\n"
            "You must output a JSON object containing the following keys:\n"
            "- \"answer\": string, the final answer to the query. If options are provided, this MUST be one of the options.\n"
            "- \"premises_used\": list of integers, the 0-based indices of the premises used to answer the query.\n\n"
            "Ensure the response is a valid JSON object and contains nothing else."
        )

        user_prompt_parts = []
        if request.premises:
            user_prompt_parts.append("Premises:")
            for idx, premise in enumerate(request.premises):
                user_prompt_parts.append(f"[{idx}] {premise}")
            user_prompt_parts.append("")

        if request.options:
            user_prompt_parts.append("Options:")
            for option in request.options:
                user_prompt_parts.append(f"- {option}")
            user_prompt_parts.append("")

        user_prompt_parts.append(f"Query: {request.query}")
        user_prompt = "\n".join(user_prompt_parts)

        try:
            logger.info(f"Sending Type 1 chat completion request to vLLM on port {configs.VLLM_PORT_TYPE1}...")
            chat_response = await self.client_type1.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = chat_response.choices[0].message.content
            logger.debug(f"Raw response content from Type 1 server: {content}")
            
            content_clean = self._clean_json_response(content)
            result = json.loads(content_clean)
            
            return PredictResponse(
                query_id=request.query_id,
                answer=str(result.get("answer", "")),
                premises_used=result.get("premises_used", [])
            )
            
        except Exception as e:
            logger.error(f"Error calling Type 1 vLLM server: {e}", exc_info=True)
            fallback_ans = request.options[0] if request.options else "Error occurred during generation"
            return PredictResponse(
                query_id=request.query_id,
                answer=fallback_ans,
                premises_used=[]
            )

    async def predict_type2(self, request: PredictRequest) -> PredictResponse:
        if configs.MOCK_VLLM:
            logger.info(f"MOCK_VLLM is enabled. Mocking Type 2 prediction for query_id={request.query_id}...")
            query_lower = request.query.lower()
            if "resistors" in query_lower and "parallel" in query_lower:
                return PredictResponse(
                    query_id=request.query_id,
                    answer="5A",
                    premises_used=[]
                )
            return PredictResponse(
                query_id=request.query_id,
                answer="42",
                premises_used=[]
            )

        model = await self._resolve_model_name(self.client_type2, is_type1=False)

        system_prompt = (
            "You are a helpful assistant for math, physics, and calculations. Solve the problem step by step.\n"
            "You must output a JSON object containing the following keys:\n"
            "- \"answer\": string, the final numerical/text answer to the query.\n"
            "- \"premises_used\": list of integers, which should be empty [] since no premises are provided.\n\n"
            "Ensure the response is a valid JSON object and contains nothing else."
        )

        user_prompt = f"Query: {request.query}"

        try:
            logger.info(f"Sending Type 2 chat completion request to vLLM on port {configs.VLLM_PORT_TYPE2}...")
            chat_response = await self.client_type2.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = chat_response.choices[0].message.content
            logger.debug(f"Raw response content from Type 2 server: {content}")
            
            content_clean = self._clean_json_response(content)
            result = json.loads(content_clean)
            
            return PredictResponse(
                query_id=request.query_id,
                answer=str(result.get("answer", "")),
                premises_used=result.get("premises_used", [])
            )
            
        except Exception as e:
            logger.error(f"Error calling Type 2 vLLM server: {e}", exc_info=True)
            return PredictResponse(
                query_id=request.query_id,
                answer="Error occurred during calculation",
                premises_used=[]
            )

    async def check_health(self) -> bool:
        """Check if both vLLM servers are responding and healthy."""
        if configs.MOCK_VLLM:
            return True
        try:
            async with httpx.AsyncClient() as client:
                res1 = await client.get(f"http://{configs.VLLM_HOST_TYPE1}:{configs.VLLM_PORT_TYPE1}/health")
                res2 = await client.get(f"http://{configs.VLLM_HOST_TYPE2}:{configs.VLLM_PORT_TYPE2}/health")
                return res1.status_code == 200 and res2.status_code == 200
        except Exception:
            return False
