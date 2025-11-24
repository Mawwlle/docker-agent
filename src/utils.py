import docker
import os

from dotenv import load_dotenv, find_dotenv
from langchain_deepseek import ChatDeepSeek

load_dotenv(find_dotenv(usecwd=True))


def get_docker_client() -> docker.DockerClient:
    return docker.from_env()


def get_llm(
    base_url: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
) -> ChatDeepSeek:
    base_url = base_url or os.getenv("LITELLM_BASE_URL", "http://a6k2.dgx:34000/v1")
    api_key = api_key or os.getenv("OPENAPI_API_KEY", "")
    model_name = model_name or os.getenv("MODEL_NAME", "qwen3-32b")

    return ChatDeepSeek(
        api_base=base_url,
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        streaming=True,
    )
