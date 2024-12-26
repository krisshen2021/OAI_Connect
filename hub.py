from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI, AsyncOpenAI
from openai import (
    OpenAIError,  # 基础异常类
    APIError,  # API 通用错误
    BadRequestError,  # 400 错误
    AuthenticationError,  # 401 错误
    PermissionDeniedError,  # 403 错误
    NotFoundError,  # 404 错误
    RateLimitError,  # 429 错误
    APIConnectionError,  # 连接错误
    APITimeoutError,  # 超时错误
    InternalServerError,  # 500 错误
    APIStatusError,  # 非 2xx 响应
)
import json, ast
import uvicorn


app = FastAPI(title="OAI HUB API Routers", description="For Inference easily")
# Load configuration from JSON files
with open("config.json") as f:
    config = json.load(f)
for api_Name in config.keys():
    config[api_Name]['api_key'] = os.getenv(f"{api_Name.upper()}_API_KEY")


class ChatMessage(BaseModel):
    role: str
    content: str | List[Dict[str, str | Dict[str, str]]]


class OAIParam(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: float = None
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    model: str
    stream: Optional[bool] = True


# def a error info process function
def openai_error_msg(e):
    try:
        error_str = str(e)
        dict_str = error_str.split(" - ", 1)[1]
        error_list = ast.literal_eval(dict_str)
        error_info = error_list[0]["error"]
        return error_info.get("message")
    except:
        return str(e)


# Create routers dynamically based on the configuration
for api_Name in config.keys():
    router = APIRouter()

    async def proxy_stream(stream):
        async for chunk in stream:
            yield f"data: {chunk.model_dump_json()}\n\n"

    @router.post(f"/chat/completions")
    async def chat_completions(request: OAIParam, api_Name=api_Name):
        client = AsyncOpenAI(
            base_url=config[api_Name]["base_url"], api_key=config[api_Name]["api_key"]
        )
        payload = request.model_dump(exclude_none=True, exclude_unset=True)
        try:
            result = await client.chat.completions.create(**payload)

            if request.stream:
                return StreamingResponse(
                    proxy_stream(result), media_type="text/event-stream"
                )
            else:
                return result

        except BadRequestError as e:
            # 处理无效请求，如错误的参数
            error_msg = openai_error_msg(e)
            print(f"Bad request: {e.status_code} - {error_msg}")
            raise HTTPException(status_code=e.status_code, detail=error_msg)

        except AuthenticationError as e:
            # 处理认证错误，如 API key 无效
            error_msg = openai_error_msg(e)
            print(f"Authentication failed: {e.status_code} - {error_msg}")
            raise HTTPException(status_code=e.status_code, detail=error_msg)

        except PermissionDeniedError as e:
            # 处理权限错误，如无权访问资源
            error_msg = openai_error_msg(e)
            print(f"Permission denied: {e.status_code} - {error_msg}")
            raise HTTPException(status_code=e.status_code, detail=error_msg)

        except NotFoundError as e:
            # 处理资源未找到错误
            error_msg = openai_error_msg(e)
            print(f"Resource not found: {e.status_code} - {error_msg}")
            raise HTTPException(status_code=e.status_code, detail=error_msg)

        except RateLimitError as e:
            # 处理速率限制错误
            error_msg = openai_error_msg(e)
            print(f"Rate limit exceeded: {e.status_code} - {error_msg}")
            raise HTTPException(status_code=e.status_code, detail=error_msg)

        except InternalServerError as e:
            # 处理服务器错误
            error_msg = openai_error_msg(e)
            print(f"Server error: {e.status_code} - {error_msg}")
            raise HTTPException(status_code=e.status_code, detail=error_msg)

        except OpenAIError as e:
            # 处理其他 OpenAI 相关错误
            error_msg = openai_error_msg(e)
            print(f"OpenAI error: {str(e)}")
            raise HTTPException(status_code=500, detail=error_msg)

    @router.get(f"/models")
    async def models(api_Name=api_Name):
        model_dict = {"object": "list", "data": []}
        model_list = config[api_Name]["models"]
        for model in model_list:
            model_dict["data"].append(
                {
                    "id": model.get("id", "No Model ID"),
                    "name": model.get("name", "No Model Name"),
                    "object": "model",
                    "owned_by": api_Name,
                }
            )
        return model_dict

    app.include_router(router, prefix=f"/{api_Name}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4848)
