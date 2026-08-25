import httpx
import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

load_dotenv(".env", override=True)
PROJECT_ID = os.getenv("PROJECT_ID")

# Global async HTTP client with connection pooling
http_client: httpx.AsyncClient = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    limits = httpx.Limits(max_keepalive_connections=500, max_connections=2000, keepalive_expiry=30.0)
    timeout = httpx.Timeout(120.0, connect=10.0)
    http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
    yield
    if http_client is not None:
        await http_client.aclose()

app = FastAPI(lifespan=lifespan)

def get_headers(request_headers: dict, env_auth_key: str) -> dict:
    headers = dict(request_headers)
    for h in ("host", "content-length", "accept-encoding"):
        headers.pop(h, None)
    if "authorization" not in headers and "Authorization" not in headers:
        token = os.getenv(env_auth_key)
        if token:
            headers["authorization"] = f"Bearer {token}"
    return headers

async def stream_completions(resp: httpx.Response):
    """Transform chat completion SSE deltas into text completion SSE format for inference-perf."""
    try:
        async for line in resp.aiter_lines():
            if not line:
                continue
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                try:
                    payload = json.loads(line[6:])
                    choices = payload.get("choices", [])
                    if choices:
                        choices[0]["text"] = choices[0].get("delta", {}).get("content", "")
                    yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
                    continue
                except Exception:
                    pass
            yield (line + "\n\n").encode("utf-8")
    finally:
        await resp.aclose()

async def forward_request(
    request: Request,
    target_url: str,
    auth_env_key: str,
    is_geap: bool = False,
    is_completions: bool = False,
):
    body = await request.json()
    body.pop("ignore_eos", None)

    if is_completions:
        prompt = body.pop("prompt", "")
        body["messages"] = [{"role": "user", "content": prompt}]

    if is_geap:
        model = body.get("model")
        if isinstance(model, str) and not model.startswith("google/"):
            body["model"] = f"google/{model}"

    headers = get_headers(dict(request.headers), auth_env_key)
    req = http_client.build_request("POST", target_url, json=body, headers=headers)
    rp_resp = await http_client.send(req, stream=True)

    if rp_resp.status_code >= 400:
        error_bytes = await rp_resp.aread()
        await rp_resp.aclose()
        return StreamingResponse(
            iter([error_bytes]),
            status_code=rp_resp.status_code,
            media_type="application/json",
        )

    if body.get("stream", False):
        streamer = stream_completions(rp_resp) if is_completions else rp_resp.aiter_bytes()
        return StreamingResponse(
            streamer,
            status_code=rp_resp.status_code,
            media_type="text/event-stream",
        )
    else:
        content_bytes = await rp_resp.aread()
        await rp_resp.aclose()
        resp_data = json.loads(content_bytes.decode("utf-8"))
        if is_completions and "choices" in resp_data and len(resp_data["choices"]) > 0:
            msg = resp_data["choices"][0].get("message", {})
            resp_data["choices"][0]["text"] = msg.get("content", "")
        return JSONResponse(resp_data, status_code=rp_resp.status_code)

AI_STUDIO_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEAP_URL = f"https://aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/global/endpoints/openapi/chat/completions"

@app.post("/aistudio/v1/chat/completions")
async def ai_studio_chat(request: Request):
    return await forward_request(request, AI_STUDIO_URL, "API_KEY")

@app.post("/aistudio/v1/completions")
async def ai_studio_completions(request: Request):
    return await forward_request(request, AI_STUDIO_URL, "API_KEY", is_completions=True)

@app.post("/geap/v1/chat/completions")
async def geap_chat(request: Request):
    return await forward_request(request, GEAP_URL, "AUTH_TOKEN", is_geap=True)

@app.post("/geap/v1/completions")
async def geap_completions(request: Request):
    return await forward_request(request, GEAP_URL, "AUTH_TOKEN", is_geap=True, is_completions=True)
