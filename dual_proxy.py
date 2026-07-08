import httpx
import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

load_dotenv(".env", override=True)
PROJECT_ID = os.getenv("PROJECT_ID")

app = FastAPI()

@app.post("/aistudio/v1/chat/completions")
async def ai_studio_proxy(request: Request):
    body = await request.json()
    if "ignore_eos" in body:
        del body["ignore_eos"]

    headers = dict(request.headers)
    if "host" in headers:
        del headers["host"]
    if "content-length" in headers:
        del headers["content-length"]

    url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

    client = httpx.AsyncClient()
    req = client.build_request("POST", url, json=body, headers=headers)
    rp_resp = await client.send(req, stream=True)

    return StreamingResponse(
        rp_resp.aiter_raw(),
        status_code=rp_resp.status_code,
        headers=dict(rp_resp.headers),
    )


@app.post("/vertexai/v1/chat/completions")
async def vertex_proxy(request: Request):
    body = await request.json()
    if "ignore_eos" in body:
        del body["ignore_eos"]

    if (
        "model" in body
        and isinstance(body["model"], str)
        and not body["model"].startswith("google/")
    ):
        body["model"] = f"google/{body['model']}"

    headers = dict(request.headers)
    if "host" in headers:
        del headers["host"]
    if "content-length" in headers:
        del headers["content-length"]

    # Forward to Vertex AI OpenAI-compatible endpoint
    url = f"https://aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/global/endpoints/openapi/chat/completions"
    
    client = httpx.AsyncClient()
    req = client.build_request("POST", url, json=body, headers=headers)
    rp_resp = await client.send(req, stream=True)

    if rp_resp.status_code == 401:
        # Read the error content
        error_content = await rp_resp.aread()
        print("--- GOOGLE ERROR RESPONSE ---")
        print(error_content.decode("utf-8"))

    return StreamingResponse(
        rp_resp.aiter_raw(),
        status_code=rp_resp.status_code,
        headers=dict(rp_resp.headers),
    )
