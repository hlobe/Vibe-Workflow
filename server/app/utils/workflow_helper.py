import os
import logging
from pathlib import Path
from typing import Optional

import httpx
from fastapi import HTTPException

from .content_fabric_client import (
    CONTENT_FABRIC_API_URL,
    generate_image as content_fabric_generate_image,
    generate_video as content_fabric_generate_video,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MU_API_KEY = os.getenv("MU_API_KEY")
CONTENT_FABRIC_WORKFLOW_ID = "content-fabric-spike"
CONTENT_FABRIC_MODEL_ID = "content-fabric-placeholder"
CONTENT_FABRIC_NODE_ID = "generate-image"
CONTENT_FABRIC_VIDEO_MODEL_ID = "content-fabric-grok-video"
CONTENT_FABRIC_VIDEO_NODE_ID = "generate-video"
CONTENT_FABRIC_RUNS: dict[str, dict] = {}

CONTENT_FABRIC_IMAGE_PROVIDERS = ("placeholder", "flow", "gpt", "grok")
CONTENT_FABRIC_VIDEO_PROVIDERS = ("grok",)


def _content_fabric_workflow_summary() -> dict:
    return {
        "id": CONTENT_FABRIC_WORKFLOW_ID,
        "workflow_id": CONTENT_FABRIC_WORKFLOW_ID,
        "name": "Content Fabric: Image → Video",
        "category": "Content Fabric",
        "thumbnail": None,
        "updated_at": "2026-05-22T00:00:00Z",
        "created_at": "2026-05-22T00:00:00Z",
    }


def _content_fabric_workflow_def() -> dict:
    return {
        **_content_fabric_workflow_summary(),
        "run_id": "content-fabric-spike-run",
        "is_owner": True,
        "is_published": False,
        "is_template": False,
        "show_temp_button": False,
        "data": {
            "nodes": [
                {
                    "id": CONTENT_FABRIC_NODE_ID,
                    "category": "image",
                    "model": CONTENT_FABRIC_MODEL_ID,
                    "position": {"x": 0, "y": 100},
                    "input_params": {"prompt": "dragon fly", "provider": "placeholder"},
                    "output_params": {"outputs": [], "resultUrl": None},
                },
                {
                    "id": CONTENT_FABRIC_VIDEO_NODE_ID,
                    "category": "video",
                    "model": CONTENT_FABRIC_VIDEO_MODEL_ID,
                    "position": {"x": 360, "y": 100},
                    "input_params": {"image_job_id": "", "prompt": "slow cinematic push-in", "provider": "grok"},
                    "output_params": {"outputs": [], "resultUrl": None},
                },
            ]
        },
        "edges": [
            {
                "id": f"{CONTENT_FABRIC_NODE_ID}->{CONTENT_FABRIC_VIDEO_NODE_ID}",
                "source": CONTENT_FABRIC_NODE_ID,
                "target": CONTENT_FABRIC_VIDEO_NODE_ID,
            }
        ],
        "run_history": {},
    }


def _content_fabric_node_schemas() -> dict:
    return {
        "categories": {
            "image": {
                "models": {
                    CONTENT_FABRIC_MODEL_ID: {
                        "name": "Content Fabric Placeholder",
                        "description": "Generate an image through the local Content Fabric API.",
                        "input_schema": {
                            "schemas": {
                                "input_data": {
                                    "type": "object",
                                    "required": ["prompt"],
                                    "properties": {
                                        "prompt": {
                                            "type": "string",
                                            "title": "Prompt",
                                            "name": "prompt",
                                            "field": "text",
                                            "description": "Text prompt describing the image.",
                                            "default": "dragon fly",
                                        },
                                        "provider": {
                                            "type": "string",
                                            "title": "Provider",
                                            "name": "provider",
                                            "enum": list(CONTENT_FABRIC_IMAGE_PROVIDERS),
                                            "description": "Image provider.",
                                            "default": "placeholder",
                                        },
                                    },
                                }
                            }
                        },
                    }
                }
            },
            "video": {
                "models": {
                    CONTENT_FABRIC_VIDEO_MODEL_ID: {
                        "name": "Content Fabric Grok Video",
                        "description": "Animate an upstream image into a video through the local Content Fabric API.",
                        "input_schema": {
                            "schemas": {
                                "input_data": {
                                    "type": "object",
                                    "required": ["image_job_id", "prompt"],
                                    "properties": {
                                        "image_job_id": {
                                            "type": "string",
                                            "title": "Image Job ID",
                                            "name": "image_job_id",
                                            "field": "text",
                                            "description": "Job id of the upstream image to animate.",
                                            "default": "",
                                        },
                                        "prompt": {
                                            "type": "string",
                                            "title": "Prompt",
                                            "name": "prompt",
                                            "field": "text",
                                            "description": "Motion prompt describing the animation.",
                                            "default": "slow cinematic push-in",
                                        },
                                        "provider": {
                                            "type": "string",
                                            "title": "Provider",
                                            "name": "provider",
                                            "enum": list(CONTENT_FABRIC_VIDEO_PROVIDERS),
                                            "description": "Video provider.",
                                            "default": "grok",
                                        },
                                    },
                                }
                            }
                        },
                    }
                }
            },
        }
    }


def _is_content_fabric_workflow(workflow_id: str) -> bool:
    return workflow_id == CONTENT_FABRIC_WORKFLOW_ID


def _is_content_fabric_node(workflow_id: str, node_id: str) -> bool:
    return _is_content_fabric_workflow(workflow_id) or node_id in (
        CONTENT_FABRIC_NODE_ID,
        CONTENT_FABRIC_VIDEO_NODE_ID,
    )


def _extract_prompt(payload: dict) -> str | None:
    return (
        payload.get("prompt")
        or payload.get("inputs", {}).get("prompt")
        or payload.get("params", {}).get("prompt")
    )


def _extract_image_job_id(payload: dict) -> str | None:
    return (
        payload.get("image_job_id")
        or payload.get("inputs", {}).get("image_job_id")
        or payload.get("params", {}).get("image_job_id")
    )


def _extract_provider(payload: dict) -> str:
    provider = (
        payload.get("provider")
        or payload.get("inputs", {}).get("provider")
        or payload.get("params", {}).get("provider")
        or "placeholder"
    )
    if provider not in CONTENT_FABRIC_IMAGE_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"provider must be one of {', '.join(CONTENT_FABRIC_IMAGE_PROVIDERS)}",
        )
    return provider


def _extract_video_provider(payload: dict) -> str:
    provider = (
        payload.get("provider")
        or payload.get("inputs", {}).get("provider")
        or payload.get("params", {}).get("provider")
        or "grok"
    )
    if provider not in CONTENT_FABRIC_VIDEO_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"video provider must be one of {', '.join(CONTENT_FABRIC_VIDEO_PROVIDERS)}",
        )
    return provider


def _absolute_content_fabric_url(url: str | None) -> str | None:
    if not url:
        return None
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"{CONTENT_FABRIC_API_URL.rstrip('/')}/{url.lstrip('/')}"


def _record_content_fabric_run(node_id: str, job: dict) -> dict:
    artifacts = job.get("artifacts") or []
    first_artifact = artifacts[0] if artifacts else {}
    kind = first_artifact.get("kind", "image")
    media_url = _absolute_content_fabric_url(first_artifact.get("url"))
    output_type = "video_url" if kind == "video" else "image_url"
    media_key = "video" if kind == "video" else "image"
    run_id = job.get("id")
    latest = {
        "id": run_id,
        "node_run_id": run_id,
        "status": job.get("status", "succeeded"),
        "started_at": job.get("created_at"),
        "finished_at": job.get("updated_at"),
        "result": {
            "id": run_id,
            "outputs": [
                {
                    "type": output_type,
                    "value": media_url,
                }
            ],
        },
    }
    CONTENT_FABRIC_RUNS[run_id] = {"nodes": {node_id: [latest]}}
    return {
        "run_id": run_id,
        "id": run_id,
        "status": job.get("status"),
        "node_id": node_id,
        "outputs": {
            media_key: media_url,
            "job": job,
        },
    }

async def get_api_key():
    api_key = MU_API_KEY
    if not api_key:
        raise HTTPException(status_code=400, detail="Setup MU_API_KEY in .env to be able to use Workflow")
    return api_key

async def proxy_request_helper(method: str, url: str, payload: Optional[dict] = None):
    api_key = await get_api_key()
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }

    async with httpx.AsyncClient() as client:
        try:
            if method.upper() == "GET":
                response = await client.get(url, headers=headers, timeout=60.0)
            elif method.upper() == "POST":
                response = await client.post(url, json=payload, headers=headers, timeout=60.0)
            elif method.upper() == "DELETE":
                response = await client.delete(url, headers=headers, timeout=60.0)
            else:
                raise HTTPException(status_code=405, detail=f"Method {method} not supported in proxy")

        except httpx.RequestError as e:
            logger.error(f"HTTPExt Request Error for {method} {url}: {e}")
            raise HTTPException(status_code=500, detail=f"Error contacting remote server: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in proxy_request_helper for {method} {url}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

    try:
        if response.content:
            resp_json = response.json()
        else:
            resp_json = {}
    except ValueError:
        resp_json = {"detail": response.text or "Unknown error from remote server"}

    if response.status_code == 200:
        return resp_json
    else:
        error_detail = resp_json.get("detail", "Something went wrong")
        logger.warning(f"Remote server returned {response.status_code}: {error_detail}")
        raise HTTPException(status_code=response.status_code, detail=error_detail)

async def create_or_update_workflow(payload: dict):
    if _is_content_fabric_workflow(payload.get("workflow_id")):
        return {
            "workflow_id": CONTENT_FABRIC_WORKFLOW_ID,
            "run_id": "content-fabric-spike-run",
            "status": "saved",
        }

    url = "https://api.muapi.ai/workflow/create"
    return await proxy_request_helper("POST", url, payload)

async def get_node_schemas_helper(workflow_id: str):
    if _is_content_fabric_workflow(workflow_id):
        return _content_fabric_node_schemas()

    url = f"https://api.muapi.ai/workflow/{workflow_id}/node-schemas"
    return await proxy_request_helper("GET", url)

async def get_api_node_schemas_helper(workflow_id: str):
    url = f"https://api.muapi.ai/workflow/{workflow_id}/api-node-schemas"
    return await proxy_request_helper("GET", url)

async def get_workflow_def_helper(workflow_id: str):
    if _is_content_fabric_workflow(workflow_id):
        return _content_fabric_workflow_def()

    url = f"https://api.muapi.ai/workflow/get-workflow-def/{workflow_id}"
    return await proxy_request_helper("GET", url)

async def get_workflow_defs_helper():
    local_workflow = _content_fabric_workflow_summary()
    if not MU_API_KEY:
        return [local_workflow]

    url = "https://api.muapi.ai/workflow/get-workflow-defs"
    remote_workflows = await proxy_request_helper("GET", url)
    if isinstance(remote_workflows, list):
        return [local_workflow, *remote_workflows]
    return [local_workflow]

async def delete_workflow_def_by_id(workflow_id: str):
    url = f"https://api.muapi.ai/workflow/delete-workflow-def/{workflow_id}"
    return await proxy_request_helper("DELETE", url)

async def update_workflow_name_helper(workflow_id: str, payload: dict):
    url = f"https://api.muapi.ai/workflow/update-name/{workflow_id}"
    return await proxy_request_helper("POST", url, payload)

async def run_workflow_helper(workflow_id: str, payload: dict):
    if _is_content_fabric_workflow(workflow_id):
        return {"run_id": "content-fabric-spike-run", "status": "ready"}

    url = f"https://api.muapi.ai/workflow/{workflow_id}/run"
    return await proxy_request_helper("POST", url, payload)

async def get_run_status_helper(run_id: str):
    if run_id in CONTENT_FABRIC_RUNS:
        return CONTENT_FABRIC_RUNS[run_id]
    if run_id.startswith("content-fabric"):
        raise HTTPException(status_code=404, detail="Run not found")

    url = f"https://api.muapi.ai/workflow/run/{run_id}/status"
    return await proxy_request_helper("GET", url)

async def run_node_helper(workflow_id: str, node_id: str, payload: dict):
    if _is_content_fabric_node(workflow_id, node_id):
        prompt = _extract_prompt(payload)
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt is required")

        if node_id == CONTENT_FABRIC_VIDEO_NODE_ID:
            image_job_id = _extract_image_job_id(payload)
            if not image_job_id:
                raise HTTPException(
                    status_code=400,
                    detail="image_job_id is required (connect the image node's output)",
                )
            video_provider = _extract_video_provider(payload)
            job = await content_fabric_generate_video(
                image_job_id=image_job_id, prompt=prompt, provider=video_provider
            )
        else:
            provider = _extract_provider(payload)
            job = await content_fabric_generate_image(prompt=prompt, provider=provider)
        return _record_content_fabric_run(node_id, job)

    url = f"https://api.muapi.ai/workflow/{workflow_id}/node/{node_id}/run"
    return await proxy_request_helper("POST", url, payload)

async def publish_workflow_helper(workflow_id: str, payload: dict):
    url = f"https://api.muapi.ai/workflow/workflow/{workflow_id}/publish"
    return await proxy_request_helper("POST", url, payload)

async def template_workflow_helper(workflow_id: str, payload: dict):
    url = f"https://api.muapi.ai/workflow/workflow/{workflow_id}/template"
    return await proxy_request_helper("POST", url, payload)

async def cloudfront_signed_url_helper(payload: dict):
    url = "https://api.muapi.ai/workflow/cloudfront-signed-url"
    return await proxy_request_helper("POST", url, payload)

async def generate_thumbnail_helper(workflow_id: str, payload: dict):
    url = f"https://api.muapi.ai/workflow/{workflow_id}/thumbnail"
    return await proxy_request_helper("POST", url, payload)

async def get_file_upload_url_helper(params: dict):
    import urllib.parse
    query_string = urllib.parse.urlencode(params)
    url = f"https://api.muapi.ai/app/get_file_upload_url?{query_string}"
    return await proxy_request_helper("GET", url)

async def get_workflow_last_run(workflow_id: str):
    url = f"https://api.muapi.ai/workflow/get-workflow-last-run/{workflow_id}"
    return await proxy_request_helper("GET", url)

async def architect_workflow_helper(payload: dict):
    url = "https://api.muapi.ai/workflow/architect"
    return await proxy_request_helper("POST", url, payload)

async def poll_architect_result_helper(id: str):
    url = f"https://api.muapi.ai/workflow/poll-architect/{id}/result"
    return await proxy_request_helper("GET", url)

async def delete_node_run_by_id_helper(node_run_id: str):
    url = f"https://api.muapi.ai/workflow/node-run/{node_run_id}"
    return await proxy_request_helper("DELETE", url)

async def update_workflow_category_helper(workflow_id: str, payload: dict):
    url = f"https://api.muapi.ai/workflow/update-category/{workflow_id}"
    return await proxy_request_helper("POST", url, payload)

async def get_workflow_api_inputs_helper(workflow_id: str):
    url = f"https://api.muapi.ai/workflow/{workflow_id}/api-inputs"
    return await proxy_request_helper("GET", url)

async def execute_workflow_via_api_helper(workflow_id: str, payload: dict):
    url = f"https://api.muapi.ai/workflow/{workflow_id}/api-execute"
    return await proxy_request_helper("POST", url, payload)

async def get_workflow_api_outputs_helper(run_id: str):
    url = f"https://api.muapi.ai/workflow/run/{run_id}/api-outputs"
    return await proxy_request_helper("GET", url)
