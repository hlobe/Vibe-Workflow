from __future__ import annotations

import asyncio

import pytest

from app.utils import workflow_helper


def run(coro):
    return asyncio.run(coro)


def test_content_fabric_workflow_is_listed():
    workflows = run(workflow_helper.get_workflow_defs_helper())

    content_workflow = next(
        item for item in workflows if item["id"] == "content-fabric-spike"
    )
    assert content_workflow["name"] == "Content Fabric: Image → Video"
    assert content_workflow["category"] == "Content Fabric"


def test_content_fabric_workflow_definition_has_image_and_video_nodes():
    workflow = run(workflow_helper.get_workflow_def_helper("content-fabric-spike"))

    assert workflow["workflow_id"] == "content-fabric-spike"
    nodes = workflow["data"]["nodes"]
    assert len(nodes) == 2

    image_node = next(n for n in nodes if n["id"] == "generate-image")
    assert image_node["category"] == "image"
    assert image_node["input_params"]["prompt"] == "dragon fly"
    assert image_node["input_params"]["provider"] == "placeholder"

    video_node = next(n for n in nodes if n["id"] == "generate-video")
    assert video_node["category"] == "video"

    assert len(workflow["edges"]) == 1
    assert workflow["run_history"] == {}


def test_content_fabric_node_schema_has_prompt_model():
    schemas = run(workflow_helper.get_node_schemas_helper("content-fabric-spike"))

    model = schemas["categories"]["image"]["models"]["content-fabric-placeholder"]
    assert model["name"] == "Content Fabric Placeholder"
    input_data = model["input_schema"]["schemas"]["input_data"]
    assert input_data["required"] == ["prompt"]
    assert input_data["properties"]["prompt"]["type"] == "string"
    assert input_data["properties"]["prompt"]["field"] == "text"


def test_injected_models_appear_in_non_content_fabric_schemas(monkeypatch):
    monkeypatch.setattr(workflow_helper, "MU_API_KEY", None)

    schemas = run(workflow_helper.get_node_schemas_helper("some-other-workflow"))

    assert set(schemas["categories"]["image"]["models"]) == {
        "content-fabric-flow",
        "content-fabric-gpt",
        "content-fabric-grok",
    }
    assert set(schemas["categories"]["video"]["models"]) == {
        "content-fabric-grok-video",
    }


def test_content_fabric_workflow_save_stays_local():
    response = run(
        workflow_helper.create_or_update_workflow(
            {
                "workflow_id": "content-fabric-spike",
                "name": "Content Fabric: Generate Image",
                "data": {"nodes": []},
                "edges": [],
            }
        )
    )

    assert response == {
        "workflow_id": "content-fabric-spike",
        "run_id": "content-fabric-spike-run",
        "status": "saved",
    }


def test_content_fabric_run_node_creates_pollable_status(monkeypatch):
    async def fake_generate_image(prompt, provider="placeholder"):
        assert prompt == "STEX capsule from UI"
        assert provider == "placeholder"
        return {
            "id": "job-123",
            "status": "succeeded",
            "prompt": prompt,
            "provider": provider,
            "artifacts": [
                {
                    "kind": "image",
                    "url": "/artifacts/job-123.svg",
                    "path": "spike-artifacts/job-123.svg",
                    "content_type": "image/svg+xml",
                }
            ],
        }

    monkeypatch.setattr(workflow_helper, "content_fabric_generate_image", fake_generate_image)

    response = run(
        workflow_helper.run_node_helper(
            "content-fabric-spike",
            "generate-image",
            {
                "params": {"prompt": "STEX capsule from UI"},
                "node_id": "AI Image",
            },
        )
    )

    assert response["run_id"] == "job-123"
    status = run(workflow_helper.get_run_status_helper("job-123"))
    latest = status["nodes"]["generate-image"][-1]
    assert latest["status"] == "succeeded"
    assert latest["result"]["outputs"][0]["type"] == "image_url"
    assert (
        latest["result"]["outputs"][0]["value"]
        == "http://127.0.0.1:8099/artifacts/job-123.svg"
    )


def test_run_node_routes_injected_image_model(monkeypatch):
    calls = []

    async def fake_generate_image(prompt, provider="placeholder"):
        calls.append({"prompt": prompt, "provider": provider})
        return {
            "id": "job-gpt",
            "status": "succeeded",
            "prompt": prompt,
            "provider": provider,
            "artifacts": [
                {
                    "kind": "image",
                    "url": "/artifacts/job-gpt.png",
                    "path": "spike-artifacts/job-gpt.png",
                    "content_type": "image/png",
                }
            ],
        }

    monkeypatch.setattr(workflow_helper, "content_fabric_generate_image", fake_generate_image)

    response = run(
        workflow_helper.run_node_helper(
            "some-other-workflow",
            "native-image-node",
            {
                "model": "content-fabric-gpt",
                "params": {"prompt": "native graph prompt"},
            },
        )
    )

    assert calls == [{"prompt": "native graph prompt", "provider": "gpt"}]
    assert response["run_id"] == "job-gpt"


def test_run_node_routes_injected_video_model(monkeypatch):
    calls = []

    async def fake_generate_video(image_job_id, prompt, provider="grok"):
        calls.append(
            {"image_job_id": image_job_id, "prompt": prompt, "provider": provider}
        )
        return {
            "id": "job-video",
            "status": "succeeded",
            "prompt": prompt,
            "provider": provider,
            "artifacts": [
                {
                    "kind": "video",
                    "url": "/artifacts/job-video.mp4",
                    "path": "spike-artifacts/job-video.mp4",
                    "content_type": "video/mp4",
                }
            ],
        }

    monkeypatch.setattr(workflow_helper, "content_fabric_generate_video", fake_generate_video)

    response = run(
        workflow_helper.run_node_helper(
            "some-other-workflow",
            "native-video-node",
            {
                "model": "content-fabric-grok-video",
                "params": {
                    "prompt": "slow cinematic push-in",
                    "image_job_id": "job-gpt",
                },
            },
        )
    )

    assert calls == [
        {
            "image_job_id": "job-gpt",
            "prompt": "slow cinematic push-in",
            "provider": "grok",
        }
    ]
    assert response["run_id"] == "job-video"


def test_content_fabric_unknown_run_returns_404():
    with pytest.raises(Exception) as exc_info:
        run(workflow_helper.get_run_status_helper("content-fabric-missing-run"))

    assert getattr(exc_info.value, "status_code", None) == 404
    assert getattr(exc_info.value, "detail", None) == "Run not found"
