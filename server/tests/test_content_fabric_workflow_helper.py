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
    assert content_workflow["name"] == "Content Fabric: Generate Image"
    assert content_workflow["category"] == "Content Fabric"


def test_content_fabric_workflow_definition_has_image_node():
    workflow = run(workflow_helper.get_workflow_def_helper("content-fabric-spike"))

    assert workflow["workflow_id"] == "content-fabric-spike"
    assert workflow["data"]["nodes"] == [
        {
            "id": "generate-image",
            "category": "image",
            "model": "content-fabric-placeholder",
            "position": {"x": 0, "y": 100},
            "input_params": {"prompt": "dragon fly"},
            "output_params": {"outputs": [], "resultUrl": None},
        }
    ]
    assert workflow["edges"] == []
    assert workflow["run_history"] == {}


def test_content_fabric_node_schema_has_prompt_model():
    schemas = run(workflow_helper.get_node_schemas_helper("content-fabric-spike"))

    model = schemas["categories"]["image"]["models"]["content-fabric-placeholder"]
    assert model["name"] == "Content Fabric Placeholder"
    input_data = model["input_schema"]["schemas"]["input_data"]
    assert input_data["required"] == ["prompt"]
    assert input_data["properties"]["prompt"]["type"] == "string"
    assert input_data["properties"]["prompt"]["field"] == "text"


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


def test_content_fabric_unknown_run_returns_404():
    with pytest.raises(Exception) as exc_info:
        run(workflow_helper.get_run_status_helper("content-fabric-missing-run"))

    assert getattr(exc_info.value, "status_code", None) == 404
    assert getattr(exc_info.value, "detail", None) == "Run not found"
