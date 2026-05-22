from __future__ import annotations

import os
from typing import Any

import httpx


CONTENT_FABRIC_API_URL = os.getenv(
    "CONTENT_FABRIC_API_URL", "http://127.0.0.1:8099"
)


async def generate_image(prompt: str, provider: str = "placeholder") -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{CONTENT_FABRIC_API_URL}/api/spike/generate-image",
            json={"prompt": prompt, "provider": provider},
        )
        response.raise_for_status()
        return response.json()


async def generate_video(
    image_job_id: str,
    prompt: str,
    provider: str = "grok",
    aspect_ratio: str = "1:1",
    seconds: int = 6,
) -> dict[str, Any]:
    # Video generation polls the provider for minutes — give it room.
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            f"{CONTENT_FABRIC_API_URL}/api/spike/generate-video",
            json={
                "image_job_id": image_job_id,
                "prompt": prompt,
                "provider": provider,
                "aspect_ratio": aspect_ratio,
                "seconds": seconds,
            },
        )
        response.raise_for_status()
        return response.json()
