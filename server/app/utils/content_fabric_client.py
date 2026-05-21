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
