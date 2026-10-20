# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Exercise configured MiniMax models through the real SDK HTTP boundary."""

import json
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
import yaml
from openai import AsyncOpenAI


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", [
    "MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "custom-minimax-model",
])
@pytest.mark.parametrize("api_base", ["https://api.minimax.io/v1", "https://api.minimaxi.com/v1"])
@pytest.mark.parametrize("stream", [False, True])
async def test_minimax_default_model_http_payload(monkeypatch, model_name, api_base, stream):
    from openjiuwen.core.foundation.llm.model_clients.openai_model_client import (
        OpenAIModelClient,
    )

    from jiuwenswarm.server.runtime.session.kv_cache_affinity_lifecycle import (
        _build_default_model,
    )

    template = Path(__file__).parents[2] / "jiuwenswarm" / "resources" / "config.yaml"
    defaults = yaml.safe_load(template.read_text())["models"]["defaults"]
    entry = deepcopy(next(item for item in defaults if
                          item.get("provider_metadata", {}).get("provider") == "MiniMax"))
    entry["is_default"] = True
    entry["model_client_config"].update(
        model_name=model_name, api_base=api_base, api_key="fixture-key",
        use_shared_llm_http_client=False,
    )
    messages = [
        {"role": "user", "content": "Look up a value."},
        {"role": "assistant", "content": None, "reasoning_content": "Use lookup.",
         "tool_calls": [{"id": "call-1", "type": "function", "function": {
             "name": "lookup", "arguments": "{}",
         }}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "A value."},
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}},
        ]},
    ]
    tools = [{"type": "function", "function": {
        "name": "lookup", "description": "Look up a value.",
        "parameters": {"type": "object", "properties": {}},
    }}]
    requests = []
    clients = []

    def respond(request):
        requests.append(request)
        response = {
            "id": "fixture", "object": "chat.completion", "created": 0,
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                         "finish_reason": "stop"}],
        }
        if stream:
            response["object"] = "chat.completion.chunk"
            response["choices"][0]["delta"] = response["choices"][0].pop("message")
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text=f"data: {json.dumps(response)}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json=response)

    def build_client(self, timeout=None):
        client = AsyncOpenAI(
            api_key=self.model_client_config.api_key,
            base_url=self.model_client_config.api_base,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        clients.append(client)
        return client

    monkeypatch.setattr(OpenAIModelClient, "_build_async_openai_client", build_client)
    model = _build_default_model({"models": {"defaults": [entry]}})
    try:
        if stream:
            chunks = [chunk async for chunk in model.stream(messages, tools=tools)]
            assert chunks
        else:
            assert (await model.invoke(messages, tools=tools)).content == "OK"
    finally:
        for client in clients:
            await client.close()
    assert len(requests) == 1
    assert str(requests[0].url) == api_base + "/chat/completions"
    payload = json.loads(requests[0].content)
    assert payload["model"] == model_name
    assert payload["stream"] is stream
    assert payload["messages"] == messages
    assert payload["tools"] == tools
    assert "service_tier" not in payload
    assert "provider_metadata" not in payload
    assert payload.get("reasoning_split") is not False
    assert payload.get("thinking", {}).get("type") != "disabled"
    assert payload.get("reasoning_effort") != "none"
