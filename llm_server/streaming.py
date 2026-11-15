"""Translate provider streams to OpenAI-compatible SSE without buffering tokens."""
import json
import httpx


async def stream_chat(client, settings, body):
    local = settings.provider == "ollama"
    model = body.model or (settings.ollama_model if local else settings.deepseek_model)
    url = settings.ollama_base_url + "/api/chat" if local else settings.deepseek_base_url + "/chat/completions"
    payload = {"model": model, "messages": [m.model_dump() for m in body.messages], "stream": True}
    headers = {}
    if local:
        payload["options"] = {"temperature": body.temperature, "num_predict": body.max_tokens}
        if body.json_mode or body.response_schema:
            payload["format"] = body.response_schema or "json"
    else:
        headers["Authorization"] = f"Bearer {settings.deepseek_api_key}"
        payload.update(temperature=body.temperature, max_tokens=body.max_tokens)
        if body.json_mode:
            payload["response_format"] = {"type": "json_object"}
    try:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if local:
                    data = json.loads(line)
                    if data.get("error"):
                        raise ValueError("Provider generation failed")
                    text = data.get("message", {}).get("content", "")
                    if text:
                        yield "data: " + json.dumps({"choices": [{"delta": {"content": text}}]}) + "\n\n"
                elif line.startswith("data: "):
                    if line[6:] == "[DONE]":
                        break
                    yield line + "\n\n"
        yield "data: [DONE]\n\n"
    except (httpx.HTTPError, ValueError, KeyError):
        yield 'data: {"error":{"message":"LLM provider unavailable"}}\n\n'
