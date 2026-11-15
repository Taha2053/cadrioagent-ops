import httpx
from fastapi.testclient import TestClient
from api.main import Settings, create_app
from tests.test_audio_engine import wav_bytes


def test_invalid_upload_and_key_never_reach_upstream():
    calls=[]
    def handler(request):
        calls.append(request)
        return httpx.Response(200,json={})
    with TestClient(create_app(Settings(api_key="unit-key"),httpx.MockTransport(handler))) as client:
        assert client.post("/analyze_pcg",files={"file":("x.wav",wav_bytes())}).status_code==401
        assert client.post("/analyze_pcg",headers={"X-API-Key":"unit-key"},files={"file":("x.wav",b"fake")}).status_code==400
    assert calls==[]


def test_gateway_forwards_consent_and_preserves_analysis():
    def handler(request):
        assert b'name="external_consent"\r\n\r\ntrue' in request.content
        return httpx.Response(200,json={"status":"abstained","report":None,"sources":[]})
    with TestClient(create_app(Settings(),httpx.MockTransport(handler))) as client:
        result=client.post("/analyze_pcg",files={"file":("x.wav",wav_bytes())},data={"external_consent":"true"})
    assert result.status_code==200 and result.json()["status"]=="abstained"


def test_gateway_hides_sensitive_upstream_error():
    def handler(request): return httpx.Response(500,text="SECRET clinical context")
    with TestClient(create_app(Settings(),httpx.MockTransport(handler))) as client:
        result=client.post("/analyze_pcg",files={"file":("x.wav",wav_bytes())})
    assert result.status_code==502
    assert "SECRET" not in result.text


def test_readiness_does_not_confuse_liveness_with_loaded_model():
    def handler(request): return httpx.Response(200,json={"status":"ok","model_loaded":False})
    with TestClient(create_app(Settings(),httpx.MockTransport(handler))) as client:
        result=client.get("/status")
    assert result.status_code==503
    assert result.json()["ready"] is False


def test_overload_rejects_before_buffering_more_recordings():
    import asyncio
    from api.main import MAX_ACTIVE_ANALYSES

    async def scenario():
        entered = asyncio.Queue()
        release = asyncio.Event()
        async def handler(request):
            entered.put_nowait(True)
            await release.wait()
            return httpx.Response(200, json={"status": "abstained", "report": None})
        app = create_app(Settings(), httpx.MockTransport(handler))
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway.test") as client:
                async def upload():
                    return await client.post("/analyze_pcg", files={"file": ("heart.wav", wav_bytes(), "audio/wav")})
                active = [asyncio.create_task(upload()) for _ in range(MAX_ACTIVE_ANALYSES)]
                try:
                    for _ in active:
                        await asyncio.wait_for(entered.get(), timeout=5)
                    overloaded = await upload()
                    assert overloaded.status_code == 429
                    assert overloaded.headers["Retry-After"] == "2"
                finally:
                    release.set()
                    replies = await asyncio.gather(*active)
                assert all(reply.status_code == 200 for reply in replies)
                assert (await upload()).status_code == 200
    asyncio.run(scenario())
