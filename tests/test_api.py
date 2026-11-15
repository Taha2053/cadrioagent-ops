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
