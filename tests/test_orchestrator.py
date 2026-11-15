"""Consumer-visible agent behavior: grounded final reports, rejection, and abstention."""
import json
import httpx
import pytest
from fastapi.testclient import TestClient
from agents import orchestrator
from tests.test_audio_engine import wav_bytes

QUOTE = "A heart sound recording cannot establish a clinical diagnosis alone."
SOURCE = {"id": "source-1", "text": QUOTE, "source": "test-guideline.pdf",
          "page": 1, "url": "https://example.test/guideline", "sha256": "unit-fixture"}
REPORT = {"summary": "The acoustic classification requires clinical interpretation.",
          "limitations": "PCG alone cannot establish a clinical diagnosis. This is research output.",
          "citations": [{"source_id": "source-1", "quote": QUOTE}],
          "next_step": "Review this recording and evidence with a qualified clinician."}


class Frames(httpx.AsyncByteStream):
    def __init__(self, text):
        self.text = text

    async def __aiter__(self):
        for piece in (self.text[:30], self.text[30:]):
            yield ("data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}) + "\n\n").encode()
        yield b"data: [DONE]\n\n"


class Evidence:
    class Connection:
        def close(self):
            pass
    client = Connection()

    def search(self, query):
        return [SOURCE]

    def ready(self):
        return {"ready": True, "points": 1}


def connect(client, handler):
    original = orchestrator.app.state.http
    client.portal.call(original.aclose)
    orchestrator.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("mode,expected", [("grounded", "ok"), ("forged", "rejected"), ("unconsented", "error"), ("private_question", "ok")])
def test_stream_emits_only_audited_authoritative_reports(monkeypatch, mode, expected):
    monkeypatch.setattr(orchestrator, "Retriever", Evidence)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    calls = []
    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/predict":
            return httpx.Response(200, json={"classification": "normal", "probability_abnormal": .1,
                                           "provenance": {"name": "decision_tree"}})
        if request.url.path == "/health":
            return httpx.Response(200, json={"provider": "deepseek" if mode == "private_question" else "ollama", "model": "test-model"})
        if mode == "unconsented":
            return httpx.Response(403, json={"detail": "Consent required"})
        report = json.loads(json.dumps(REPORT))
        if mode == "forged":
            report["citations"][0]["quote"] = "This invented quotation is not present in the retrieved evidence."
        payload = json.loads(request.content)
        assert payload["external_consent"] is (mode == "private_question")
        assert payload["response_schema"]["required"] == ["summary", "limitations", "citations", "next_step"]
        if mode == "private_question":
            assert "Jane Fixture" not in json.dumps(payload)
            assert "fixture@example.test" not in json.dumps(payload)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Frames(json.dumps(report)))

    with TestClient(orchestrator.app) as client:
        connect(client, handler)
        response = client.post("/analyze/stream", files={"file": ("heart.wav", wav_bytes(), "audio/wav")},
            data={"prompt": "Jane Fixture, fixture@example.test: what does this mean?",
                  "external_consent": str(mode == "private_question").lower()})
    assert response.status_code == 200
    events = []
    for frame in response.text.strip().split("\n\n"):
        lines = frame.splitlines()
        events.append((lines[0][7:], json.loads(lines[1][6:])))
    assert events[-1] == ("done", {})
    if expected == "error":
        assert any(event == "error" for event, _ in events)
        assert not any(event == "result" for event, _ in events)
    else:
        result = next(data for event, data in events if event == "result")
        assert result["status"] == expected
        assert result["audit"]["passed"] is (expected == "ok")
        assert (result["report"] is not None) is (expected == "ok")
        assert any(event == "token" and data["draft"] for event, data in events)
        assert "content" not in result
    assert calls[0] == "/predict"


def test_missing_evidence_abstains_without_generation(monkeypatch):
    class EmptyEvidence(Evidence):
        def search(self, query):
            return []
    monkeypatch.setattr(orchestrator, "Retriever", EmptyEvidence)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    calls = []
    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"classification": "abnormal", "probability_abnormal": .8, "provenance": {}})
    with TestClient(orchestrator.app) as client:
        connect(client, handler)
        response = client.post("/analyze", files={"file": ("heart.wav", wav_bytes(), "audio/wav")})
        invalid = client.post("/analyze", files={"file": ("heart.wav", b"not a WAV", "audio/wav")})
    assert response.json()["status"] == "abstained"
    assert response.json()["report"] is None
    assert invalid.status_code == 400
    assert calls == ["/predict"]
