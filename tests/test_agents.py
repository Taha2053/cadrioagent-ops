import asyncio
import json
import httpx
from fastapi.testclient import TestClient
from agents.schemas import Report, audit
from agents.graph import build_graph
from llm_server.main import create_app
from llm_server.settings import Settings

QUOTE = "A heart sound recording cannot establish a clinical diagnosis alone."

def report(**changes):
    data = {"summary":"The acoustic pattern requires clinical interpretation.",
        "limitations":"PCG alone cannot establish a diagnosis; this is research output.",
        "citations":[{"source_id":"source-1","quote":QUOTE}],
        "next_step":"Review this recording and evidence with a qualified clinician."}
    data.update(changes)
    return Report.model_validate(data)


def test_audit_rejects_fabricated_evidence_and_dosing():
    sources = [{"id":"source-1", "text":QUOTE}]
    assert audit(report(), sources) == []
    assert audit(report(citations=[{"source_id":"invented","quote":QUOTE}]), sources)
    assert audit(report(citations=[{"source_id":"source-1","quote":"An invented quote that is not in the source."}]), sources)
    assert audit(report(summary="Start taking 20 mg every morning."), sources)


def test_deepseek_never_receives_unconsented_requests():
    contacted = []
    def handler(request):
        contacted.append(request)
        return httpx.Response(200,json={})
    settings=Settings(provider="deepseek",ollama_base_url="http://local",ollama_model="mistral:7b-instruct",
        deepseek_base_url="https://external.test",deepseek_api_key="test-key-not-real",deepseek_model="deepseek-chat",request_timeout_seconds=5)
    with TestClient(create_app(settings,httpx.MockTransport(handler))) as client:
        response=client.post("/v1/chat/completions",json={"messages":[{"role":"user","content":"sensitive prompt"}],"stream":True})
    assert response.status_code == 403
    assert not contacted


def test_graph_abstains_without_evidence_and_does_not_call_llm():
    calls=[]
    async def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200,json={"classification":"normal","probability_abnormal":.1,"provenance":{}})
    class EmptyEvidence:
        def search(self, query): return []
    async def scenario():
        async def emit(event,data): pass
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await build_graph(client,EmptyEvidence(),emit).ainvoke({"content":b"wave","prompt":"", "external_consent":False})
    result=asyncio.run(scenario())
    assert result["status"] == "abstained"
    assert result["report"] is None
    assert len(calls)==1 and calls[0].endswith("/predict")
