from rag_engine.core import chunks
from evaluation.run import retrieval


def test_chunks_cover_tail_without_empty_loop():
    text=" ".join(f"word{i}" for i in range(300))
    pieces=list(chunks(text,size=200,overlap=30))
    assert pieces[0].startswith("word0 ")
    assert pieces[-1].endswith("word299")
    assert all(0 < len(piece) <= 200 for piece in pieces)
    assert all(word in " ".join(pieces) for word in text.split())


def test_retrieval_scores_use_adjudicated_ids_not_document_count():
    class Search:
        def search(self,query,limit): return [{"id":"noise"},{"id":"relevant"}]
    scores=retrieval([{"query":"question","relevant_ids":["relevant","missing"]}],Search(),k=4)
    assert scores=={"recall_at_k":.5,"precision_at_k":.25,"mrr":.5}
