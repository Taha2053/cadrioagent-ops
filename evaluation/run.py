"""Evaluate labeled retrieval or completed reports; never score generated fixtures as real performance."""
import argparse
import json
from pathlib import Path
from agents.schemas import Report, audit
from rag_engine.core import Retriever


def retrieval(rows, retriever, k=4):
    scores = []
    for row in rows:
        relevant = set(row["relevant_ids"])
        if not relevant:
            raise ValueError("Every retrieval query requires adjudicated relevant_ids")
        hits = [r["id"] for r in retriever.search(row["query"], limit=k)]
        positions = [i+1 for i, identifier in enumerate(hits) if identifier in relevant]
        scores.append({"recall_at_k": len(set(hits)&relevant)/len(relevant),
            "precision_at_k": len(set(hits)&relevant)/k,
            "mrr": 1/min(positions) if positions else 0})
    return {key: sum(r[key] for r in scores)/len(scores) for key in scores[0]}


def reports(rows):
    schema_pass = citation_pass = safe_abstentions = total_abstentions = 0
    for row in rows:
        payload = row["result"]
        if payload.get("report"):
            try:
                report = Report.model_validate(payload["report"])
                schema_pass += 1
                citation_pass += int(not audit(report, payload.get("sources", [])))
            except ValueError:
                pass
        if row.get("expected_abstention"):
            total_abstentions += 1
            safe_abstentions += int(payload.get("status") in {"abstained", "rejected"} and not payload.get("report"))
    return {"schema_pass_rate": schema_pass/len(rows), "citation_and_rule_safety_pass_rate": citation_pass/len(rows),
        **({"expected_abstention_rate": safe_abstentions/total_abstentions} if total_abstentions else {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["retrieval", "reports"])
    parser.add_argument("dataset", help="JSONL with human-adjudicated relevance or completed analysis results")
    parser.add_argument("--output", default="vault/evaluations/latest.json")
    parser.add_argument("--tracking-uri", default="sqlite:///vault/mlflow.db")
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.dataset).read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Evaluation dataset is empty")
    scores = retrieval(rows, Retriever()) if args.kind == "retrieval" else reports(rows)
    import mlflow
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("cardioagent-evaluation")
    with mlflow.start_run(run_name=args.kind):
        mlflow.log_metrics(scores)
        mlflow.log_param("cases", len(rows))
        # Never upload clinical prompts or full reports to telemetry.
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": args.kind, "cases": len(rows), "metrics": scores}, indent=2))
        mlflow.log_artifact(str(path))
    print(json.dumps(scores, indent=2))

if __name__ == "__main__":
    main()
