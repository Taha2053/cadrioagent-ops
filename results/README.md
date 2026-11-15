# Measured results

These are observed experiments, not a completed five-model comparison or clinical validation.

## Classical benchmark

`classical-baseline.json` is the recorded MLflow run `13830edecb444fc5b836d97bb6216b92`.
Both algorithms used all 2,960 eligible recordings after source-label/SQI/duplicate filtering;
1,112 source rows were excluded. Acquisition/recording-group split: 1,776 training,
592 validation, 592 test. Test prevalence: 125 abnormal / 592 total (21.1%).
No verified patient mapping exists: this is **not patient-independent validation**.
Only the first five seconds of each recording are represented.

| Model | Test abnormal recall | Test specificity | Test abnormal precision | Test AUROC | Test AUPRC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 98.4% | 71.09% | 47.67% | 0.9088 | 0.6601 |
| RBF SVM | 76.8% | 95.07% | 80.67% | 0.9607 | 0.8188 |

The tree was selected on **validation**, not these test results: validation recall
95.2%, specificity 68.95%, above the preset 0.5 specificity gate. On test it produced
123 true positives, 2 false negatives, **135 false positives**, and 332 true negatives.
The SVM has much better specificity and precision; recall-first selection is a
research specification decision, not proof the tree is clinically preferable.

ONNX/Python maximum probability differences on the export verification batch:
2.14e-8 (tree), 2.44e-6 (SVM), both below 1e-4. Export parity is numerical consistency,
not an independent accuracy evaluation.

## Inference and retrieval smoke verification

On this Linux host, one warmup plus 20 sequential HTTP requests to the local audio
service, using an actual 8-second/2 kHz recording and the deployed tree, measured
p50 **5.33 ms**, p95 **6.83 ms**, mean 5.60 ms. This includes upload, canonical
preprocessing and prediction. It excludes LLM generation, cold start, container
networking, MLflow telemetry (disabled for this measurement), and concurrent load.
These numbers are machine-specific, not an SLA.

`evaluation/esc_smoke.jsonl` contains four assistant-curated literal lookups checked
against the downloaded ESC patient guideline. They are **not clinician-adjudicated**
or an independent medical QA benchmark. Against both the initial 32-chunk patient-only
index and the expanded 806-chunk professional/patient corpus, actual retrieval measured
Recall@4=1.0 and MRR=1.0; known-target Precision@4=0.25.
Each query has one annotated target, not an exhaustive relevance set, so that
precision is not a general estimate of evidence relevance. Do not tune on these
smoke queries and then claim independent retrieval performance.

A real browser run completed ONNX → Qdrant → local Mistral → deterministic audit,
with streamed draft tokens and an accepted final report. An earlier run produced
an incorrectly wrapped JSON object and was rejected; Ollama schema-constrained
generation fixed the producer rather than weakening the auditor. Browser checks
also exercised an actual 2 kHz upload, resampled preview playback, 390-pixel layout
without horizontal overflow, and axe WCAG A/AA checks with zero reported violations.
Automated accessibility checks do not replace keyboard/screen-reader review.

The same workflow also completed against the six Docker services at the user's
configured gateway port 8080, including actual live-model generation and an accepted
final audit. The inference image was inspected: neither TensorFlow nor PyTorch is
installed. Core verification passed 28 tests; the optional deep-export verification
separately passed all three CNN/ResNet/Transformer–CNN cases. Core test coverage
was 59%; CLI training/ingestion smoke runs are not counted as unit coverage.

## Outstanding evidence before quality claims

- Complete GPU training/tuning and held-out comparison of the three deep model families.
- Clinician-adjudicated retrieval/LLM safety and clinical-correctness evaluations.
- Verified patient identities or a genuinely independent external dataset.
- Probability calibration and confidence intervals on a separately reserved cohort.
- Realistic concurrent-load measurements, cold-start latency, and memory/VRAM profiling.
- DeepSeek live verification after a local key is configured, with per-request consent.

Never label code availability, a smoke test, or schema/quote matching as medical readiness.
