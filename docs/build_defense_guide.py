"""Build the study PDF: uv run --locked --group docs python docs/build_defense_guide.py.

Metrics and artifact sizes come from the committed benchmark, not invented examples.
The remaining explanations describe the active source modules, not legacy coursework.
"""
from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import reportlab
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "CardioAgent-Ops-Defense-Guide.pdf"
REPORT = json.loads((ROOT / "results" / "five-model-benchmark.json").read_text())
MODELS = {m["name"]: m for m in REPORT["models"]}
WIDTH = A4[0] - 84
INK = colors.HexColor("#18283B")
TEAL = colors.HexColor("#087F8C")
MUTED = colors.HexColor("#526378")
RULE = colors.HexColor("#D9E2EA")
PALE = colors.HexColor("#F2F6F9")
AMBER = colors.HexColor("#FFF4DE")


def register_fonts():
    bundled = Path(reportlab.__file__).parent / "fonts"
    candidates = [
        (Path("/usr/share/fonts/truetype/dejavu"), "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf"),
        (bundled, "Vera.ttf", "VeraBd.ttf", "Vera.ttf"),
    ]
    for directory, regular, bold, mono in candidates:
        paths = [directory / name for name in (regular, bold, mono)]
        if all(path.exists() for path in paths):
            for name, path in zip(("Body", "BodyBold", "Code"), paths):
                pdfmetrics.registerFont(TTFont(name, str(path)))
            pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")
            return
    raise RuntimeError("Install DejaVu Sans fonts, or use a ReportLab distribution with bundled Vera fonts")


register_fonts()
STYLES = {
    "body": ParagraphStyle("body", fontName="Body", fontSize=9.6, leading=13.15,
        textColor=INK, spaceAfter=5, allowWidows=0, allowOrphans=0),
    "small": ParagraphStyle("small", fontName="Body", fontSize=8.1, leading=11,
        textColor=MUTED, spaceAfter=5),
    "title": ParagraphStyle("title", fontName="BodyBold", fontSize=22, leading=27,
        textColor=INK, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="BodyBold", fontSize=11.1, leading=14,
        textColor=TEAL, spaceBefore=6, spaceAfter=5, keepWithNext=True),
    "cell": ParagraphStyle("cell", fontName="Body", fontSize=8.65, leading=11.5,
        textColor=INK, spaceAfter=0),
    "thead": ParagraphStyle("thead", fontName="BodyBold", fontSize=8.1, leading=10.5,
        textColor=colors.white, spaceAfter=0),
    "code": ParagraphStyle("code", fontName="Code", fontSize=8.0, leading=10.7,
        textColor=INK, spaceAfter=0, alignment=TA_LEFT),
}


def p(text, style="body"):
    return Paragraph(text, STYLES[style])


def h(text):
    return p(text, "h2")


def table(rows, widths, header=True):
    rendered = [[p(str(value), "thead" if header and i == 0 else "cell") for value in row]
                for i, row in enumerate(rows)]
    result = Table(rendered, colWidths=[WIDTH * part for part in widths], hAlign="LEFT")
    commands = [("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LINEBELOW", (0, 0), (-1, -1), .35, RULE)]
    if header:
        commands += [("BACKGROUND", (0, 0), (-1, 0), INK),
                     ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE])]
    result.setStyle(TableStyle(commands))
    result.spaceAfter = 7
    return result


def box(text, warning=False):
    result = Table([[p(text)]], colWidths=[WIDTH])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), AMBER if warning else PALE),
        ("BOX", (0, 0), (-1, -1), .5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    result.spaceAfter = 7
    return result


def code(text):
    lines = text.strip("\n").splitlines()
    for line in lines:
        if pdfmetrics.stringWidth(line, "Code", 8.0) > WIDTH - 20:
            raise ValueError(f"Code line is too wide: {line}")
    content = "<br/>".join(escape(line).replace(" ", "&#160;") or "&#160;" for line in lines)
    result = Table([[p(content, "code")]], colWidths=[WIDTH])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    result.spaceAfter = 7
    return result


class Chain(Flowable):
    """A compact vector diagram for a genuinely sequential pipeline."""
    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self.width, self.height = WIDTH, 47

    def draw(self):
        gap = 14
        width = (self.width - gap * (len(self.labels) - 1)) / len(self.labels)
        canvas = self.canv
        for i, label in enumerate(self.labels):
            x = i * (width + gap)
            canvas.setFillColor(PALE)
            canvas.setStrokeColor(RULE)
            canvas.roundRect(x, 9, width, 32, 5, fill=1, stroke=1)
            canvas.setFillColor(INK)
            canvas.setFont("BodyBold", 8)
            lines = label.split("\n")
            for j, line in enumerate(lines):
                canvas.drawCentredString(x + width / 2, 25 - j * 10 + (len(lines) == 1) * -3, line)
            if i < len(self.labels) - 1:
                canvas.setStrokeColor(TEAL)
                start = x + width + 3
                canvas.line(start, 25, start + 8, 25)
                canvas.line(start + 5, 28, start + 8, 25)
                canvas.line(start + 5, 22, start + 8, 25)


def refs(text):
    return [Spacer(1, 4), p("<b>Open in the project:</b> " + text, "small")]


def page(title, subtitle):
    return [p(title, "title"), p(subtitle, "small")]


def overview():
    out = page("What you built & reached", "01 / THE STORY — Start your oral presentation here.")
    out += [box('<b>Your 60-second opening:</b> “CardioAgent-Ops is a research platform that combines '
        'heart-sound classification with traceable guideline retrieval. I harmonized PASCAL and PhysioNet, '
        'compared five audio model families with validation-only tuning, tracked experiments in MLflow, '
        'and deployed a verified ONNX artifact. Three LangGraph agents connect that score to ESC evidence '
        'and an audited LLM report. A streaming web interface exposes the six-service Docker system. '
        'It is educational research, not a diagnosis or a treatment tool.”'),
        h("Six containers, one entry point"),
        table([
            ["Service", "Responsibility", "Host port", "RAM snapshot"],
            ["api-gateway", "FastAPI + frontend + upload/SSE proxy", "8080*", "43.36 MiB"],
            ["agent-orchestrator", "Three-node LangGraph and report audit", "8003", "314.6 MiB"],
            ["audio-engine", "Canonical audio features + ONNX Runtime", "8002", "327.2 MiB"],
            ["llm-server", "Provider facade: Ollama OR DeepSeek", "8001", "42.14 MiB"],
            ["vector-db", "Qdrant evidence vectors and payloads", "6333", "72.21 MiB"],
            ["mlflow-server", "Experiments, artifacts, model registry", "5000", "1.57 GiB"],
        ], [.22, .43, .12, .23]),
        p('*8080 is this machine’s configured gateway port; Compose defaults to 8000. Ports bind to '
          '127.0.0.1. Ollama runs on the host, not as a seventh Compose container. The frontend is served '
          'by the gateway. RAM values are historical working-set snapshots during concurrent CPU training, '
          'not isolated peaks; Ollama separately used 2,328 MiB VRAM.', "small"),
        h("What is actually completed"),
        p('<b>Five real trained/exported families</b>, a full 2,960-recording benchmark, a deployed Decision Tree, '
          '806 indexed ESC chunks, real local Mistral generation, live browser/API integration, MLflow '
          'registration, tests and six-image CI builds. The full CPU training workflow took about 33.2 minutes. '
          'Colab is prepared but that GPU reproduction is <b>not</b> a completed experiment.'),
        box('<b>Grading-sensitive deviation:</b> the assignment explicitly requests LLM QLoRA/PEFT and GGUF '
            'training/export. This implementation deliberately does <b>not</b> train LLM weights, following '
            'your clarified scope. Prompting/RAG are not QLoRA. Its LLM container is a facade to host Ollama '
            'rather than the requested self-contained fine-tuned Ollama runtime. Disclose these differences.', True),
        p('<b>Reading map:</b> 2 data • 3 signal processing • 4 models/optimization • 5 results/metrics • '
          '6 notebooks/MLOps • 7 RAG/PDFs • 8 agents/safety • 9 demonstration • 10 oral questions.', "small"),
        p('<b>French terms you may hear:</b> rappel = recall; spécificité = specificity; précision = '
          'positive predictive value; rééchantillonnage = resampling; fenêtre = window; trame = '
          'analysis frame; lot = batch; surapprentissage = overfitting; jeu de validation ≠ jeu de test.', "small")]
    return out + refs("README.md → Specification decisions; compose.yaml; results/README.md; updated-documents.pdf, pp. 2–7 (local reference).")


def data_page():
    out = page("Data, splitting & leakage", "02 / REPRODUCIBLE EXPERIMENT — Explain the unit of evaluation before showing a score.")
    out += [h("Two sources, one binary research task"),
        p('<b>PhysioNet:</b> read REFERENCE.csv; -1 → normal (0), +1 → abnormal (1). '
          'When supplied, REFERENCE-SQI.csv excludes poor-quality recordings. <b>PASCAL:</b> normal → 0; '
          'murmur, extrahls and extrastole → 1. Unlabeled, artifact, missing or ineligible rows are excluded. '
          'These are dataset acoustic labels, not a model that diagnoses particular valve diseases.'),
        p('Compute a SHA-256 of each WAV file before splitting, remove exact duplicates, and reject conflicting '
          'labels on identical files. After filtering: <b>2,960 eligible recordings; 1,112 excluded source rows</b>. '
          'Store the source, label, acquisition/recording group, hash and partition in an immutable manifest.'),
        table([
            ["Partition", "Normal (0)", "Abnormal (1)", "Total / purpose"],
            ["Training · 60%", "1,401", "375", "1,776 · fit parameters"],
            ["Validation · 20%", "467", "125", "592 · choose configuration/champion"],
            ["Test · 20%", "467", "125", "592 · final held-out comparison"],
        ], [.23, .17, .18, .42]),
        h("Why these splits are safer than a random file split"),
        p('Use <b>StratifiedGroupKFold(5, shuffle=True, random_state=42)</b>, stratifying by source AND '
          'class. One fold becomes test, a second validation and the remaining three training. This is '
          '<b>one fixed 60/20/20 experiment, not reported five-fold cross-validation</b>. Acquisition groups '
          'stay in one partition: PASCAL uses timestamps/recording IDs; PhysioNet uses recording IDs.'),
        box('<b>Important limit:</b> patient identities are not verified. Therefore say '
            '“acquisition/recording-group split,” not “patient-independent evaluation.” Different WAV '
            'headers can hide identical audio from byte hashing; related recordings without shared IDs '
            'can still leak. No external patient cohort was evaluated.', True),
        h("Imbalance: weighting, not synthetic oversampling"),
        p('Only training labels determine class weights: <b>w(c) = Ntrain / (2 × Ntrain,c)</b>. '
          'Here normal ≈ 0.634 and abnormal ≈ 2.368. Tree/SVM use class_weight="balanced"; deep models '
          'receive the same weighting formula in fit(). No SMOTE, augmentation pipeline or test balancing '
          'is implemented. Validation and test retain their observed prevalence.'),
        h("Where leakage would happen"),
        p('The SVM StandardScaler is fitted <b>inside its training-only pipeline</b>; never scale the full '
          'dataset before splitting. Hyperparameters and champion use validation only; the fixed 0.5 '
          'decision threshold is not tuned on test. Test is evaluated after all selections are fixed. '
          'Reusing these visible test results for new model choices would require a new reserved cohort.'),
        p('<b>Notebook role:</b> 01_eda_data_wrangling.ipynb exposes label distributions, split checks and '
          'waveform/Mel inspection. The reusable package functions, not copied notebook logic, define '
          'the experiment contract.')]
    return out + refs("training/data.py → manifest(), split(), write_manifest(); vault/manifests/audio.json; notebooks/01_eda_data_wrangling.ipynb.")


def audio_page():
    out = page("Samples, windows & audio features", "03 / SIGNAL PROCESSING — “Chunk,” “sample,” “frame” and “batch” are not interchangeable.")
    out += [Chain(["WAV bytes", "Mono + 2 kHz", "DC / peak\n5 s fixed length", "84 features OR\n96 × 96 Mel"]),
        table([
            ["Term", "Exact meaning in this implementation"],
            ["Audio sample", "One waveform amplitude. At 2,000 Hz: 2,000 samples/second; spacing = 0.5 ms."],
            ["Recording window", "Only the first 5 seconds are read: 10,000 samples after resampling. Short clips are zero-padded; longer clips are truncated. No all-recording sliding-window aggregation."],
            ["STFT frame / hop", "Mel analysis uses n_fft=256 (128 ms), hop_length=105 (52.5 ms). Frames overlap; the output is fixed to 96 time frames."],
            ["Training batch", "16 recordings/features per optimizer update, not 16 raw amplitude samples."],
            ["PDF chunk", "A text passage of up to 1,200 characters with 180-character overlap; unrelated to audio sampling."],
        ], [.23, .77]),
        h("Canonical preprocessing: identical in training and serving"),
        p('Validate RIFF/WAVE; accept bounded sample rates (1–192 kHz) and at most eight channels. Read at '
          'most the first five seconds, average channels to mono, resample to <b>2 kHz</b>, remove the DC '
          'offset (subtract the mean), divide by peak amplitude and pad to 10,000 values. Reject empty, '
          'silent or non-finite audio. Peak normalization reduces gain differences, not all recording noise.'),
        p('<b>Why 2 kHz?</b> It follows the project specification and limits computation. Nyquist is 1 kHz; '
          'the Mel filterbank covers 20–1,000 Hz. This discards higher-frequency information and is a '
          'design tradeoff, not proof every relevant acoustic feature is preserved. There is no separate '
          'explicit denoising/band-pass stage or heartbeat segmentation.'),
        h("Two representations, chosen for the algorithm"),
        p('<b>Tree/SVM: 84 values.</b> Forty MFCC coefficients summarize the log-Mel spectral envelope. '
          'Their 40 means + 40 standard deviations, plus mean/std of zero-crossing rate and RMS energy, '
          'give 84. ZCR describes sign changes; RMS describes energy. Aggregation is compact but loses '
          'fine temporal order. ZCR/RMS use Librosa’s default analysis windows, not the Mel frame settings.'),
        code('mel = librosa.feature.melspectrogram(y=signal, sr=2000,\n'
             '    n_fft=256, hop_length=105, n_mels=96, fmin=20, fmax=1000)\n'
             'db = librosa.power_to_db(mel, ref=np.max)\n'
             '# Deep input: fix time axis to 96, add channel axis, divide by 80.'),
        p('<b>Deep models: 96 × 96 × 1.</b> A log-Mel image retains frequency versus time. Power-to-dB '
          'uses the clip maximum and the default 80 dB floor; division by 80 places values roughly in '
          '[-1, 0]. Training batches are (B, 96, 96, 1), versus (B, 84) for classical models.'),
        box('<b>Versioned contract:</b> pcg-2000hz-5s-mfcc40-mel96-v1. Changing extraction requires '
            'retraining and new metadata. Legacy 80-value/flattened-1,200-value coursework features are '
            '<b>not</b> compatible with the deployed model.')]
    return out + refs("audio_engine/contract.py → decode(), features(); audio_engine/deployment.py; do not confuse legacy /preprocess with canonical /predict.")


def models_page():
    out = page("Five models & how they learn", "04 / TRAINING — Hyperparameter tuning is not the same as learning weights.")
    tree = MODELS['decision_tree']['parameters']
    svm = MODELS['svm']['parameters']
    out += [table([
        ["Family", "Representation and architecture", "Selected configuration"],
        ["Decision Tree", "84 features; greedy, weighted Gini splits. Small depth limits complexity; minimum leaf size avoids tiny leaves.",
         f"max_depth={tree['max_depth']}; min_samples_leaf={tree['min_samples_leaf']}"],
        ["RBF SVM", "84 features → training-fitted StandardScaler → SVC. Nonlinear distance-based kernel; no neural optimizer.",
         f"C={svm['C']}; gamma={svm['gamma']}; probability=True"],
        ["Scratch CNN", "Mel image; Conv(16,32,64), 3×3 + ReLU, each followed by 2× pooling; global average pooling.",
         f"Adam LR={MODELS['cnn']['parameters']['learning_rate']}; dropout=0.25"],
        ["ResNet50", "Keras Applications, ImageNet weights, include_top=False; grayscale copied to 3 channels; ImageNet affine preprocessing.",
         f"Warmup LR={MODELS['resnet50']['parameters']['learning_rate']}; top-stage LR=0.000001"],
        ["Transformer–CNN", "Same CNN; reshape to 12 temporal tokens, Dense(64), 2-head self-attention, residual/norm + 128→64 feedforward.",
         f"Adam LR={MODELS['transformer_cnn']['parameters']['learning_rate']}; dropout=0.25"],
    ], [.19, .51, .30]),
        h("What was tuned, and why"),
        p('Small validation search: tree depth {3, 6, 12, unlimited} with leaf minimum 3; SVM '
          '<b>C ∈ {0.1, 1, 10}</b> and <b>gamma ∈ {scale, 0.01}</b>; deep learning rates {0.001, 0.0001}. '
          'C controls SVM penalty strength; gamma controls how local each RBF influence is. Search was '
          'bounded for CPU efficiency: four tree trials, six SVM trials, two per deep family—not an exhaustive optimum.'),
        h("Optimization and regularization"),
        p('The tree learns splits, not gradients. SVM uses libsvm’s constrained solver, not Adam. '
          '<b>All deep models use Adam + binary cross-entropy</b>, a final one-unit sigmoid, training-only '
          'class weights, dropout 0.25 and validation-loss early stopping. Adam adapts updates using first '
          'and second gradient moments; learning rate controls step size. Batch size 16 limits memory; '
          '1,776 training examples imply 111 updates per complete epoch. Maximum: 20 epochs per fit phase.'),
        code('# Reduced training excerpt; X_train/X_val denote fixed partitions.\n'
             'stop = EarlyStopping(monitor="val_loss", patience=3,\n'
             '                     restore_best_weights=True)\n'
             'model.fit(X_train, y_train, validation_data=(X_val, y_val),\n'
             '          epochs=20, batch_size=16, class_weight=weights, callbacks=[stop])'),
        code('# Full training entry point; do not rerun during the demo:\n'
             'uv run --locked --extra deep-learning python -m training.benchmark \\\n'
             '  --epochs 20 --batch-size 16 --trials 6'),
        p('<b>ResNet fine-tuning:</b> first freeze the ImageNet backbone and train the new head; then '
          'unfreeze only conv5_ layers, keep batch-normalization frozen and use LR/100. This adapts '
          'generic visual filters to Mel images; ImageNet is <b>not</b> a heart-sound pretrained dataset. '
          'The hybrid adds temporal attention after convolution; it is not a full pretrained AST/Conformer.'),
        box('<b>“Are best weights saved?”</b> EarlyStopping restores the lowest-validation-loss weights '
            '<b>in RAM within each fit phase</b>. The selected configuration is persisted as ONNX, containing '
            'its learned parameters. There is no per-epoch disk checkpoint or resumable optimizer state; '
            'ResNet does not compare a global best across warmup and fine-tuning. No LLM weights are trained.', True)]
    return out + refs("training/benchmark.py → deep_model(), run(); keras.io/api/applications/resnet/; source code is authoritative, not legacy train_svm.py.")


def metrics_page():
    out = page("Actual results & how to defend them", "05 / HELD-OUT BENCHMARK — Positive means the dataset’s abnormal acoustic class.")
    names = {'decision_tree':'Decision Tree', 'svm':'RBF SVM', 'cnn':'Scratch CNN',
             'resnet50':'ResNet50', 'transformer_cnn':'Transformer–CNN'}
    rows = [["Model", "Recall", "Specificity", "Precision", "F1", "AUROC", "AUPRC"]]
    for name, model in MODELS.items():
        m = model['test']
        rows.append([names[name], *[f"{m[key]*100:.2f}%" for key in ('recall_abnormal','specificity','precision_abnormal')],
                     *[f"{m[key]:.4f}" for key in ('f1_abnormal','auroc','auprc')]])
    out += [p('All five families used the same fixed split. <b>Test: 592 recordings, 125 abnormal '
              '(21.1% prevalence).</b> These are completed CPU experiments, not patient-independent or clinical validation.'),
        table(rows, [.20, .13, .14, .14, .11, .14, .14]),
        h("Why the tree is champion—not “best on every metric”"),
        p('Before reading test scores: keep models with <b>validation specificity ≥ 0.50</b>, then maximize '
          'validation abnormal recall; break ties by specificity and AUPRC. The tree’s validation recall '
          'was <b>95.2%</b> with specificity <b>68.95%</b>. Predictions use a fixed 0.5 score threshold. '
          'The gate prevents a trivial all-abnormal classifier (100% recall, 0% specificity) from winning; '
          '0.50 is a research acceptance rule, not a clinical standard.'),
        h("Tree confusion matrix: understand each error"),
        table([
            ["Dataset truth ↓ / prediction →", "Predicted normal", "Predicted abnormal"],
            ["Actual normal", "TN = 332", "FP = 135 · false alarms"],
            ["Actual abnormal", "FN = 2 · missed positives", "TP = 123"],
        ], [.42, .29, .29]),
        table([
            ["Metric / formula", "What the professor expects you to explain"],
            ["Recall = TP / (TP + FN)", "Among actual abnormal labels, how many were caught? Tree: 123/125 = 98.4%."],
            ["Specificity = TN / (TN + FP)", "Among actual normal labels, how many stayed negative? Tree: 332/467 = 71.09%."],
            ["Precision = TP / (TP + FP)", "Among positive predictions, how many match abnormal labels? Tree: 123/258 = 47.67%."],
            ["F1 = 2PR / (P + R)", "Balance of positive precision and recall; ignores true negatives."],
        ], [.41, .59]),
        p('<b>AUROC</b> summarizes positive/negative ranking across thresholds. <b>AUPRC</b> here is '
          'average precision and emphasizes the positive class; interpret it with prevalence (random-ranking '
          'baseline ≈ 0.211 here). The SVM has much better specificity/precision; ResNet has the highest '
          'test AUPRC. Do not retrospectively choose a different champion using these test results.'),
        box('<b>Honest conclusion:</b> high tree recall comes with 135 false positives. Its accuracy is '
            '76.86%, below an all-normal baseline’s 78.89%—but that baseline has zero abnormal recall. '
            'Accuracy alone is misleading. Scores are not verified calibrated disease probabilities; '
            'no confidence intervals or external-patient validation were established.', True)]
    return out + refs("results/five-model-benchmark.json (generated metric table); results/README.md; training/benchmark.py → metrics() and champion selection.")


def mlops_page():
    out = page("Notebooks → MLflow → deployed artifact", "06 / MLOPS — Prove where the live model came from.")
    out += [Chain(["Manifest +\nversioned features", "Train / validate\nnested MLflow", "ONNX + JSON\nhash / parity", "Promote +\nrestart inference"]),
        h("Notebooks explore; package code executes the contract"),
        p('<b>01_eda_data_wrangling:</b> inspect labels, groups and audio. <b>02_audio_benchmark:</b> '
          'launch the package benchmark and inspect saved comparisons/confusion counts. '
          '<b>03_colab_gpu:</b> check GPU, mount Drive, preserve datasets/MLflow/vault, export actual artifacts. '
          'Notebook files contain no fabricated executed outputs. Completed results came from the real '
          'CPU run; GPU/Colab reproduction is prepared, not claimed as completed.'),
        h("MLflow hierarchy and registry"),
        p('<b>Parent benchmark run → family child run → tuning trial run.</b> Log configurations, validation '
          'metrics, per-epoch deep losses, fit timings, manifest, feature contract and ONNX/metadata. '
          'Compute all test results only after configuration/champion selection. The registered model '
          'is <b>CardioAgentAudio</b>; <b>champion</b> is an explicit alias, not automatically replaced by '
          'every experiment. Inference telemetry contains timings/model identifiers, not audio or prompts.'),
        table([
            ["Private vault path", "What it centralizes"],
            ["manifests/ ; runs/", "Immutable partitions; per-run models, metadata and benchmark.json"],
            ["models/", "Active cardio_model.onnx + cardio_model.json; live registry.json receipt"],
            ["documents/ ; rag/ ; cache/", "Source ESC PDFs, provenance and cached embedding weights"],
            ["evaluations/ ; reports/ ; exports/", "Local evaluation outputs, reports and transferable artifacts"],
        ], [.40, .60]),
        p('This vault is a <b>private filesystem workspace, not HashiCorp Vault</b> and not encrypted secret '
          'management. Keys stay in ignored .env. Qdrant and the live MLflow server additionally use Docker '
          'named volumes: back up those plus the vault consistently. They are not published to Git.'),
        h("Export/activation gates—not a fallback classifier"),
        p('Classical models export with skl2onnx (including the SVM scaler); deep models use tf2onnx, '
          'opset 17. Compare ONNX and native Python probabilities on up to 16 validation examples: '
          '<b>max absolute difference ≤ 0.0001</b>. All five passed; the largest observed difference '
          'was 2.44×10⁻⁶. Promotion verifies the recorded parity result, SHA-256, feature version, validation '
          'specificity and that ONNX Runtime can load the artifact; it does <b>not</b> rerun native-model parity.'),
        code('# Optional reproduction, NOT needed during the demonstration:\n'
             'RUN=73e89075b35d4eefb46a25d55b183cb4\n'
             'uv run python -m training.promote \\\n'
             '  "vault/runs/$RUN/decision_tree.onnx"\n'
             'docker compose restart audio-engine'),
        p('<b>Active lineage:</b> family run 3ad74f18b00f4fae92223f43869a235f; live registry version 2 '
          '(offline store: version 4). Counters are store-specific; registry.json links version, hash '
          'and training run. Never merge Colab SQLite into running MLflow. Missing/mismatched artifacts '
          'fail readiness; no fallback classifier.')]
    return out + refs("training/benchmark.py → export_model(), verify_artifact(), activate(); training/promote.py; vault/models/registry.json; notebooks/.")


def rag_page():
    out = page("RAG: PDFs become cited evidence", "07 / CHUNKING + RETRIEVAL — Qdrant stores vectors; our ingestion code performs chunking.")
    out += [Chain(["ESC PDFs\npage text", "1,200 chars\n180 overlap", "384-D vectors\nQdrant cosine", "Query → top 4\nscore ≥ 0.55"]),
        h("What PDFs, and who uploads them?"),
        p('The indexed corpus contains the <b>102-page 2025 ESC/EACTS professional valvular-disease '
          'guideline</b> and a <b>16-page patient companion</b>: <b>806 chunks</b>. The canonical journal '
          'download returned HTTP 403, so the source manifest explicitly records a public university '
          'mirror as download_url while retaining the original source URL. PDFs stay in the private vault.'),
        box('<b>There is no user-facing PDF upload.</b> The application accepts WAV plus an optional '
            'question. Administrators add licensed source PDFs through the ingestion CLI and HTTPS source '
            'manifest. The assignment asks for ESC document indexing, not explicitly a user PDF-upload '
            'screen. Do not claim that the UI can ingest a visitor’s PDF.'),
        code('uv run python -m rag_engine.ingest --fetch\n'
             '# Licensed custom sources: --sources path/to/source-manifest.json\n'
             '# PDFs are read from vault/documents/esc/.'),
        h("Chunking: why overlap, and what the numbers mean"),
        p('pypdf extracts each page; collapse whitespace, take passages of <b>up to 1,200 characters</b>, '
          'prefer a word boundary, and move forward with <b>180 characters of overlap</b>. These are '
          '<b>characters, not tokens or audio samples</b>. Discard passages shorter than 80 characters. '
          'Overlap reduces the chance that a useful statement is split away from its context, at the cost '
          'of extra storage and near-duplicate retrieval. Page-local chunks preserve page provenance but '
          'cannot bridge a statement spread across two pages.'),
        h("Embedding and search"),
        p('Lazy-load <b>BAAI/bge-small-en-v1.5</b> using FastEmbed on CPU. Passage embeddings have '
          '<b>384 dimensions</b>; query_embed encodes the question with the appropriate query path. '
          'Embed in batches of 16 and upsert in batches of 64. Qdrant searches by cosine similarity and '
          'returns at most four passages above <b>0.55</b>. This threshold is an initial setting, not an '
          'optimized medical relevance boundary. No sufficiently relevant hits → abstain without an LLM report.'),
        p('Each payload retains text, filename, page, original/download URLs, PDF SHA-256 and embedding '
          'model. Stable IDs derive from PDF hash + page + chunk index. Repeating the same ingestion '
          'upserts the same IDs; changed PDFs should use a new collection so obsolete chunks do not remain.'),
        h("What retrieval evaluation actually establishes"),
        p('On <b>four assistant-curated literal lookup queries</b>: Recall@4 = 1.0 and MRR = 1.0; '
          'known-target Precision@4 = 0.25. Recall counts retrieved annotated targets; MRR averages '
          '1/(rank of first target). Each query has only one annotated target, not a complete relevance '
          'set, so 0.25 is not general evidence-relevance precision. This is a small smoke test, '
          '<b>not</b> clinician-adjudicated medical QA, semantic entailment or independent RAG quality.')]
    return out + refs("rag_engine/core.py → chunks(), Retriever; rag_engine/ingest.py; rag_engine/sources.json; evaluation/esc_smoke.jsonl; evaluation/run.py.")


def agents_page():
    out = page("Exactly three agents, bounded reasoning", "08 / ORCHESTRATION + SAFETY — An agent is a specialized graph node, not necessarily another LLM.")
    out += [Chain(["AudioAgent\nONNX result", "KnowledgeRAG-Agent\nESC evidence", "ReasoningLLM-Agent\ngenerate + audit"]),
        p('<b>AudioAgent</b> sends the WAV to the audio service and validates its acoustic label/score. '
          '<b>KnowledgeRAG-Agent</b> retrieves passages using a fixed auscultation query plus the optional '
          'question. <b>ReasoningLLM-Agent</b> combines that derived result with retrieved evidence, '
          'generates structured JSON, and runs deterministic validation. The auditor is inside the last '
          'node—not a fourth autonomous agent. Only the last node makes an LLM generation call.'),
        h("Why LangGraph and streaming?"),
        p('StateGraph makes the sequence and shared state explicit: content, question, consent, audio, '
          'sources, report, audit and provenance. The graph is a finite START → three nodes → END chain, '
          'not an open-ended planning loop. SSE carries stage updates, draft tokens and the final event '
          'over HTTP. Drafts are labeled <b>unaudited</b>; streaming improves perceived responsiveness, '
          'not classification quality or total model compute.'),
        h("The LLM has a specific, limited role"),
        p('Default: local <b>Ollama / mistral:7b-instruct</b>. It writes an educational synthesis; it '
          'neither analyzes the waveform directly nor replaces the ONNX classifier. Temperature 0, '
          'max_tokens 1,400 and a report JSON schema constrain generation. No QLoRA, LLM weight update '
          'or secret automatic provider switch is implemented.'),
        p('<b>DeepSeek:</b> explicitly configure the provider/key locally, recreate llm-server, and require '
          'external_consent=true <b>per request</b>. Send only derived classifier output and public '
          'retrieved passages. Never send raw audio or identifying information; the free-text question '
          'stays local for retrieval. Reject cloud-tagged/remote aliases masquerading as local Ollama. '
          'The DeepSeek live path has not been verified with a real key.'),
        h("Validation: fail closed, but do not claim semantic proof"),
        code('# Reduced from agents/schemas.py; whitespace is normalized first.\n'
             'if citation.source_id not in allowed:\n'
             '    errors.append("Unknown citation ID")\n'
             'elif normalized_quote not in allowed[citation.source_id]:\n'
             '    errors.append("Citation quote does not match retrieved evidence")'),
        p('<b>Pydantic:</b> forbid extra fields; require summary, limitations, 1–4 citations, source IDs, '
          '25–700-character quotes and a fixed clinician-review next_step. Restrict citation IDs to '
          'retrieved sources; audit verbatim quotes and conservative unsafe/instruction-like phrases. '
          'Malformed/unsafe output → <b>rejected, report=null</b>; no evidence → <b>abstained</b>. '
          'Documents/questions are untrusted data, not instructions. These checks do not establish '
          'clinical correctness, semantic support or comprehensive prompt-injection resistance.'),
        box('<b>Resource/security controls:</b> upload ≤20 MiB; prompt ≤2,000 characters; UI duration '
            '≤120 s. Gateway admission four before upload parsing (overflow → 429); graph concurrency '
            'two; SSE queue 32; disconnect cancellation and provider timeouts. Pooled clients/sessions; '
            'no raw clinical content in telemetry. Optional gateway API key. This is not hospital-grade TLS infrastructure.')]
    return out + refs("agents/graph.py; agents/schemas.py; agents/orchestrator.py; llm_server/main.py; api/main.py; api/web/.")


def demo_page():
    out = page("A defensible live demonstration", "09 / RUNBOOK — Prepare today; do not retrain everything in front of the professor.")
    out += [h("Before the evaluation: preserve the working state"),
        p('Keep datasets, vault, Docker named volumes and local Ollama weights. Build/start beforehand, '
          'warm the model once, and test one complete request. Use Docker Compose V2; if the CLI cannot '
          'find compose, check/install its plugin first. The current gateway is <b>localhost:8080</b> '
          '(default fresh-install port is 8000). Readiness—not only /health—must pass. Never use down -v.'),
        code('docker compose up -d --wait\n'
             'curl -fsS http://localhost:8080/status\n'
             'ollama list\n'
             '# Browser: http://localhost:8080\n'
             '# MLflow:  http://localhost:5000'),
        h("Five-minute presentation order"),
        p('<b>1.</b> Explain the six services and three agents. <b>2.</b> Check services in the workbench, '
          'upload a public WAV, show waveform/playback and analyze. Playback is resampled; the original WAV '
          'is uploaded. <b>3.</b> Explain stage events, unaudited drafts, citations and the final audit. '
          'Respect rejections—never disable checks. <b>4.</b> Show the five-model table/confusion counts, '
          'then MLflow nested runs, registry and active artifact provenance.'),
        code('# Standalone classifier proof, without waiting for LLM generation:\n'
             'curl -fsS http://localhost:8002/predict \\\n'
             '  -F file=@datasets/PhysioNet/training-a/a0004.wav'),
        p('<b>Real held-out examples checked for this guide:</b> a0004.wav has dataset label abnormal '
          'and the deployed tree returned abnormal, score ≈0.666 (true positive). a0029.wav has dataset '
          'label normal but returned abnormal, also ≈0.666 (<b>false positive</b>). Same score is '
          'possible because different feature vectors reach the same tree leaf. These are illustrative '
          'cases, not a new accuracy estimate; use all 592 test recordings for metrics.'),
        h("Performance: label the measurement conditions"),
        table([
            ["Recorded experiment", "Observed", "What it does NOT establish"],
            ["Audio-only HTTP; 1 warmup + 20 serial requests", "p50 5.33 ms; p95 6.83 ms", "Host measurement, telemetry off; excludes cold start, Docker networking, LLM and concurrent load."],
            ["One full request during CPU training", "First token 8.96 s; total 151.61 s", "Final audit REJECTED. Not a success/reliability score or representative latency benchmark."],
            ["ONNX artifact size", "Tree 1,160 B; ResNet50 94,070,259 B", "Size and numerical parity do not prove medical accuracy. Tree is efficient but has many false positives."],
        ], [.33, .25, .42]),
        p('<b>Completed verification:</b> 28 core tests, 59% measured core coverage; three optional '
          'deep-export cases separately passed. CI ran tests, coverage, Compose validation and six image '
          'builds (not automatic deployment/retraining). Browser proof covered actual 2 kHz audio, playback, '
          '390-pixel layout and zero reported axe A/AA violations; it is not exhaustive accessibility review.'),
        p('<b>If something fails:</b> inspect /status → failing dependency; missing artifact → restore '
          'verified ONNX+JSON and restart audio-engine; empty index → licensed PDF ingestion; unreachable '
          'Ollama → model installed and container-to-host access/firewall; 429 → wait; rejected report → '
          'show the audit error and limits. Keep a genuine previous report as clearly labeled backup, '
          'never present it as a live run.', "small")]
    return out + refs("results/README.md; results/runtime-profile.json; evaluation/latency.py; README.md quick start; docker compose logs --tail 100.")


def viva_page():
    out = page("Viva answers & your two-day plan", "10 / REHEARSE OUT LOUD — Explain a decision, its evidence and its limitation.")
    questions = [
        ("What did you build?", "A shared data/feature contract, five-model benchmark, gated artifact lifecycle, evidence index, three-agent graph, audited streaming workbench and six-service deployment. Trace each to its module."),
        ("Why not send audio to Mistral?", "ONNX handles the acoustic task. Mistral receives a derived score and text evidence, not the waveform. This separates responsibilities, limits payloads and protects raw recordings."),
        ("Why classical AND deep models?", "Same dataset/split, different tradeoffs. The SVM is more specific, ResNet leads test AUPRC, and the tree wins the preset validation-recall rule. Complexity alone is not quality."),
        ("Parameters versus hyperparameters?", "Weights/splits/support vectors are learned from training. Depth, C, gamma and learning rate are chosen with validation. Test estimates performance; it never tunes the model."),
        ("Did you fine-tune anything?", "Yes: audio hyperparameters and ResNet’s top stage. No: LLM weights. The assignment’s QLoRA/GGUF deliverable is a disclosed deviation from your clarified scope."),
        ("Best weights saved every epoch?", "No. EarlyStopping restores the best weights in memory per phase; selected models persist as ONNX. No per-epoch disk checkpoint, optimizer resume or global cross-phase ResNet best exists."),
        ("What does 98.4% mean?", "Abnormal recall: 123 of 125 held-out positive labels caught, two missed. Not overall accuracy or diagnostic accuracy; tree precision is only 47.67%."),
        ("Why not always predict abnormal?", "Recall would be 100%, specificity 0%. The validation specificity gate rejects that degenerate solution. Our gate is a research rule, not clinical approval."),
        ("Are scores disease probabilities?", "No: uncalibrated acoustic scores at threshold 0.5. Normal does not exclude disease; abnormal does not establish it. Training class weights also affect score interpretation."),
        ("Are you patient-independent?", "No verified patient IDs. Known acquisition/recording groups and exact-file duplicates are isolated; unknown relationships and header-varied duplicate audio remain risks."),
        ("Why chunks and overlap?", "Retrieve compact passages: up to 1,200 characters, 180 overlap. Audio instead uses five-second windows of 10,000 samples; a training batch contains 16 recordings."),
        ("Does Qdrant prove correctness?", "Similarity search finds passages, not truth. Source/quote checks verify provenance, not entailment. Four literal lookup smoke cases do not establish clinical RAG quality."),
        ("Can users upload PDFs?", "Not in this UI. Administrators index PDFs using the CLI/source manifest. End users submit WAV plus an optional question; do not claim a nonexistent feature."),
        ("Why ONNX, Docker and uv?", "ONNX enables lean serving without training frameworks; Docker isolates six roles; uv.lock pins dependencies. Hash/version/parity gates prevent accidental mismatches, not medical errors."),
        ("What remains before real-world use?", "External patient validation, calibration/intervals, clinician-adjudicated RAG/report checks, controlled load/peak-memory profiling and live DeepSeek consent tests. This is research, not a clinical-ready product."),
    ]
    for question, answer in questions:
        out.append(p(f"<b>{question}</b> {answer}"))
    out += [box('<b>Day 1:</b> read pp. 1–5 beside the code; derive the four confusion counts and rehearse '
                'the five model differences. <b>Day 2:</b> trace pp. 6–9 in MLflow/UI, run the actual WAV '
                'demo, check dependencies and rehearse these answers without reading. Finish with a '
                'five-minute end-to-end presentation, not a list of tools.')]
    return out


def launch_page():
    out = page("Quick launch & how it works", "11 / TERMINAL CHEAT SHEET — Run commands from the cardioagent-ops project directory.")
    out += [
        h("1 · Start the local LLM (terminal 1)"),
        code("OLLAMA_HOST=0.0.0.0:11434 ollama serve"),
        p('Leave it running. If Ollama already runs as a service, do not start another instance; '
          'configure that service to accept Docker-to-host connections. Binding to 0.0.0.0 also '
          'exposes Ollama to your network: use a trusted network/firewall.', "small"),
        h("2 · Prepare and launch (terminal 2)"),
        code("uv sync --locked --dev\n"
             "uv run python scripts/init_vault.py\n"
             "ollama pull mistral:7b-instruct\n"
             "docker compose up --build -d\n"
             "docker compose ps\n"
             "docker compose port api-gateway 8000"),
        p('Initialization preserves the existing .env. Pull downloads the model only if needed. '
          'The last command prints the actual app port: default <b>http://localhost:8000</b>; '
          'this project’s configured demo uses <b>http://localhost:8080</b>. '
          'MLflow: <b>http://localhost:5000</b>. Open the app and click <b>Check services</b>.', "small"),
        h("3 · First launch only: populate the PDF index"),
        code("docker compose exec agent-orchestrator \\\n"
             "  python -m rag_engine.ingest --vault /vault --fetch"),
        p('Wait for the orchestrator container to be running. This indexes the professional ESC/EACTS '
          '2025 valvular guideline and its patient companion. Existing model/PDF files live in the vault; '
          'no retraining is needed to launch with a verified artifact. An empty Qdrant volume still '
          'needs ingestion. Wait for completion before analysis.', "small"),
        h("4 · What happens when you upload a WAV"),
        p('<b>Gateway (8000 inside Docker)</b> serves the UI and forwards requests. '
          '<b>Orchestrator (8003)</b> runs exactly three specialized graph agents. '
          '<b>Audio engine (8002)</b> preprocesses the WAV and runs the verified ONNX classifier. '
          '<b>Qdrant (6333)</b> returns embedded PDF passages. <b>LLM facade (8001)</b> calls host '
          'Ollama/Mistral to generate text from derived results and retrieved evidence—not raw audio. '
          'The graph audits the report before acceptance. <b>MLflow (5000)</b> tracks experiments '
          'and model provenance. No LLM fine-tuning is performed.'),
        h("5 · Diagnose, stop and restart"),
        code("curl --fail http://localhost:8001/ready\n"
             "docker compose logs -f --tail=50\n"
             "# Ctrl+C exits log following; containers keep running\n"
             "docker compose down\n"
             "# Next launch: start Ollama, then:\n"
             "docker compose up -d"),
        p('Do not use <b>down -v</b>: it deletes named volumes, including the index and MLflow data. '
          'No reinstall, model download or reindex is needed on normal restarts. '
          'Missing dependencies/artifacts mean unavailable—not a fallback prediction. '
          '<b>Research only; acoustic labels are not clinical diagnoses.</b>', "small"),
    ]
    return out


PAGES = [overview, data_page, audio_page, models_page, metrics_page,
         mlops_page, rag_page, agents_page, demo_page, viva_page, launch_page]


def decorate(canvas, document):
    canvas.saveState()
    canvas.setFillColor(TEAL)
    canvas.setFont("BodyBold", 8)
    canvas.drawString(42, A4[1] - 29, "CARDIOAGENT-OPS  /  ORAL DEFENSE PLAYBOOK")
    canvas.setStrokeColor(RULE)
    canvas.line(42, A4[1] - 39, A4[0] - 42, A4[1] - 39)
    canvas.line(42, 34, A4[0] - 42, 34)
    canvas.setFillColor(MUTED)
    canvas.setFont("Body", 7.3)
    canvas.drawString(42, 22, "Research only • Source-backed results • Never a medical diagnosis")
    canvas.drawRightString(A4[0] - 42, 22, f"{document.page} / {len(PAGES)}")
    canvas.restoreState()


def main():
    if not REPORT['complete_five_model_benchmark'] or REPORT['subset']:
        raise ValueError("The guide requires the completed full five-model benchmark")
    story = []
    titles = []
    for index, make in enumerate(PAGES):
        if index:
            story.append(PageBreak())
        elements = make()
        titles.append(elements[0].getPlainText())
        story.extend(elements)
    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4,
        rightMargin=42, leftMargin=42, topMargin=56, bottomMargin=44,
        title="CardioAgent-Ops — Oral Defense Guide with Launch Appendix",
        author="CardioAgent-Ops project", subject="Architecture, data, models, evaluation and demonstration",
        pageCompression=1)
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    pages = PdfReader(OUTPUT).pages
    if len(pages) != len(PAGES):
        raise ValueError(f"Expected exactly {len(PAGES)} pages, generated {len(pages)}; fix content/layout")
    for number, (pdf_page, title) in enumerate(zip(pages, titles), 1):
        text = " ".join(pdf_page.extract_text().split())
        if title not in text or f"{number} / {len(PAGES)}" not in text:
            raise ValueError(f"Content moved away from its intended page: {title}")
    print(f"Built {OUTPUT.relative_to(ROOT)}: {len(pages)} pages, {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
