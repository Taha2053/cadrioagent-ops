"""Source labels, duplicate-aware recording groups, and immutable splits."""
from __future__ import annotations
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def manifest(root: Path) -> tuple[list[dict], list[dict]]:
    records, exclusions = [], []
    for reference in sorted((root / "PhysioNet").glob("**/REFERENCE.csv")):
        sqi_path = reference.with_name("REFERENCE-SQI.csv")
        sqi = {r[0]: int(r[2]) for r in csv.reader(sqi_path.open())} if sqi_path.exists() else {}
        for name, label, *_ in csv.reader(reference.open()):
            path = reference.parent / f"{name}.wav"
            if int(label) not in {-1, 1} or sqi.get(name, 1) != 1 or not path.exists():
                exclusions.append({"path": str(path), "reason": "missing, unlabeled, or poor SQI"})
                continue
            records.append({"path": str(path.resolve()), "label": int(int(label) == 1),
                "source": "physionet", "group": f"physionet:{name}"})
    for source in (root / "pascal").glob("set_[ab].csv"):
        for row in csv.DictReader(source.open()):
            path = root / "pascal" / row["fname"]
            label = row["label"].strip().lower()
            if label not in {"normal", "murmur", "extrahls", "extrastole"} or not path.exists():
                exclusions.append({"path": str(path), "reason": f"excluded label {label!r} or missing"})
                continue
            # B recordings share an acquisition timestamp across locations/variants.
            match = re.search(r"_(\d{12,})", path.stem)
            group = match.group(1) if match else path.stem.split("__")[-1]
            records.append({"path": str(path.resolve()), "label": int(label != "normal"),
                "source": "pascal", "group": f"pascal:{group}"})
    hashes = {}
    unique = []
    for record in records:
        digest = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
        record["sha256"] = digest
        if digest in hashes:
            if hashes[digest]["label"] != record["label"]:
                raise ValueError(f"Conflicting labels for duplicate {record['path']}")
            exclusions.append({"path": record["path"], "reason": "exact duplicate"})
        else:
            hashes[digest] = record
            unique.append(record)
    if not unique:
        raise ValueError("No labeled recordings found; expected datasets/{PhysioNet,pascal}")
    return unique, exclusions


def split(records: list[dict], seed: int = 42) -> list[dict]:
    # Group-stratify by source AND class so each source is represented in each partition.
    strata = np.array([f"{r['source']}:{r['label']}" for r in records])
    groups = np.array([r["group"] for r in records])
    partition = np.full(len(records), "train", dtype=object)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(strata, strata, groups))
    partition[folds[0][1]] = "test"
    partition[folds[1][1]] = "validation"
    result = [dict(r, split=str(p)) for r, p in zip(records, partition)]
    for name in ("train", "validation", "test"):
        if {r["label"] for r in result if r["split"] == name} != {0, 1}:
            raise ValueError(f"Partition {name} lacks both classes; use more recordings")
    return result


def write_manifest(root: Path, output: Path, seed: int = 42) -> list[dict]:
    records, excluded = manifest(root)
    result = split(records, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"seed": seed, "grouping_limit": "Recording/acquisition IDs, not verified patient IDs; patient independence cannot be claimed", "records": result, "excluded": excluded}, indent=2))
    return result
