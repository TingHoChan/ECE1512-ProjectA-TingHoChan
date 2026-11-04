"""Audit persisted Project A results without torch, numpy, or scikit-learn.

This checks saved predictions and selection histories. It does not rerun the
model, inspect HDF5 features, or provide confidence intervals. AUROC is recomputed
by all positive-negative pairs, giving half credit for ties. ACC and macro F1
are computed directly from an integer confusion matrix. CE/focal losses are
reconstructed from CSV probabilities, subject to their rounding/underflow.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from datetime import datetime, timezone


DATASETS = ("C16", "C17", "BRACS")
METHODS = ("baseline", "focal")


def read_csv(path):
    with path.open(encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def label_prediction(row, n_class):
    # range iteration retains the lowest class index for an exact tie, matching
    # the documented argmax decision rule without relying on tensor libraries.
    return int(row["label"]), max(range(n_class), key=lambda c: float(row[f"p{c}"]))


def recompute(rows, n_class, method):
    if not rows:
        raise ValueError("No test predictions")
    cm = [[0] * n_class for _ in range(n_class)]
    probabilities, labels = [], []
    ce_sum, focal_sum, zeros = 0.0, 0.0, 0
    for row in rows:
        label, prediction = label_prediction(row, n_class)
        if not 0 <= label < n_class:
            raise ValueError("Class index outside configured range")
        prob = [float(row[f"p{c}"]) for c in range(n_class)]
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in prob):
            raise ValueError("Nonfinite or out-of-range probability")
        if abs(sum(prob) - 1) > 1e-5:
            raise ValueError("Probabilities fail sum-to-one check")
        probabilities.append(prob)
        labels.append(label)
        cm[label][prediction] += 1
        target_probability = prob[label]
        if target_probability == 0:
            zeros += 1
        else:
            ce = -math.log(target_probability)
            ce_sum += ce
            focal_sum += (1 - target_probability) ** 2 * ce
    per_class, aucs, f1s = [], [], []
    for c in range(n_class):
        support, predicted = sum(cm[c]), sum(r[c] for r in cm)
        tp = cm[c][c]
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        pos = [p[c] for p, y in zip(probabilities, labels) if y == c]
        neg = [p[c] for p, y in zip(probabilities, labels) if y != c]
        auc = (sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
               / (len(pos) * len(neg))) if pos and neg else None
        per_class.append({"class_index": c, "support": support,
                          "predicted_count": predicted,
                          "recall": tp / support if support else None,
                          "f1": f1, "auc": auc})
        aucs.append(auc)
        f1s.append(f1)
    metrics = {"acc": 100 * sum(cm[c][c] for c in range(n_class)) / len(rows),
               "auc": sum(aucs) / n_class if all(a is not None for a in aucs) else None,
               "f1": sum(f1s) / n_class,
               "ce_loss": ce_sum / len(rows) if not zeros else None,
               "loss": (ce_sum if method == "baseline" else focal_sum) / len(rows) if not zeros else None}
    return metrics, cm, per_class, zeros


def audit_run(root, dataset, method):
    folder = root / "results" / f"{dataset}_{method}_seed42"
    path = folder / "metrics.json"
    if not path.exists():
        return {"status": "pending", "reason": "metrics.json is not available"}, None
    result = json.loads(path.read_text(encoding="utf8"))
    if result.get("status") != "complete":
        return {"status": "pending", "reason": "Run is not marked complete"}, None
    checks = {}
    rows = read_csv(folder / "test_predictions.csv")
    splits = json.loads((folder / "splits.json").read_text(encoding="utf8"))
    hist = read_csv(folder / "history.csv")
    n_class = result["config"]["n_class"]
    metrics, cm, per_class, zeros = recompute(rows, n_class, method)
    tolerances = {"acc": 1e-9, "auc": 1e-9, "f1": 1e-9,
                  "ce_loss": 5e-5, "loss": 5e-5}
    errors = {}
    for name, computed in metrics.items():
        expected = result["test"].get(name)
        if computed is None and zeros and name in ("ce_loss", "loss"):
            # Softmax CSV cannot recover a finite logit-space loss after underflow.
            errors[name] = None
            continue
        checks[f"test_{name}_matches"] = close(computed, expected, tolerances[name])
        errors[name] = abs(computed - expected) if computed is not None and expected is not None else None
    slide_ids = [r["slide_id"] for r in rows]
    checks["test_ids_unique"] = len(slide_ids) == len(set(slide_ids))
    checks["test_ids_match_manifest"] = set(slide_ids) == set(splits["test"]["slides"])
    checks["test_size_matches_manifest"] = len(rows) == splits["test"]["n"]
    counts = Counter(int(r["label"]) for r in rows)
    checks["test_class_counts_match_manifest"] = [counts[c] for c in range(n_class)] == splits["test"]["class_counts"]
    sets = [set(splits[s]["slides"]) for s in ("train", "val", "test")]
    checks["split_slide_ids_disjoint"] = all(not sets[i] & sets[j] for i, j in ((0,1),(0,2),(1,2)))
    checks["split_hash_matches_run"] = splits["split_sha256"] == result["split_sha256"]
    checks["30_epochs_recorded"] = len(hist) == result["config"]["train_epoch"] == 30
    checks["epochs_sequential"] = [int(r["epoch"]) for r in hist] == list(range(1, len(hist)+1))
    scores = [float(r["val_auc"]) + float(r["val_f1"]) for r in hist]
    selected = max(range(len(scores)), key=lambda i: scores[i])
    checks["earliest_max_validation_epoch_matches"] = selected + 1 == result["best_epoch"]
    checks["selection_score_matches"] = close(scores[selected], result["validation_score"], 1e-12)
    checks["current_source_hashes_match_run"] = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in result.get("source_sha256", {}).items())
    audit = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
             "recomputed_test": metrics, "recorded_test": result["test"],
             "absolute_metric_differences": errors, "tolerances": tolerances,
             "confusion_matrix": cm, "per_class": per_class,
             "csv_target_probability_underflows": zeros,
             "earliest_best_epoch_recomputed": selected+1,
             "test_n": len(rows), "split_sha256": result["split_sha256"],
             "patient_overlap_train_val_n": len(splits.get("patient_overlap_train_val", [])) if "patient_overlap_train_val" in splits else None,
             "patient_overlap_train_test_n": len(splits.get("patient_overlap_train_test", [])) if "patient_overlap_train_test" in splits else None,
             "sources": [str((folder / f).relative_to(root)) for f in ("metrics.json", "history.csv", "test_predictions.csv", "splits.json")]}
    return audit, (result, rows)


def audit_pair(a, b):
    if a is None or b is None:
        return {"status": "pending"}
    am, ar = a
    bm, br = b
    a_rows, b_rows = {r["slide_id"]: r for r in ar}, {r["slide_id"]: r for r in br}
    checks = {"same_split_hash": am["split_sha256"] == bm["split_sha256"],
              "same_test_slide_ids": a_rows.keys() == b_rows.keys(),
              "same_config": am["config"] == bm["config"],
              "same_trainable_parameter_count": am["parameters"] == bm["parameters"],
              "same_seed": am["seed"] == bm["seed"] == 42}
    if not checks["same_test_slide_ids"]:
        return {"status": "failed", "checks": checks}
    checks["same_test_labels"] = all(a_rows[s]["label"] == b_rows[s]["label"] for s in a_rows)
    corrected, lost = [], []
    n_class = am["config"]["n_class"]
    for slide in a_rows:
        ay, ap = label_prediction(a_rows[slide], n_class)
        by, bp = label_prediction(b_rows[slide], n_class)
        if ap != ay and bp == by:
            corrected.append(slide)
        if ap == ay and bp != by:
            lost.append(slide)
    delta = {k: bm["test"][k] - am["test"][k] for k in ("acc", "auc", "f1", "ce_loss")}
    checks["accuracy_delta_equals_net_corrected"] = close(delta["acc"], 100 * (len(corrected)-len(lost))/len(ar), 1e-9)
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
            "focal_minus_baseline": delta, "corrected_count": len(corrected),
            "lost_count": len(lost), "corrected_slide_ids": corrected, "lost_slide_ids": lost}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "output" / "independent_metrics_audit.json"
    audits, payload = {}, {}
    for dataset in DATASETS:
        for method in METHODS:
            key = f"{dataset}_{method}_seed42"
            try:
                audits[key], payload[dataset, method] = audit_run(root, dataset, method)
            except Exception as exc:
                audits[key], payload[dataset, method] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, None
    paired = {dataset: audit_pair(payload[dataset, "baseline"], payload[dataset, "focal"]) for dataset in DATASETS}
    complete = all(r["status"] == "passed" for r in audits.values())
    failed = any(r["status"] == "failed" for r in [*audits.values(), *paired.values()])
    document = {"status": "failed" if failed else "passed" if complete else "pending",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "method": "Python standard library only; pairwise Mann-Whitney AUROC with half credit for ties; confusion-derived accuracy/macro F1; probability-derived CE/focal",
                "scope": "Audits persisted predictions, split manifests, source hashes, and checkpoint selection histories. Does not rerun models or establish statistical significance.",
                "runs": audits, "paired_comparisons": paired}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({"status": document["status"], "run_statuses": {k:v["status"] for k,v in audits.items()},
                      "pair_statuses": {k:v["status"] for k,v in paired.items()}, "output": str(output)}, indent=2))
    if failed or (args.require_complete and not complete):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

