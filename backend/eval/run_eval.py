"""
Evaluation harness for the Quebec Rx Patient Scanner.

WHY THIS FILE EXISTS
--------------------
You cannot call a pipeline "production-grade" until you can answer, with a
number: "Did my change make it better or worse?" This script is that number.

It runs the EXISTING pipeline (ocr_engine.extract_patient_info) over a folder
of labeled sample images and reports, per field (name / dob / ramq):

  - accuracy   : of the images that HAVE a ground-truth value, how many did we
                 get exactly right?
  - precision  : of the times we PRODUCED a value, how many were correct?
                 (high precision = "when it speaks, trust it")
  - recall     : of the values that SHOULD have been found, how many did we find
                 correctly? (high recall = "it rarely misses")
  - CER        : Character Error Rate -- how far off were we, letter by letter,
                 even when not exactly right? (0.0 = perfect, 1.0 = total miss)

For a MEDICAL scanner, precision usually matters more than recall: it is far
safer to say "I'm not sure, please verify" than to confidently print the wrong
patient's date of birth. The harness reports both so you can see the trade-off.

HOW TO RUN
----------
  1. Put your sample images in   backend/eval/images/
  2. Label them in               backend/eval/ground_truth.json
     (copy ground_truth.example.json to get started)
  3. From the backend/ directory:
         python eval/run_eval.py
     Options:
         python eval/run_eval.py --images eval/images --truth eval/ground_truth.json
         python eval/run_eval.py --report eval/last_report.json   # save machine-readable report

This imports the pipeline directly (no web server needed) and loads the
EasyOCR model ONCE, then reuses it for every image -- the same singleton
pattern main.py uses in production.
"""

import argparse
import json
import os
import sys
import time
import unicodedata

# --- Make the backend package importable no matter where we're invoked from ---
# ocr_engine.py lives one directory up from this file (backend/), and it does
# `from logger import write_scan_log`, so backend/ must be on sys.path.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(THIS_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ---------------------------------------------------------------------------
# Normalization: make "comparable" versions of strings so trivial formatting
# differences (spaces, case, accents, separators) don't count as errors.
# ---------------------------------------------------------------------------
def strip_accents(text: str) -> str:
    """ÉLODIE -> ELODIE. OCR and typists disagree on accents constantly."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def norm_name(value):
    if value is None:
        return None
    v = strip_accents(str(value)).upper()
    # collapse all non-letters to single spaces, trim: "Tremblay, Jean" ~ "TREMBLAY JEAN"
    out, prev_space = [], False
    for ch in v:
        if ch.isalpha():
            out.append(ch)
            prev_space = False
        elif not prev_space:
            out.append(" ")
            prev_space = True
    return " ".join("".join(out).split())


def norm_ramq(value):
    if value is None:
        return None
    return "".join(c for c in str(value).upper() if c.isalnum())


def norm_dob(value):
    """Reduce any date to its digits so 1990-05-12 == 1990/05/12 == 19900512."""
    if value is None:
        return None
    return "".join(c for c in str(value) if c.isdigit())


NORMALIZERS = {"name": norm_name, "dob": norm_dob, "ramq": norm_ramq}


# ---------------------------------------------------------------------------
# Character Error Rate via Levenshtein edit distance (no external deps).
# CER = edit_distance(prediction, truth) / len(truth).
# It rewards "almost right" -- a single swapped digit in a RAMQ scores ~0.08,
# not a flat 0% like exact-match would.
# ---------------------------------------------------------------------------
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def char_error_rate(pred, truth) -> float:
    if not truth:
        return 0.0 if not pred else 1.0
    return levenshtein(pred or "", truth) / len(truth)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------
FIELDS = ["name", "dob", "ramq"]


def evaluate(images_dir, truth_path, report_path=None):
    with open(truth_path, "r", encoding="utf-8") as f:
        truth = json.load(f)

    # Allow the example/template to carry a leading comment key.
    truth = {k: v for k, v in truth.items() if not k.startswith("_")}
    if not truth:
        print("No labeled images in ground truth. Add entries to", truth_path)
        return

    # Heavy imports happen only now, so --help etc. stay instant.
    print("Loading OCR model (EasyOCR fr+en)... this is a one-time cost.")
    import easyocr
    from ocr_engine import extract_patient_info, get_device_gpu_status

    reader = easyocr.Reader(["fr", "en"], gpu=get_device_gpu_status(), verbose=False)

    per_image = []
    latencies = []
    failure_reasons = {}

    for fname, labels in truth.items():
        img_path = os.path.join(images_dir, fname)
        if not os.path.isfile(img_path):
            print(f"  [skip] missing image: {img_path}")
            continue

        with open(img_path, "rb") as fimg:
            img_bytes = fimg.read()

        t0 = time.time()
        res = extract_patient_info(img_bytes, reader=reader)
        latencies.append((time.time() - t0) * 1000.0)

        fr = (res.get("diagnostics") or {}).get("failure_reason")
        if fr:
            failure_reasons[fr] = failure_reasons.get(fr, 0) + 1

        row = {"image": fname, "fields": {}}
        for field in FIELDS:
            normalize = NORMALIZERS[field]
            gt_raw = labels.get(field)
            pred_raw = res.get(field)
            gt = normalize(gt_raw) if gt_raw not in (None, "") else None
            pred = normalize(pred_raw) if pred_raw not in (None, "") else None

            row["fields"][field] = {
                "truth": gt_raw,
                "pred": pred_raw,
                "has_truth": gt is not None,
                "predicted": pred is not None,
                "correct": (gt is not None and pred == gt),
                "cer": char_error_rate(pred, gt) if gt is not None else None,
            }
        per_image.append(row)

    _print_report(per_image, latencies, failure_reasons)

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "per_image": per_image,
                    "latency_ms": latencies,
                    "failure_reasons": failure_reasons,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"\nMachine-readable report written to {report_path}")


def _pct(n, d):
    return f"{(100.0 * n / d):5.1f}%" if d else "   n/a"


def _print_report(per_image, latencies, failure_reasons):
    n = len(per_image)
    print("\n" + "=" * 70)
    print(f"EVALUATION REPORT  ({n} images scored)")
    print("=" * 70)

    # Per-image grid
    print(f"\n{'image':<32} {'name':>6} {'dob':>6} {'ramq':>6}")
    print("-" * 56)
    for row in per_image:
        marks = []
        for field in FIELDS:
            c = row["fields"][field]
            if not c["has_truth"]:
                marks.append("  -  ")      # nothing to grade
            elif c["correct"]:
                marks.append("  OK ")
            elif c["predicted"]:
                marks.append(" WRONG")      # produced a value, but wrong (precision hit)
            else:
                marks.append(" MISS ")      # produced nothing (recall hit)
        img = (row["image"][:31]) if len(row["image"]) > 31 else row["image"]
        print(f"{img:<32} {marks[0]:>6} {marks[1]:>6} {marks[2]:>6}")

    # Per-field metrics
    print("\n" + "-" * 70)
    print(f"{'field':<8} {'accuracy':>9} {'precision':>10} {'recall':>8} {'mean CER':>9}")
    print("-" * 70)
    all_correct_count = 0
    for field in FIELDS:
        has_truth = [r["fields"][field] for r in per_image if r["fields"][field]["has_truth"]]
        predicted = [c for c in has_truth if c["predicted"]]
        correct = [c for c in has_truth if c["correct"]]
        cers = [c["cer"] for c in has_truth if c["cer"] is not None]
        mean_cer = f"{(sum(cers) / len(cers)):.3f}" if cers else "  n/a"
        print(
            f"{field:<8} "
            f"{_pct(len(correct), len(has_truth)):>9} "       # accuracy
            f"{_pct(len(correct), len(predicted)):>10} "      # precision
            f"{_pct(len(correct), len(has_truth)):>8} "       # recall
            f"{mean_cer:>9}"
        )

    # Whole-record success: all three fields correct on the same image
    full = 0
    gradable = 0
    for r in per_image:
        fields = [r["fields"][f] for f in FIELDS if r["fields"][f]["has_truth"]]
        if len(fields) == len(FIELDS):
            gradable += 1
            if all(c["correct"] for c in fields):
                full += 1
    print("-" * 70)
    print(f"Full-record accuracy (all 3 fields correct): {_pct(full, gradable)}  "
          f"({full}/{gradable} fully-labeled images)")

    # Latency
    if latencies:
        s = sorted(latencies)
        p50 = s[len(s) // 2]
        p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
        print(f"\nLatency (ms):  mean {sum(s)/len(s):7.1f}   p50 {p50:7.1f}   p95 {p95:7.1f}")

    if failure_reasons:
        print("\nFailure reasons:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"  {count:>3}x  {reason}")
    print("=" * 70)
    print("Legend:  OK = exactly right | WRONG = produced wrong value (hurts precision) |")
    print("         MISS = produced nothing (hurts recall) | - = no ground-truth label")


def main():
    ap = argparse.ArgumentParser(description="Score the OCR pipeline against labeled images.")
    ap.add_argument("--images", default=os.path.join(THIS_DIR, "images"),
                    help="Folder containing sample images (default: eval/images)")
    ap.add_argument("--truth", default=os.path.join(THIS_DIR, "ground_truth.json"),
                    help="JSON mapping image filename -> {name, dob, ramq}")
    ap.add_argument("--report", default=None,
                    help="Optional path to write a machine-readable JSON report")
    args = ap.parse_args()

    if not os.path.isfile(args.truth):
        print(f"Ground-truth file not found: {args.truth}")
        print("Copy eval/ground_truth.example.json to eval/ground_truth.json and fill it in.")
        sys.exit(1)

    evaluate(args.images, args.truth, args.report)


if __name__ == "__main__":
    main()
