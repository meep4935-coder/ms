# Evaluation Harness (Phase 0)

This is the measurement tool for the scanner. Every future change — a new
threshold, a different OCR engine, better preprocessing — gets judged here
**before** you keep it. No numbers, no merge.

## The one rule of production CV

> Optimize nothing until you can measure it. A "better" pipeline that you
> can't prove is better is just a different pipeline.

## Setup (one time)

1. Drop your sample images into `images/` (this folder is git-ignored on
   purpose — these are **patient health cards / PHI** and must never be
   committed).
2. Create your labels:
   ```bash
   cp ground_truth.example.json ground_truth.json
   ```
   Edit `ground_truth.json`: one entry per image, with the known-correct
   `name`, `dob`, `ramq`. Delete the `_comment` key and the sample rows.
   - Use the pipeline's output formats: DOB `YYYY-MM-DD`, name
     `Lastname, Firstname`, RAMQ `AAAA 0000 0000`.
   - Set a field to `null` if the image genuinely lacks it — the harness
     won't grade a field it has no label for.
   - **Start small.** 15–25 images covering your real failure modes
     (glare, blur, angle, different card versions) beats 500 clean ones.

## Run

From the `backend/` directory (so the model and imports resolve):

```bash
python eval/run_eval.py
```

Save a machine-readable report to diff against later:

```bash
python eval/run_eval.py --report eval/baseline_report.json
```

## Reading the numbers

Per field you get four metrics. They answer different questions:

| Metric | Question it answers | Why you care (medical) |
|---|---|---|
| **Accuracy** | Of labeled images, how many did we get exactly right? | Headline score. |
| **Precision** | When we DID output a value, how often was it correct? | **The safety metric.** Low precision = confidently wrong patient data. |
| **Recall** | Of values that should be found, how many did we find? | Coverage. Low recall = lots of manual re-entry. |
| **CER** | How wrong were the near-misses, char by char? | Tells apart "one smudged digit" from "total garbage". |

There is a real **precision/recall trade-off**: raising confidence thresholds
makes the scanner abstain more (higher precision, lower recall); lowering them
makes it guess more (higher recall, lower precision). For a tool that fills in
a patient's DOB, you almost always want to **tune for precision and let a human
fill the gaps** — this harness is how you'll find that tuning point in Phase 2.

`WRONG` in the grid hurts precision. `MISS` hurts recall. `-` means you gave no
label for that field, so it's not graded.

## What this does NOT do yet (future phases)

- No bounding-box / localization scoring (only the final field values).
- No per-preprocessing-step attribution (comes in Phase 1 error analysis).
- No automatic regression gate in CI (comes with deployment, Phase 5).
