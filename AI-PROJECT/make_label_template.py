"""make_label_template.py — Generate a ground-truth label template from a *_details.csv.

Usage:
    python make_label_template.py <details.csv> [--step 1.0]

Output:
    <video>_labels.csv  (same directory as the details file)

The generated CSV has one row per N-second interval (default 1 second) for
each student detected in the details file.  The ``true_state`` column is
left blank for you to fill in manually.

Example output:
    student,start,end,true_state
    Student_1,0,1,
    Student_1,1,2,
    ...

Labelling advice
----------------
- Use 1-second intervals (the default) — no need for finer granularity.
  The evaluator skips frames near state-transition boundaries automatically
  (default margin: ±1.5 s), so coarse 1-second annotations already yield
  clean precision/recall numbers.

- Valid states: OK, DISTRACTED, TALKING, SLEEPING, ABSENT, PHONE_USAGE, BODY_ONLY

- At transition points, round to the nearest second — the boundary margin
  will handle the uncertainty on both sides.

- Leave rows blank (empty true_state) to exclude them from evaluation entirely.
  Use this for ambiguous moments you cannot confidently label.
"""

import argparse
import csv
import sys
from pathlib import Path


VALID_STATES = {"OK", "DISTRACTED", "TALKING", "SLEEPING",
                "ABSENT", "PHONE_USAGE", "BODY_ONLY"}


def make_template(details_csv, step=1.0):
    details_csv = Path(details_csv)
    if not details_csv.exists():
        print(f"[ERROR] File not found: {details_csv}")
        sys.exit(1)

    # Collect students and their timestamp range
    students = {}   # name -> (t_min, t_max)
    with open(details_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("student", "?").strip()
            t = float(row.get("timestamp", 0) or 0)
            if name not in students:
                students[name] = [t, t]
            else:
                students[name][0] = min(students[name][0], t)
                students[name][1] = max(students[name][1], t)

    if not students:
        print("[ERROR] No student data found in details CSV.")
        sys.exit(1)

    # Build rows
    rows = []
    for name in sorted(students):
        t_min, t_max = students[name]
        t = t_min
        while t < t_max:
            start = round(t, 3)
            end   = round(min(t + step, t_max), 3)
            rows.append({"student": name, "start": start, "end": end, "true_state": ""})
            t += step

    # Write output
    stem = details_csv.stem
    if stem.endswith("_details"):
        stem = stem[: -len("_details")]
    out_path = details_csv.parent / f"{stem}_labels.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["student", "start", "end", "true_state"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Template written: {out_path}")
    print(f"     {len(rows)} rows across {len(students)} student(s).")
    print(f"     Fill in the 'true_state' column, then run the evaluator:")
    print(f"     python -m src.analytics.evaluate --mode segment \\")
    print(f"       --labels {out_path} \\")
    print(f"       --details {details_csv} \\")
    print(f"       --margin 1.5")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate ground-truth label template")
    ap.add_argument("details", help="Path to *_details.csv")
    ap.add_argument("--step", type=float, default=1.0,
                    help="Interval size in seconds (default: 1.0)")
    args = ap.parse_args()
    make_template(args.details, step=args.step)
