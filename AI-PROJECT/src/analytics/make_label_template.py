r"""Create a segment-label template from a *_details.csv file.

Usage:
    py -3.12 -m src.analytics.make_label_template ^
        output\video_analysis\classroom_test_details.csv

The output CSV uses the existing evaluation schema:

    student,start,end,true_state

Fill ``true_state`` manually with one of the canonical states printed by this
script, then run ``src.analytics.evaluate --mode segment``.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from src.states import EVAL_STATES


FIELDS = ["student", "start", "end", "true_state"]


def read_details(details_csv):
    rows = []
    with open(details_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            student = (row.get("student") or "").strip()
            if not student:
                continue
            try:
                timestamp = float(row.get("timestamp", 0) or 0)
            except ValueError:
                continue
            rows.append({"student": student, "timestamp": timestamp})
    return rows


def build_template_rows(detail_rows):
    by_student = defaultdict(list)
    for row in detail_rows:
        by_student[row["student"]].append(row["timestamp"])

    template = []
    for student in sorted(by_student):
        times = by_student[student]
        start = min(times)
        end = max(times)
        template.append({
            "student": student,
            "start": f"{start:.3f}",
            "end": f"{end:.3f}",
            "true_state": "",
        })
    return template


def write_template(rows, output_csv, overwrite=False):
    output_csv = Path(output_csv)
    if output_csv.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_csv}. Use --overwrite or choose --output."
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def default_output_for(details_csv):
    details_csv = Path(details_csv)
    return details_csv.with_name("labels_segment_template.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Create a segment-label CSV template from a details CSV."
    )
    parser.add_argument("details", help="Path to *_details.csv")
    parser.add_argument(
        "--output",
        help="Output CSV path. Default: labels_segment_template.csv next to details.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    args = parser.parse_args()

    details_csv = Path(args.details)
    if not details_csv.exists():
        print(f"[ERROR] Details CSV not found: {details_csv}")
        return 1

    detail_rows = read_details(details_csv)
    if not detail_rows:
        print(f"[ERROR] No student/timestamp rows found in: {details_csv}")
        return 1

    output_csv = Path(args.output) if args.output else default_output_for(details_csv)
    template_rows = build_template_rows(detail_rows)
    try:
        write_template(template_rows, output_csv, args.overwrite)
    except FileExistsError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print("=" * 64)
    print(" LABEL TEMPLATE CREATED")
    print("=" * 64)
    print(f" Details CSV : {details_csv}")
    print(f" Output CSV  : {output_csv}")
    print(f" Students    : {len(template_rows)}")
    print(" Valid states: " + ", ".join(EVAL_STATES))
    print()
    print("Next steps:")
    print("  1. Open the output CSV and fill true_state for each segment.")
    print("  2. Split rows manually if a student changes behavior mid-video.")
    print("  3. Run:")
    print(
        "     py -3.12 -m src.analytics.evaluate --mode segment "
        f"--labels {output_csv} --details {details_csv}"
    )
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
