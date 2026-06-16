"""
batch_evaluate.py
=================
Batch runner for the E-Proctoring system.

Usage:
    python batch_evaluate.py [--show] [--skip-processed] [--stats-only]

Steps:
  1. Discover every test_video* folder under input/
  2. Find the video file (.mp4 / .MOV / .avi) and its ground-truth CSV
  3. Run video_analyzer.py on each pair (unless --skip-processed is set and
     the output details CSV already exists)
  4. Compare predictions vs ground-truth and compute accuracy metrics per video
  5. Print + save a summary statistics table (output/batch_accuracy_report.txt)
     and a styled HTML table (output/batch_accuracy_report.html)

Ground-truth CSV formats supported
-----------------------------------
  Format A (standard): student, start, end, true_state
      student start/end are SECONDS (integer or float).
  Format B (old):      start, end, model_prediction, actual_state
      start/end are "M:SS" strings; actual_state may be empty -> skipped.
"""

import argparse
import csv
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# States used for matching
ALL_STATES = ["OK", "DISTRACTED", "TALKING", "SLEEPING", "ABSENT", "PHONE_USAGE", "BODY_ONLY"]


# ---------------------------------------------------------------------------
# Helper: parse ground-truth CSVs
# ---------------------------------------------------------------------------

def _mmss_to_seconds(s: str) -> float:
    """Convert 'M:SS' or 'MM:SS' or plain seconds string to float."""
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60.0 + seconds
    return float(s)


def parse_ground_truth(csv_path: Path) -> list[dict]:
    """
    Return a list of dicts:
        {"student": str, "start": float, "end": float, "true_state": str}

    Handles both CSV format variants automatically.
    Rows with empty / missing true_state are skipped.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return []

    first_keys = set(rows[0].keys())
    # Detect Format B by the presence of 'actual_state' column
    if "actual_state" in first_keys:
        segments = []
        for row in rows:
            true_state = (row.get("actual_state") or "").strip()
            if not true_state:
                continue
            try:
                start = _mmss_to_seconds(row["start"])
                end = _mmss_to_seconds(row["end"])
            except (KeyError, ValueError):
                continue
            segments.append({
                "student": "Student_1",   # Format B is always single-student
                "start": start,
                "end": end,
                "true_state": true_state,
            })
        return segments
    else:
        # Format A: student, start, end, true_state
        # NOTE: `end` is EXCLUSIVE, matching the built-in evaluate.py logic:
        #   s <= t < end   (same as collect_pairs_segment in evaluate.py)
        segments = []
        for row in rows:
            true_state = (row.get("true_state") or "").strip()
            if not true_state:
                continue
            student = (row.get("student") or "Student_1").strip()
            try:
                start = float(row.get("start") or 0)
                end = float(row.get("end") or start)
            except ValueError:
                continue
            segments.append({
                "student": student,
                "start": start,
                "end": end,   # exclusive: s <= t < end
                "true_state": true_state,
            })
        return segments


def true_state_at(student: str, t: float, segments: list[dict]):
    """Return the ground-truth state for (student, time t), or None."""
    for seg in segments:
        if seg["student"] == student and seg["start"] <= t < seg["end"]:
            return seg["true_state"]
    return None


# ---------------------------------------------------------------------------
# Read prediction details CSV
# ---------------------------------------------------------------------------

def read_details(details_csv: Path) -> list[dict]:
    """Return rows from a *_details.csv as list of dicts."""
    rows = []
    with open(details_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "t": float(r.get("timestamp") or 0),
                "student": (r.get("student") or "Student_1").strip(),
                "state": (r.get("state") or "").strip(),
            })
    return rows


# ---------------------------------------------------------------------------
# Evaluate one video
# ---------------------------------------------------------------------------

def evaluate_video(details_csv: Path, ground_truth_csv: Path) -> dict:
    """
    Compare prediction vs ground truth for one video.

    Returns a dict:
        accuracy, total, correct, per_state metrics, …
    """
    segments = parse_ground_truth(ground_truth_csv)
    if not segments:
        return {"error": "No usable ground-truth segments"}

    details = read_details(details_csv)
    if not details:
        return {"error": "Details CSV is empty"}

    pairs = []
    unmatched = 0
    for d in details:
        if not d["state"]:
            continue
        yt = true_state_at(d["student"], d["t"], segments)
        if yt is None:
            unmatched += 1
            continue
        pairs.append((yt, d["state"]))

    if not pairs:
        return {
            "error": (
                f"No matching (true, pred) pairs "
                f"(details rows={len(details)}, segments={len(segments)}, "
                f"unmatched={unmatched})"
            )
        }

    total = len(pairs)
    correct = sum(1 for yt, yp in pairs if yt == yp)
    accuracy = correct / total if total else 0.0

    # Per-state
    per_state: dict[str, dict] = {}
    for state in ALL_STATES:
        tp = sum(1 for yt, yp in pairs if yt == state and yp == state)
        fp = sum(1 for yt, yp in pairs if yt != state and yp == state)
        fn = sum(1 for yt, yp in pairs if yt == state and yp != state)
        support = tp + fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_state[state] = {
            "precision": prec, "recall": rec, "f1": f1, "support": support
        }

    present = [s for s in ALL_STATES if per_state[s]["support"] > 0]
    macro_acc = (
        sum(per_state[s]["recall"] for s in present) / len(present)
        if present else 0.0
    )

    return {
        "total": total,
        "correct": correct,
        "unmatched": unmatched,
        "accuracy": accuracy,
        "macro_recall": macro_acc,
        "per_state": per_state,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Discover input folders
# ---------------------------------------------------------------------------

def discover_inputs() -> list[dict]:
    """
    Walk INPUT_DIR for test_video* sub-directories.
    Returns list of {"folder", "video", "gt_csv"} dicts.
    """
    entries = []
    for folder in sorted(INPUT_DIR.glob("test_video*")):
        if not folder.is_dir():
            continue
        # find video file
        videos = [
            f for f in folder.iterdir()
            if f.suffix.lower() in VIDEO_EXTENSIONS
        ]
        if not videos:
            print(f"[SKIP] {folder.name}: no video file found")
            continue
        video = videos[0]

        # find CSV — any .csv file in the folder
        csvs = [f for f in folder.iterdir() if f.suffix.lower() == ".csv"]
        if not csvs:
            print(f"[SKIP] {folder.name}: no CSV file found")
            continue
        gt_csv = csvs[0]

        entries.append({
            "folder": folder,
            "video": video,
            "gt_csv": gt_csv,
        })
    return entries


# ---------------------------------------------------------------------------
# Run video_analyzer
# ---------------------------------------------------------------------------

def run_video_analyzer(video: Path, output_dir: Path, show=False) -> Path | None:
    """
    Invoke video_analyzer.analyze_video() for *video* and return the
    path to the produced *_details.csv, or None on failure.
    """
    try:
        from src.video_analyzer import analyze_video
        result = analyze_video(video, show=show, output_dir=output_dir)
        details_csv = Path(result["detail_csv"])
        return details_csv if details_csv.exists() else None
    except Exception as exc:
        print(f"  [ERROR] video_analyzer failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_summary_table(results: list[dict]) -> str:
    """Print and return the plain-text summary table."""
    header = (
        f"{'Video':<20} {'Status':<10} {'Total':>6} {'Correct':>7} "
        f"{'Accuracy':>9} {'MacroRec':>9}"
    )
    sep = "-" * len(header)
    lines = ["", "=" * len(header), "  BATCH EVALUATION RESULTS", "=" * len(header), header, sep]

    for r in results:
        name = r["folder"].name
        if r.get("error"):
            lines.append(f"  {name:<20} {'ERROR':<10}  {r['error']}")
        else:
            status = "OK"
            lines.append(
                f"  {name:<20} {status:<10} {r['total']:>6} {r['correct']:>7} "
                f"{_pct(r['accuracy']):>9} {_pct(r['macro_recall']):>9}"
            )

    # Overall aggregate
    valid = [r for r in results if not r.get("error")]
    if valid:
        total_all = sum(r["total"] for r in valid)
        correct_all = sum(r["correct"] for r in valid)
        overall_acc = correct_all / total_all if total_all else 0.0
        avg_macro = sum(r["macro_recall"] for r in valid) / len(valid)
        lines.append(sep)
        lines.append(
            f"  {'OVERALL':<20} {'':<10} {total_all:>6} {correct_all:>7} "
            f"{_pct(overall_acc):>9} {_pct(avg_macro):>9}"
        )

    lines += ["=" * len(header), ""]
    report = "\n".join(lines)
    print(report)
    return report


def generate_html_report(results: list[dict]) -> str:
    """Build a beautiful HTML page with the accuracy statistics."""
    valid = [r for r in results if not r.get("error")]
    total_all = sum(r["total"] for r in valid)
    correct_all = sum(r["correct"] for r in valid)
    overall_acc = correct_all / total_all if total_all else 0.0
    avg_macro = sum(r["macro_recall"] for r in valid) / len(valid) if valid else 0.0

    def color_for_acc(v: float) -> str:
        if v >= 0.75:
            return "#22c55e"
        if v >= 0.50:
            return "#f59e0b"
        return "#ef4444"

    def bar(v: float) -> str:
        pct = int(v * 100)
        color = color_for_acc(v)
        return (
            f'<div style="background:#1e293b;border-radius:4px;height:8px;width:100%;">'
            f'<div style="background:{color};width:{pct}%;height:8px;border-radius:4px;"></div></div>'
        )

    rows_html = ""
    for r in results:
        name = r["folder"].name
        if r.get("error"):
            rows_html += (
                f'<tr><td>{name}</td><td colspan="5" style="color:#ef4444;">'
                f'ERROR: {r["error"]}</td></tr>'
            )
        else:
            acc_c = color_for_acc(r["accuracy"])
            mac_c = color_for_acc(r["macro_recall"])
            rows_html += f"""
            <tr>
              <td>{name}</td>
              <td style="color:#38bdf8;">{r['video'].name}</td>
              <td style="text-align:right;">{r['total']:,}</td>
              <td style="text-align:right;">{r['correct']:,}</td>
              <td>
                <span style="color:{acc_c};font-weight:700;">{_pct(r['accuracy'])}</span>
                {bar(r['accuracy'])}
              </td>
              <td>
                <span style="color:{mac_c};font-weight:700;">{_pct(r['macro_recall'])}</span>
                {bar(r['macro_recall'])}
              </td>
            </tr>"""

    # Per-state breakdown for valid results
    state_rows = ""
    if valid:
        agg_state: dict[str, dict] = {s: {"tp": 0, "fp": 0, "fn": 0} for s in ALL_STATES}
        for r in valid:
            for s in ALL_STATES:
                ps = r["per_state"][s]
                support = ps["support"]
                # reconstruct tp/fn from recall and support
                tp = round(ps["recall"] * support) if support else 0
                fn = support - tp
                agg_state[s]["tp"] += tp
                agg_state[s]["fn"] += fn

        for state in ALL_STATES:
            d = agg_state[state]
            tp, fn = d["tp"], d["fn"]
            support = tp + fn
            if support == 0:
                rec_str = "<em>—</em>"
            else:
                rec = tp / support if support else 0.0
                color = color_for_acc(rec)
                rec_str = f'<span style="color:{color};font-weight:700;">{_pct(rec)}</span>'
            state_rows += f"<tr><td>{state}</td><td style='text-align:right'>{support:,}</td><td>{rec_str}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>E-Proctoring Batch Evaluation Report</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{background:#0b1220;color:#e2e8f0;font-family:'Inter',sans-serif;padding:32px;min-height:100vh;}}
    h1{{color:#38bdf8;font-size:1.8rem;margin-bottom:4px;}}
    .subtitle{{color:#64748b;font-size:.9rem;margin-bottom:28px;}}
    .cards{{display:flex;gap:16px;margin-bottom:32px;flex-wrap:wrap;}}
    .card{{background:#1e293b;border-radius:12px;padding:20px 28px;min-width:150px;flex:1;}}
    .card-label{{color:#64748b;font-size:.8rem;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;}}
    .card-value{{font-size:2rem;font-weight:700;}}
    table{{width:100%;border-collapse:collapse;background:#1e293b;border-radius:12px;overflow:hidden;margin-bottom:32px;}}
    thead tr{{background:#0f172a;}}
    th{{padding:12px 16px;text-align:left;color:#94a3b8;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em;}}
    td{{padding:12px 16px;border-bottom:1px solid #0f172a;font-size:.9rem;}}
    tr:last-child td{{border-bottom:none;}}
    tr:hover td{{background:#243044;}}
    .section-title{{font-size:1.1rem;font-weight:600;color:#cbd5e1;margin-bottom:12px;}}
    .footer{{color:#334155;font-size:.8rem;margin-top:32px;}}
  </style>
</head>
<body>
  <h1>🎓 E-Proctoring Batch Evaluation Report</h1>
  <p class="subtitle">Generated on {time.strftime('%Y-%m-%d %H:%M:%S')} &nbsp;·&nbsp; {len(results)} videos evaluated</p>

  <div class="cards">
    <div class="card">
      <div class="card-label">Videos</div>
      <div class="card-value" style="color:#38bdf8;">{len(results)}</div>
    </div>
    <div class="card">
      <div class="card-label">Total Frames Evaluated</div>
      <div class="card-value" style="color:#a78bfa;">{total_all:,}</div>
    </div>
    <div class="card">
      <div class="card-label">Overall Accuracy</div>
      <div class="card-value" style="color:{color_for_acc(overall_acc)};">{_pct(overall_acc)}</div>
    </div>
    <div class="card">
      <div class="card-label">Avg Macro Recall</div>
      <div class="card-value" style="color:{color_for_acc(avg_macro)};">{_pct(avg_macro)}</div>
    </div>
  </div>

  <p class="section-title">Per-Video Accuracy</p>
  <table>
    <thead>
      <tr>
        <th>Test Folder</th>
        <th>Video File</th>
        <th>Frames</th>
        <th>Correct</th>
        <th>Accuracy</th>
        <th>Macro Recall</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <p class="section-title">Per-State Recall (aggregated)</p>
  <table style="max-width:480px;">
    <thead>
      <tr>
        <th>State</th>
        <th>Ground-Truth Frames</th>
        <th>Recall</th>
      </tr>
    </thead>
    <tbody>
      {state_rows}
    </tbody>
  </table>

  <p class="footer">E-Proctoring System · AI Project Group 8 · batch_evaluate.py</p>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate all test videos")
    parser.add_argument("--show", action="store_true",
                        help="Show annotated video window while processing")
    parser.add_argument("--skip-processed", action="store_true",
                        help="Skip videos whose output details CSV already exists")
    parser.add_argument("--stats-only", action="store_true",
                        help="Skip running video_analyzer; only compute stats from existing outputs")
    args = parser.parse_args()

    entries = discover_inputs()
    if not entries:
        print("[ERROR] No test_video* folders found under input/")
        sys.exit(1)

    print(f"\n[INFO] Found {len(entries)} video(s) to process.\n")

    results = []
    for i, entry in enumerate(entries, 1):
        folder: Path = entry["folder"]
        video: Path  = entry["video"]
        gt_csv: Path = entry["gt_csv"]

        print(f"[{i}/{len(entries)}] {folder.name}  <-  {video.name}")
        print(f"        GT CSV : {gt_csv.name}")

        # ---- Where the details CSV will be written ----
        output_dir = OUTPUT_DIR / "video_analysis" / video.stem
        details_csv = output_dir / f"{video.stem}_details.csv"

        # ---- Run or skip ----
        if args.stats_only:
            if not details_csv.exists():
                print(f"  [SKIP] Details CSV not found: {details_csv}")
                results.append({"folder": folder, "video": video, "error": "No details CSV"})
                continue
            print(f"  [OK] Using existing: {details_csv.name}")
        elif args.skip_processed and details_csv.exists():
            print(f"  [SKIP] Already processed: {details_csv.name}")
        else:
            print(f"  [RUN] Running video_analyzer …")
            t0 = time.time()
            produced = run_video_analyzer(video, output_dir, show=args.show)
            elapsed = time.time() - t0
            if produced is None:
                print(f"  [ERROR] video_analyzer did not produce a details CSV")
                results.append({"folder": folder, "video": video, "error": "video_analyzer failed"})
                continue
            details_csv = produced
            print(f"  [OK] Done in {elapsed:.1f}s  →  {details_csv.name}")

        # ---- Evaluate ----
        print(f"  [EVAL] Comparing predictions vs ground truth …")
        eval_result = evaluate_video(details_csv, gt_csv)
        eval_result["folder"] = folder
        eval_result["video"]  = video
        results.append(eval_result)
        if eval_result.get("error"):
            print(f"  [WARN] {eval_result['error']}")
        else:
            print(
                f"  [RESULT] Accuracy={_pct(eval_result['accuracy'])}  "
                f"MacroRecall={_pct(eval_result['macro_recall'])}  "
                f"({eval_result['correct']}/{eval_result['total']} frames)"
            )
        print()

    # ---- Summary ----
    report_txt = print_summary_table(results)

    # Save plain-text report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = OUTPUT_DIR / "batch_accuracy_report.txt"
    txt_path.write_text(report_txt, encoding="utf-8")
    print(f"[OK] Text report saved : {txt_path}")

    # Save HTML report
    html_content = generate_html_report(results)
    html_path = OUTPUT_DIR / "batch_accuracy_report.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"[OK] HTML report saved : {html_path}")


if __name__ == "__main__":
    main()
