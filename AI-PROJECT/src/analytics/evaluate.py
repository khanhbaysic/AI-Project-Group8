"""
Evaluation Harness for the E-Proctoring System
===============================================

Computes Precision / Recall / F1 for EACH state + a Confusion Matrix, by
comparing:
  - GROUND TRUTH (the team's manual labels)  vs
  - PREDICTION   (the state the system wrote into *_details.csv)

SUPPORTS 2 LABELLING MODES (the team picks one and uses it consistently):

  MODE A - "one label per clip" (simplest, recommended for 4/6):
     The ground-truth file is a CSV:
         clip,true_state
         clip01,OK
         clip02,PHONE_USAGE
         ...
     The system produces one *_details.csv per clip. You pass the folder
     containing the details + the label file -> each clip takes its most
     frequent state (mode) as the prediction.

  MODE B - "labels by time segment" (more precise, for the final version):
     The ground-truth file is a CSV:
         student,start,end,true_state
         Student_1,0,5,OK
         Student_1,5,12,PHONE_USAGE
         ...
     Matched against each row in *_details.csv by (student, timestamp).

HOW TO RUN:

  Mode A:
     py -m src.analytics.evaluate --mode clip ^
        --labels labels_clip.csv --details-dir output/video_analysis

  Mode B:
     py -m src.analytics.evaluate --mode segment ^
        --labels labels_segment.csv --details output/video_analysis/classroom_test_details.csv

Result: prints the metrics table + saves confusion_matrix.png and
evaluation_report.txt.

Changes nothing in the pipeline. Only READS files. Needs matplotlib (optional).
"""

import argparse
import csv
import sys
from collections import defaultdict, Counter
from pathlib import Path

from src.states import EVAL_STATES as STATES


# ---------------------------------------------------------------------------
# Read data
# ---------------------------------------------------------------------------

def read_details(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "t": float(r.get("timestamp", 0) or 0),
                "student": r.get("student", "?"),
                "state": (r.get("state") or "").strip(),
            })
    return rows


def collect_pairs_clip(labels_csv, details_dir):
    """MODE A: one label per clip. Return a list of (y_true, y_pred)."""
    details_dir = Path(details_dir)
    pairs = []
    missing = []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clip = row["clip"].strip()
            y_true = row["true_state"].strip()
            if not y_true:
                missing.append(f"{clip} (missing true_state)")
                continue
            # find the matching details file: <clip>_details.csv
            cand = details_dir / f"{clip}_details.csv"
            if not cand.exists():
                # otherwise look for a file containing the clip name
                hits = list(details_dir.glob(f"*{clip}*details*.csv"))
                if hits:
                    cand = hits[0]
                else:
                    missing.append(clip)
                    continue
            det = read_details(cand)
            states = [d["state"] for d in det if d["state"]]
            if not states:
                missing.append(clip)
                continue
            y_pred = Counter(states).most_common(1)[0][0]
            pairs.append((y_true, y_pred))
    return pairs, missing


def collect_pairs_segment(labels_csv, details_csv):
    """MODE B: labels by (student, start-end). Match each details row."""
    # read labels into a dict: student -> list of (start,end,state)
    segs = defaultdict(list)
    with open(labels_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            true_state = row["true_state"].strip()
            if not true_state:
                continue
            segs[row["student"].strip()].append((
                float(row["start"]), float(row["end"]),
                true_state
            ))

    def true_state_at(student, t):
        for s, e, st in segs.get(student, []):
            if s <= t < e:
                return st
        return None

    pairs = []
    unmatched = 0
    for d in read_details(details_csv):
        if not d["state"]:
            continue
        yt = true_state_at(d["student"], d["t"])
        if yt is None:
            unmatched += 1
            continue
        pairs.append((yt, d["state"]))
    return pairs, unmatched


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------

def confusion(pairs, labels):
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    M = [[0] * n for _ in range(n)]
    for yt, yp in pairs:
        if yt in idx and yp in idx:
            M[idx[yt]][idx[yp]] += 1
    return M


def metrics_per_class(M, labels):
    n = len(labels)
    out = {}
    total = sum(sum(r) for r in M)
    correct = sum(M[i][i] for i in range(n))
    for i, lab in enumerate(labels):
        tp = M[i][i]
        fp = sum(M[j][i] for j in range(n)) - tp
        fn = sum(M[i][j] for j in range(n)) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        support = tp + fn
        out[lab] = {"precision": prec, "recall": rec, "f1": f1,
                    "support": support}
    accuracy = correct / total if total else 0.0
    # macro avg is computed only over classes that have a ground-truth label.
    # support=0 classes are still printed so the missing test set is visible.
    present = [l for l in labels if out[l]["support"] > 0]
    macro = {
        "precision": sum(out[l]["precision"] for l in present) / len(present) if present else 0,
        "recall": sum(out[l]["recall"] for l in present) / len(present) if present else 0,
        "f1": sum(out[l]["f1"] for l in present) / len(present) if present else 0,
    }
    return out, accuracy, macro, total


def missing_ground_truth_states(per_class, labels):
    """Return states with zero ground-truth support in the evaluated samples."""
    return [lab for lab in labels if per_class[lab]["support"] == 0]


# ---------------------------------------------------------------------------
# Output results
# ---------------------------------------------------------------------------

def format_report(per_class, accuracy, macro, total, labels):
    missing = missing_ground_truth_states(per_class, labels)
    lines = []
    lines.append("=" * 64)
    lines.append(" EVALUATION REPORT")
    lines.append("=" * 64)
    lines.append(f" Total samples        : {total}")
    lines.append(f" Accuracy             : {accuracy*100:.2f}%")
    lines.append(
        " States without ground-truth labels: "
        + (", ".join(missing) if missing else "None")
    )
    lines.append("-" * 64)
    lines.append(f" {'State':<14}{'Precision':>10}{'Recall':>10}"
                 f"{'F1':>8}{'Support':>9}")
    lines.append("-" * 64)
    for lab in labels:
        m = per_class[lab]
        if m["support"] == 0:
            lines.append(f" {lab:<14}{'NO LABELS':>28}{m['support']:>9}")
            continue
        lines.append(f" {lab:<14}{m['precision']:>10.3f}{m['recall']:>10.3f}"
                     f"{m['f1']:>8.3f}{m['support']:>9}")
    lines.append("-" * 64)
    lines.append(f" {'MACRO AVG*':<14}{macro['precision']:>10.3f}"
                 f"{macro['recall']:>10.3f}{macro['f1']:>8.3f}")
    lines.append(" *Macro avg is computed only over states with labels.")
    lines.append("=" * 64)
    return "\n".join(lines)


def save_confusion_png(M, labels, out_path):
    present = [i for i, l in enumerate(labels) if sum(M[i]) > 0 or
               any(M[j][i] for j in range(len(labels)))]
    if not present:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    labs = [labels[i] for i in present]
    data = np.array([[M[i][j] for j in present] for i in present], dtype=float)

    fig, ax = plt.subplots(figsize=(1.1 * len(labs) + 2, 1.0 * len(labs) + 2),
                           dpi=130)
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")
    im = ax.imshow(data, cmap="cividis")
    ax.set_xticks(range(len(labs)))
    ax.set_yticks(range(len(labs)))
    ax.set_xticklabels(labs, rotation=45, ha="right", color="#e2e8f0", fontsize=9)
    ax.set_yticklabels(labs, color="#e2e8f0", fontsize=9)
    ax.set_xlabel("Predicted", color="#e2e8f0")
    ax.set_ylabel("Ground truth", color="#e2e8f0")
    ax.set_title("Confusion Matrix", color="#38bdf8", fontweight="bold", pad=12)
    mx = data.max() if data.max() > 0 else 1
    for i in range(len(labs)):
        for j in range(len(labs)):
            v = int(data[i, j])
            ax.text(j, i, str(v), ha="center", va="center",
                    color="#ffffff" if data[i, j] < mx * 0.6 else "#000000",
                    fontsize=9)
    for sp in ax.spines.values():
        sp.set_color("#1e293b")
    fig.tight_layout()
    fig.savefig(out_path, facecolor="#0b1220", bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Evaluation harness e-proctoring")
    ap.add_argument("--mode", choices=["clip", "segment"], required=True)
    ap.add_argument("--labels", required=True, help="Ground-truth label CSV file")
    ap.add_argument("--details-dir", help="(clip mode) folder containing *_details.csv")
    ap.add_argument("--details", help="(segment mode) path to one *_details.csv file")
    ap.add_argument("--out-dir", default=".", help="Folder to save results")
    args = ap.parse_args()

    if args.mode == "clip":
        if not args.details_dir:
            ap.error("clip mode needs --details-dir")
        pairs, info = collect_pairs_clip(args.labels, args.details_dir)
        if info:
            print(f"[WARNING] Could not find details for clips: {info}")
    else:
        if not args.details:
            ap.error("segment mode needs --details")
        pairs, info = collect_pairs_segment(args.labels, args.details)
        if info:
            print(f"[INFO] Details rows not matching any label: {info}")

    if not pairs:
        print("[ERROR] No (true,pred) pairs found. Check the label file.")
        print("      If the file came from make_label_template.py, fill the true_state column first.")
        sys.exit(1)

    M = confusion(pairs, STATES)
    per_class, acc, macro, total = metrics_per_class(M, STATES)
    report = format_report(per_class, acc, macro, total, STATES)
    print(report)
    missing_states = missing_ground_truth_states(per_class, STATES)
    print(
        "[INFO] States with no labels: "
        + (", ".join(missing_states) if missing_states else "None")
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation_report.txt").write_text(report, encoding="utf-8")
    png = out_dir / "confusion_matrix.png"
    if save_confusion_png(M, STATES, png):
        print(f"\n[OK] Saved confusion matrix: {png}")
    print(f"[OK] Saved text report     : {out_dir / 'evaluation_report.txt'}")


if __name__ == "__main__":
    main()