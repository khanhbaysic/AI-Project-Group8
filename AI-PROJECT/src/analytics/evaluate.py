"""
Evaluation Harness cho E-Proctoring System
===========================================

Tinh Precision / Recall / F1 cho TUNG trang thai + Confusion Matrix,
bang cach so sanh:
  - GROUND TRUTH (nhan tay cua nhom)  vs
  - PREDICTION   (trang thai he thong xuat ra trong *_details.csv)

HO TRO 2 KIEU GAN NHAN (nhom chon 1 kieu, thong nhat ca nhom):

  KIEU A - "mot nhan cho ca clip" (don gian nhat, khuyen dung cho 4/6):
     File ground truth la CSV:
         clip,true_state
         clip01,OK
         clip02,PHONE_USAGE
         ...
     Va he thong tao 1 file *_details.csv cho moi clip. Khi do truyen
     thu muc chua cac details + file nhan -> moi clip lay trang thai
     pho bien nhat (mode) lam prediction.

  KIEU B - "nhan theo doan thoi gian" (chinh xac hon, cho ban cuoi):
     File ground truth la CSV:
         student,start,end,true_state
         Student_1,0,5,OK
         Student_1,5,12,PHONE_USAGE
         ...
     So khop voi tung dong trong *_details.csv theo (student, timestamp).

CACH CHAY:

  Kieu A:
     py -m src.analytics.evaluate --mode clip ^
        --labels labels_clip.csv --details-dir output/video_analysis

  Kieu B:
     py -m src.analytics.evaluate --mode segment ^
        --labels labels_segment.csv --details output/video_analysis/classroom_test_details.csv

Ket qua: in bang chi so ra man hinh + luu confusion_matrix.png va
evaluation_report.txt.

Khong sua gi trong pipeline. Chi DOC file. Chi can matplotlib (tuy chon).
"""

import argparse
import csv
import sys
from collections import defaultdict, Counter
from pathlib import Path

from src.states import EVAL_STATES as STATES


# ---------------------------------------------------------------------------
# Doc du lieu
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
    """KIEU A: moi clip 1 nhan. Tra ve list (y_true, y_pred)."""
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
            # tim file details tuong ung: <clip>_details.csv
            cand = details_dir / f"{clip}_details.csv"
            if not cand.exists():
                # thu tim file chua ten clip
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
    """KIEU B: nhan theo (student, start-end). Match tung dong details."""
    # doc nhan thanh dict: student -> list (start,end,state)
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
# Tinh chi so
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
    # macro avg chi tinh tren cac lop co ground-truth label.
    # Cac lop support=0 van duoc in trong report de thay ro test set con thieu.
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
# Xuat ket qua
# ---------------------------------------------------------------------------

def format_report(per_class, accuracy, macro, total, labels):
    missing = missing_ground_truth_states(per_class, labels)
    lines = []
    lines.append("=" * 64)
    lines.append(" BAO CAO DANH GIA (EVALUATION REPORT)")
    lines.append("=" * 64)
    lines.append(f" Tong so mau (samples): {total}")
    lines.append(f" Accuracy             : {accuracy*100:.2f}%")
    lines.append(
        " States without ground-truth labels: "
        + (", ".join(missing) if missing else "None")
    )
    lines.append("-" * 64)
    lines.append(f" {'Trang thai':<14}{'Precision':>10}{'Recall':>10}"
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
    ax.set_xlabel("Du doan (Predicted)", color="#e2e8f0")
    ax.set_ylabel("Thuc te (Ground truth)", color="#e2e8f0")
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
    ap.add_argument("--labels", required=True, help="File CSV nhan ground truth")
    ap.add_argument("--details-dir", help="(mode clip) thu muc chua *_details.csv")
    ap.add_argument("--details", help="(mode segment) duong dan 1 file *_details.csv")
    ap.add_argument("--out-dir", default=".", help="Thu muc luu ket qua")
    args = ap.parse_args()

    if args.mode == "clip":
        if not args.details_dir:
            ap.error("mode clip can --details-dir")
        pairs, info = collect_pairs_clip(args.labels, args.details_dir)
        if info:
            print(f"[CANH BAO] Khong tim thay details cho clip: {info}")
    else:
        if not args.details:
            ap.error("mode segment can --details")
        pairs, info = collect_pairs_segment(args.labels, args.details)
        if info:
            print(f"[INFO] So dong details khong khop nhan: {info}")

    if not pairs:
        print("[LOI] Khong co cap (true,pred) nao. Kiem tra lai file nhan.")
        print("      Neu file duoc tao tu make_label_template.py, hay dien cot true_state truoc.")
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
        print(f"\n[OK] Da luu confusion matrix: {png}")
    print(f"[OK] Da luu bao cao text   : {out_dir / 'evaluation_report.txt'}")


if __name__ == "__main__":
    main()
