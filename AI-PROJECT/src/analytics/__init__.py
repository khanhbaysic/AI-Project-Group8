"""Post-analysis utilities shared by video_analyzer and main.

After a session (video or webcam), call ``run_post_analytics(detail_csv)``
to auto-generate inside ``AI-PROJECT/analysis_statistics/``:
  1. Attention heatmap + session report  (always)
  2. Confusion matrix + evaluation report (only if a labels file is found)
"""

from pathlib import Path


def _get_stats_dir():
    """Return ``PROJECT_ROOT / "analysis_statistics"``, creating it if needed."""
    from src.config import PROJECT_ROOT
    stats_dir = PROJECT_ROOT / "analysis_statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    return stats_dir


def run_post_analytics(detail_csv, labels_csv=None):
    """Generate heatmap and (optionally) confusion matrix from a details CSV.

    All outputs are written to ``AI-PROJECT/analysis_statistics/``.

    Parameters
    ----------
    detail_csv : str or Path
        Path to the ``*_details.csv`` produced by video_analyzer or main.
    labels_csv : str or Path or None
        Optional explicit ground-truth segment labels CSV. If omitted, the
        usual sibling/root lookup is used.
    """
    detail_csv = Path(detail_csv)
    stats_dir = _get_stats_dir()

    # ---- 1. Session report (heatmap) ---- always runs
    try:
        from src.analytics.session_report import generate as _gen_report
        _gen_report(detail_csv, stats_dir)
    except Exception as exc:
        print(f"[WARN] Session report generation failed: {exc}")

    # ---- 2. Confusion matrix ---- only if ground-truth labels exist
    _try_confusion_matrix(detail_csv, stats_dir, labels_csv)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _try_confusion_matrix(detail_csv, output_dir, labels_csv=None):
    """Run evaluation if a matching ground-truth labels file is found."""
    try:
        from src.analytics.evaluate import (
            collect_pairs_segment,
            confusion,
            metrics_per_class,
            format_report,
            save_confusion_png,
            STATES,
        )
    except ImportError as exc:
        print(f"[WARN] Could not import evaluate module: {exc}")
        return

    labels_path = _find_labels(detail_csv, labels_csv)
    if labels_path is None:
        print("[INFO] No ground-truth labels file found -- skipping confusion matrix.")
        print("       To enable, place a segment labels CSV next to the details file")
        print("       (e.g. <video>_labels.csv) or as labels_segment.csv in the project root.")
        return

    try:
        pairs, unmatched = collect_pairs_segment(str(labels_path), str(detail_csv))
        if not pairs:
            print(f"[WARN] Labels found ({labels_path.name}) but no matching "
                  f"(true, pred) pairs. Check student names and timestamps.")
            return
        if unmatched:
            print(f"[INFO] {unmatched} detail rows did not match any label segment.")

        M = confusion(pairs, STATES)
        per_class, acc, macro, total = metrics_per_class(M, STATES)
        report_text = format_report(per_class, acc, macro, total, STATES)
        print(report_text)

        output_dir = Path(output_dir)
        report_path = output_dir / "evaluation_report.txt"
        report_path.write_text(report_text, encoding="utf-8")

        png_path = output_dir / "confusion_matrix.png"
        if save_confusion_png(M, STATES, png_path):
            print(f"[OK] Confusion matrix saved: {png_path}")
        print(f"[OK] Evaluation report saved: {report_path}")
    except Exception as exc:
        print(f"[WARN] Confusion matrix generation failed: {exc}")


def _find_labels(detail_csv, labels_csv=None):
    """Search for a ground-truth labels file that matches *detail_csv*.

    Lookup order:
      1. ``<video_stem>_labels.csv``  in the same directory as detail_csv
         (e.g. ``classroom_test_labels.csv`` for ``classroom_test_details.csv``)
      2. ``labels_segment.csv`` in the project root (AI-PROJECT/)
    """
    if labels_csv:
        candidate = Path(labels_csv)
        if candidate.exists():
            return candidate
        print(f"[WARN] Ground-truth labels file not found: {candidate}")
        return None

    detail_csv = Path(detail_csv)

    # 1. Sibling file:  <stem>_labels.csv
    stem = detail_csv.stem
    if stem.endswith("_details"):
        stem = stem[: -len("_details")]
    candidate = detail_csv.parent / f"{stem}_labels.csv"
    if candidate.exists():
        return candidate

    # 2. Project-root fallback
    try:
        from src.config import PROJECT_ROOT
        candidate = PROJECT_ROOT / "labels_segment.csv"
        if candidate.exists():
            return candidate
    except ImportError:
        pass

    return None
