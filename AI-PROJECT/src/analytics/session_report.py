"""
Privacy-First Session Analytics for the E-Proctoring System
============================================================

Hau-buoi-hoc (post-session) analytics. Doc file *_details.csv ma
video_analyzer da xuat ra, KHONG dung anh khuon mat, chi dung du lieu hinh
hoc (EAR / MAR / Yaw / state / attention_score) -> tao ra:

  1) class_attention_heatmap.png  : Heatmap chu y ca lop theo thoi gian
                                     (truc X = thoi gian, truc Y = sinh vien)
  2) session_report.html          : Bao cao privacy-first, khong co anh mat,
                                     chi co bieu do + so lieu hinh hoc.

CACH CHAY:
    python -m src.analytics.session_report output/video_analysis/classroom_test_details.csv

Module nay HOAN TOAN DOC LAP voi pipeline real-time. No chi DOC file CSV,
khong sua gi trong main.py / video_analyzer.py.

Khong can thu vien ngoai (chi dung chuan + matplotlib neu co; neu khong co
matplotlib van tao duoc heatmap SVG thuan tuy bang HTML).
"""

import csv
import sys
import html
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.states import (
    OK, BODY_ONLY, DISTRACTED, TALKING, PHONE_USAGE, SLEEPING, ABSENT,
    ALL_STATES, LABEL_VI,
)


# ---------------------------------------------------------------------------
# Cau hinh mau & nhan trang thai  (de chinh sua tap trung mot cho)
# ---------------------------------------------------------------------------

STATE_COLORS = {
    OK:          "#22c55e",   # xanh la  - tap trung
    BODY_ONLY:   "#94a3b8",   # xam      - chi thay than, khong ro mat
    DISTRACTED:  "#f59e0b",   # cam      - mat tap trung
    TALKING:     "#a855f7",   # tim      - noi chuyen
    PHONE_USAGE: "#ef4444",   # do       - dung dien thoai
    SLEEPING:    "#3b82f6",   # xanh duong - ngu gat
    ABSENT:      "#1e293b",   # den      - vang mat
}
STATE_ORDER = ALL_STATES

STATE_LABEL_VI = LABEL_VI


# ---------------------------------------------------------------------------
# 1. Doc va gom du lieu tu file details.csv
# ---------------------------------------------------------------------------

def load_details(csv_path):
    """Doc CSV chi tiet. Tra ve list cac dict da ep kieu so."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "t": float(r.get("timestamp", 0) or 0),
                    "student": r.get("student", "?"),
                    "state": (r.get("state") or "OK").strip(),
                    "ear": float(r.get("ear", 0) or 0),
                    "mar": float(r.get("mar", 0) or 0),
                    "yaw": float(r.get("yaw", 0) or 0),
                    "pitch": float(r.get("pitch", 0) or 0),
                    "score": float(r.get("attention_score", 100) or 100),
                    "phone": str(r.get("phone_detected", "")).lower() == "true",
                    "patterns": (r.get("patterns") or "").strip(),
                })
            except ValueError:
                continue
    return rows


def build_timeline(rows, n_bins=60):
    """
    Chia thoi gian thanh n_bins o. Voi moi (sinh vien, o thoi gian) lay
    trang thai xuat hien nhieu nhat (mode) -> luoi de ve heatmap.
    Tra ve: students(list), bins(list bien thoi gian), grid[student][bin]=state,
    va score_grid[student][bin]=diem chu y trung binh.
    """
    if not rows:
        return [], [], {}, {}

    students = sorted({r["student"] for r in rows})
    t_min = min(r["t"] for r in rows)
    t_max = max(r["t"] for r in rows)
    span = max(t_max - t_min, 1e-6)
    width = span / n_bins

    # gom theo (student, bin)
    states_in = defaultdict(list)   # (s, b) -> [state,...]
    scores_in = defaultdict(list)   # (s, b) -> [score,...]
    for r in rows:
        b = min(int((r["t"] - t_min) / width), n_bins - 1)
        states_in[(r["student"], b)].append(r["state"])
        scores_in[(r["student"], b)].append(r["score"])

    grid = {s: {} for s in students}
    score_grid = {s: {} for s in students}
    for s in students:
        for b in range(n_bins):
            sts = states_in.get((s, b))
            if sts:
                # trang thai xuat hien nhieu nhat trong o nay
                grid[s][b] = max(set(sts), key=sts.count)
                sc = scores_in[(s, b)]
                score_grid[s][b] = sum(sc) / len(sc)
            else:
                grid[s][b] = None          # khong co du lieu
                score_grid[s][b] = None

    bins = [t_min + (i + 0.5) * width for i in range(n_bins)]
    return students, bins, grid, score_grid


# ---------------------------------------------------------------------------
# 2. Tinh thong ke tom tat cho moi sinh vien (privacy-safe: chi so, khong anh)
# ---------------------------------------------------------------------------

def per_student_stats(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["student"]].append(r)

    stats = {}
    for s, items in by.items():
        items.sort(key=lambda x: x["t"])
        dur = max(x["t"] for x in items) - min(x["t"] for x in items)
        # dem ti le tung trang thai
        cnt = defaultdict(int)
        for x in items:
            cnt[x["state"]] += 1
        total = len(items)
        dist = {k: cnt[k] / total for k in cnt}
        final_score = items[-1]["score"]
        avg_score = sum(x["score"] for x in items) / total
        # ti le tap trung = % thoi gian o trang thai OK
        focus_pct = dist.get("OK", 0.0) * 100
        flags = sorted({x["patterns"] for x in items if x["patterns"]})
        phone = any(x["phone"] for x in items)
        stats[s] = {
            "duration": dur,
            "final_score": final_score,
            "avg_score": avg_score,
            "focus_pct": focus_pct,
            "dist": dist,
            "flags": flags,
            "phone": phone,
        }
    return stats


# ---------------------------------------------------------------------------
# 3. Ve heatmap PNG bang matplotlib (neu co). Khong co thi bo qua, HTML tu ve.
# ---------------------------------------------------------------------------

def render_heatmap_png(students, bins, grid, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception:
        return False  # khong co matplotlib -> HTML se tu ve heatmap

    state_idx = {st: i for i, st in enumerate(STATE_ORDER)}
    import numpy as np
    n_b = len(bins)
    data = np.full((len(students), n_b), np.nan)
    for si, s in enumerate(students):
        for b in range(n_b):
            st = grid[s].get(b)
            if st in state_idx:
                data[si, b] = state_idx[st]

    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([STATE_COLORS[s] for s in STATE_ORDER])
    cmap.set_bad("#0b1220")
    norm = BoundaryNorm(range(len(STATE_ORDER) + 1), cmap.N)

    fig, ax = plt.subplots(figsize=(12, 0.7 * len(students) + 2), dpi=130)
    fig.patch.set_facecolor("#0b1220")
    ax.set_facecolor("#0b1220")
    ax.imshow(data, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_yticks(range(len(students)))
    ax.set_yticklabels(students, color="#e2e8f0", fontsize=11)
    xticks = list(range(0, n_b, max(1, n_b // 8)))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{bins[i]:.0f}s" for i in xticks], color="#94a3b8")
    ax.set_xlabel("Thoi gian buoi hoc", color="#e2e8f0", fontsize=12)
    ax.set_title("HEATMAP CHU Y CA LOP  (privacy-first, khong dung anh mat)",
                 color="#38bdf8", fontsize=14, pad=14, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color("#1e293b")

    legend = [Patch(facecolor=STATE_COLORS[s], label=STATE_LABEL_VI[s])
              for s in STATE_ORDER]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, facecolor="#0b1220", edgecolor="#1e293b",
              labelcolor="#e2e8f0", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, facecolor="#0b1220", bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# 4. Sinh bao cao HTML privacy-first (tu ve heatmap bang div neu thieu PNG)
# ---------------------------------------------------------------------------

def render_html(students, bins, grid, score_grid, stats,
                source_name, png_ok, out_path):

    def cell(state):
        if state is None:
            return "#0b1220"
        return STATE_COLORS.get(state, "#334155")

    # luoi heatmap thuan HTML (luon co, du co PNG hay khong)
    rows_html = []
    for s in students:
        cells = "".join(
            f'<div class="cell" style="background:{cell(grid[s].get(b))}" '
            f'title="{html.escape(s)} | {bins[b]:.1f}s | '
            f'{STATE_LABEL_VI.get(grid[s].get(b) or "", "khong co du lieu")}"></div>'
            for b in range(len(bins))
        )
        label_score = stats.get(s, {}).get("final_score", 0)
        rows_html.append(
            f'<div class="hm-row"><div class="hm-label">{html.escape(s)}'
            f'<span class="hm-score">{label_score:.0f}</span></div>'
            f'<div class="hm-track">{cells}</div></div>'
        )
    heatmap_html = "\n".join(rows_html)

    legend_html = "".join(
        f'<span class="lg"><i style="background:{STATE_COLORS[s]}"></i>'
        f'{STATE_LABEL_VI[s]}</span>'
        for s in STATE_ORDER
    )

    # the thong ke tung sinh vien
    cards = []
    for s in students:
        st = stats[s]
        score = st["final_score"]
        ring = ("#22c55e" if score >= 70 else
                "#f59e0b" if score >= 40 else "#ef4444")
        bars = "".join(
            f'<div class="bar"><span>{STATE_LABEL_VI.get(k, k)}</span>'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{v*100:.0f}%;background:{STATE_COLORS.get(k,"#64748b")}">'
            f'</div></div><b>{v*100:.0f}%</b></div>'
            for k, v in sorted(st["dist"].items(), key=lambda x: -x[1])
        )
        flags = ""
        if st["flags"]:
            flags = ('<div class="flags">' +
                     "".join(f'<span class="flag">{html.escape(f)}</span>'
                             for f in st["flags"]) + "</div>")
        if st["phone"]:
            flags += '<div class="flags"><span class="flag phone">Phat hien dien thoai</span></div>'
        cards.append(f"""
        <div class="card">
          <div class="card-head">
            <div class="ring" style="--c:{ring};--p:{score:.0f}">
              <span>{score:.0f}</span>
            </div>
            <div>
              <h3>{html.escape(s)}</h3>
              <p>{st['duration']:.1f}s &middot; chu y TB {st['avg_score']:.0f}
                 &middot; tap trung {st['focus_pct']:.0f}%</p>
            </div>
          </div>
          <div class="bars">{bars}</div>
          {flags}
        </div>""")
    cards_html = "\n".join(cards)

    png_block = ""
    if png_ok:
        png_block = (
            '<div class="png-wrap"><img src="class_attention_heatmap.png" '
            'alt="Heatmap chu y ca lop"></div>'
        )

    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    n_students = len(students)
    span = bins[-1] if bins else 0

    return _HTML_TEMPLATE.format(
        source=html.escape(source_name),
        now=now,
        n_students=n_students,
        span=f"{span:.0f}",
        legend=legend_html,
        heatmap=heatmap_html,
        png_block=png_block,
        cards=cards_html,
        n_cols=len(bins),
    ).encode("utf-8").decode("utf-8") and _write(out_path, _HTML_TEMPLATE.format(
        source=html.escape(source_name),
        now=now,
        n_students=n_students,
        span=f"{span:.0f}",
        legend=legend_html,
        heatmap=heatmap_html,
        png_block=png_block,
        cards=cards_html,
        n_cols=len(bins),
    ))


def _write(path, text):
    Path(path).write_text(text, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# HTML template (dark, editorial, privacy-first)
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bao cao phien giam sat - Privacy First</title>
<style>
  :root {{ --bg:#0b1220; --panel:#0f1a2e; --line:#1e293b; --ink:#e2e8f0;
           --mut:#94a3b8; --accent:#38bdf8; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--ink);
          font-family:"Segoe UI",system-ui,sans-serif; line-height:1.55;
          padding:32px 18px 64px; }}
  .wrap {{ max-width:1040px; margin:0 auto; }}
  .badge {{ display:inline-flex; align-items:center; gap:8px;
            background:rgba(56,189,248,.12); color:var(--accent);
            border:1px solid rgba(56,189,248,.35); padding:6px 12px;
            border-radius:999px; font-size:12px; font-weight:600;
            letter-spacing:.04em; text-transform:uppercase; }}
  h1 {{ font-size:30px; margin:14px 0 4px; letter-spacing:-.02em; }}
  .sub {{ color:var(--mut); font-size:14px; }}
  .meta {{ display:flex; gap:24px; flex-wrap:wrap; margin:22px 0 8px; }}
  .meta div {{ background:var(--panel); border:1px solid var(--line);
               border-radius:12px; padding:12px 16px; min-width:120px; }}
  .meta b {{ display:block; font-size:24px; color:#fff; }}
  .meta span {{ font-size:12px; color:var(--mut); }}
  section {{ margin-top:38px; }}
  h2 {{ font-size:18px; margin-bottom:14px; display:flex; align-items:center;
        gap:10px; }}
  h2::before {{ content:""; width:4px; height:18px; background:var(--accent);
                border-radius:2px; }}
  .legend {{ display:flex; flex-wrap:wrap; gap:14px; margin:6px 0 16px;
             font-size:12px; color:var(--mut); }}
  .lg {{ display:inline-flex; align-items:center; gap:6px; }}
  .lg i {{ width:13px; height:13px; border-radius:3px; display:inline-block; }}
  .hm {{ background:var(--panel); border:1px solid var(--line);
         border-radius:14px; padding:16px; overflow-x:auto; }}
  .hm-row {{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }}
  .hm-label {{ width:120px; flex:none; font-size:13px; display:flex;
               justify-content:space-between; padding-right:8px; }}
  .hm-score {{ color:var(--accent); font-weight:700; }}
  .hm-track {{ display:grid; gap:2px; flex:1;
               grid-template-columns:repeat({n_cols},1fr); }}
  .cell {{ height:22px; border-radius:2px; transition:transform .1s; }}
  .cell:hover {{ transform:scaleY(1.25); outline:1px solid #fff; }}
  .png-wrap {{ margin-top:16px; }}
  .png-wrap img {{ width:100%; border-radius:14px; border:1px solid var(--line); }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
           gap:16px; }}
  .card {{ background:var(--panel); border:1px solid var(--line);
           border-radius:16px; padding:18px; }}
  .card-head {{ display:flex; align-items:center; gap:14px; margin-bottom:14px; }}
  .card h3 {{ font-size:16px; }}
  .card p {{ font-size:12px; color:var(--mut); }}
  .ring {{ --p:0; width:62px; height:62px; flex:none; border-radius:50%;
           display:grid; place-items:center; font-weight:800; font-size:18px;
           color:#fff;
           background:conic-gradient(var(--c) calc(var(--p)*1%),
                      #1e293b 0); }}
  .ring span {{ width:46px; height:46px; border-radius:50%; background:var(--panel);
                display:grid; place-items:center; }}
  .bar {{ display:grid; grid-template-columns:96px 1fr 38px; align-items:center;
          gap:8px; font-size:11px; color:var(--mut); margin-bottom:5px; }}
  .bar-track {{ background:#1e293b; border-radius:99px; height:8px; overflow:hidden; }}
  .bar-fill {{ height:100%; border-radius:99px; }}
  .bar b {{ color:var(--ink); text-align:right; }}
  .flags {{ margin-top:10px; display:flex; flex-wrap:wrap; gap:6px; }}
  .flag {{ background:rgba(245,158,11,.14); color:#fbbf24;
           border:1px solid rgba(245,158,11,.3); border-radius:8px;
           padding:3px 9px; font-size:11px; }}
  .flag.phone {{ background:rgba(239,68,68,.14); color:#f87171;
                 border-color:rgba(239,68,68,.3); }}
  .note {{ margin-top:40px; background:rgba(34,197,94,.07);
           border:1px solid rgba(34,197,94,.25); border-radius:14px;
           padding:18px 20px; font-size:13px; color:#bbf7d0; }}
  .note b {{ color:#86efac; }}
  footer {{ margin-top:36px; text-align:center; color:#475569; font-size:12px; }}
</style>
</head>
<body>
<div class="wrap">
  <span class="badge">&#128274; Privacy-First &middot; Khong luu anh khuon mat</span>
  <h1>Bao cao phien giam sat lop hoc</h1>
  <p class="sub">Nguon du lieu: {source} &middot; Tao luc {now}</p>

  <div class="meta">
    <div><b>{n_students}</b><span>Sinh vien duoc theo doi</span></div>
    <div><b>{span}s</b><span>Thoi luong phan tich</span></div>
    <div><b>0</b><span>Anh khuon mat luu tru</span></div>
  </div>

  <section>
    <h2>Heatmap chu y ca lop theo thoi gian</h2>
    <div class="legend">{legend}</div>
    <div class="hm">
      {heatmap}
    </div>
    {png_block}
  </section>

  <section>
    <h2>Phan tich tung sinh vien</h2>
    <div class="grid">
      {cards}
    </div>
  </section>

  <div class="note">
    <b>Nguyen tac Privacy-First:</b> Bao cao nay duoc tao hoan toan tu du lieu
    hinh hoc (goc dau Yaw/Pitch, ti le mat EAR, ti le mieng MAR, trang thai
    hanh vi). KHONG co khung hinh goc hay anh khuon mat nao duoc luu lam bang
    chung. Giao vien van nam duoc ai mat tap trung vao luc nao, nhung danh tinh
    truc quan cua sinh vien khong bi luu tru &mdash; giam thieu rui ro lo lot
    du lieu sinh trac hoc.
  </div>

  <footer>E-Proctoring System &middot; Nhom 8 &middot; Module phan tich hau-buoi-hoc</footer>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate(details_csv, out_dir=None):
    details_csv = Path(details_csv)
    if out_dir is None:
        out_dir = details_csv.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_details(details_csv)
    if not rows:
        print("[LOI] Khong doc duoc du lieu tu", details_csv)
        return

    students, bins, grid, score_grid = build_timeline(rows, n_bins=60)
    stats = per_student_stats(rows)

    png_path = out_dir / "class_attention_heatmap.png"
    png_ok = render_heatmap_png(students, bins, grid, png_path)

    html_path = out_dir / "session_report.html"
    render_html(students, bins, grid, score_grid, stats,
                details_csv.name, png_ok, html_path)

    print("=" * 56)
    print(" BAO CAO PRIVACY-FIRST DA TAO XONG")
    print("=" * 56)
    print(f" Sinh vien    : {len(students)}")
    print(f" Heatmap PNG  : {'co' if png_ok else 'bo qua (thieu matplotlib)'}")
    print(f" Bao cao HTML : {html_path}")
    if png_ok:
        print(f" Anh heatmap  : {png_path}")
    print("=" * 56)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cach dung: python -m src.analytics.session_report <details.csv>")
        sys.exit(1)
    generate(sys.argv[1])