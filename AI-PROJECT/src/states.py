"""Canonical state names for the E-Proctoring System.

This is the **single source of truth** for the 7 behavioural states.
Import from here instead of using raw strings, to prevent typos and drift.

The string *values* are intentionally kept identical to what existing
CSVs and configs already contain, so backwards-compatibility is preserved.
"""

# ---------------------------------------------------------------------------
# Individual constants
# ---------------------------------------------------------------------------

OK          = "OK"           # focused / no issues
DISTRACTED  = "DISTRACTED"   # looking away or head turned
TALKING     = "TALKING"      # mouth oscillation detected
SLEEPING    = "SLEEPING"     # eyes closed too long
ABSENT      = "ABSENT"       # no face / body detected
PHONE_USAGE = "PHONE_USAGE"  # phone detected near student
BODY_ONLY   = "BODY_ONLY"    # body visible but face unclear

# ---------------------------------------------------------------------------
# Ordered list (used for confusion-matrix axes and heatmap legends)
# ---------------------------------------------------------------------------

ALL_STATES = [OK, BODY_ONLY, DISTRACTED, TALKING, PHONE_USAGE, SLEEPING, ABSENT]

# Same order expected by evaluate.py
EVAL_STATES = [OK, DISTRACTED, TALKING, SLEEPING, ABSENT, PHONE_USAGE, BODY_ONLY]

# ---------------------------------------------------------------------------
# Human-readable labels (Vietnamese, matching the existing session report)
# ---------------------------------------------------------------------------

LABEL_VI = {
    OK:          "Tap trung",
    BODY_ONLY:   "Khong ro mat",
    DISTRACTED:  "Mat tap trung",
    TALKING:     "Noi chuyen",
    PHONE_USAGE: "Dung dien thoai",
    SLEEPING:    "Ngu gat",
    ABSENT:      "Vang mat",
}

# Shared visual palette. Values are CSS hex colors; OpenCV callers can convert
# them to BGR tuples when drawing frames.
STATE_COLORS = {
    OK:          "#22c55e",
    BODY_ONLY:   "#94a3b8",
    DISTRACTED:  "#f59e0b",
    TALKING:     "#a855f7",
    PHONE_USAGE: "#ef4444",
    SLEEPING:    "#3b82f6",
    ABSENT:      "#1e293b",
}
