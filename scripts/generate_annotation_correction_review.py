"""Annotation Correction Visual Review Generator.

Produces per-frame comparison panels comparing:
  - Panel 1: Raw frame (no boxes)
  - Panel 2: Submitted human box drawn on raw frame (RED)
  - Panel 3: Proposed corrected box drawn on raw frame (GREEN)

ALL BOXES ARE CANDIDATES.  Nothing is written to the fixture or ground-truth
files.  The script preserves all originals and simply renders visual evidence
so the reviewer can approve or reject each proposed correction.

Outputs:
  outputs/annotation_correction_review/review_f<NNNN>.png  (per-frame 3-panel)
  outputs/annotation_correction_review/correction_review.html (full HTML report)
  outputs/annotation_correction_review/audit_trail.json  (machine-readable audit)

Usage:
  python scripts/generate_annotation_correction_review.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
FRAMES_DIR = Path("outputs/evaluation/benchmark_gate_v2/frames")
OUTPUT_DIR = Path("outputs/annotation_correction_review")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HUMAN_EXPORT = Path("outputs/evaluation/benchmark_gate_v2/ppe1_independent_annotations_completed.json")
V1_FIXTURE = Path("tests/fixtures/evaluation/ppe1_annotations.json")
V2_FIXTURE = Path("tests/fixtures/evaluation/ppe1_annotations_v2.json")

# ── Policy A proposed corrections (from validation report) ─────────────────────
# Format: frame_index -> [x, y, w, h]  (pixel, top-left origin)
# These are CANDIDATES only — they have NOT been approved by the reviewer.
PROPOSED_CORRECTIONS: dict[int, list[int]] = {
    0:   [560, 105, 300, 550],
    30:  [520,  90, 330, 555],
    60:  [475,  90, 365, 580],
    90:  [490,  88, 360, 585],
    120: [495,  88, 355, 585],
    150: [500,  85, 345, 585],
    180: [505, 118, 315, 550],
    210: [510, 150, 355, 520],
    240: [520, 178, 355, 500],
    270: [520, 192, 360, 490],
    300: [520, 195, 355, 485],
    330: [520, 202, 355, 485],
}

# Discrepancy rationale per frame (from validation report)
RATIONALE: dict[int, str] = {
    0:   "Hard hat apex truncated ~38px (y=143→105); left elbow clipped (x=588→560).",
    30:  "Hard hat apex truncated ~63px (y=153→90); right shoulder clipped (x=574→520); bottom clips hand (y2=653→645).",
    60:  "Raised left forearm reaching overhead COMPLETELY excluded (x=568→475); hard hat truncated ~60px.",
    90:  "Raised arm truncated (x=549→490); hard hat truncated ~81px (y=169→88); bottom clips hand on clipboard.",
    120: "Raised arm truncated (x=545→495); hard hat truncated ~66px (y=154→88).",
    150: "SEVERE: hard hat truncated ~110px (y=195→85); raised hand reaching overhead truncated (x=581→500).",
    180: "Worker bending: hard hat truncated ~109px (y=227→118); right forearm truncated (x=597→505).",
    210: "Hard hat truncated ~114px (y=264→150); leaning arms truncated (x=604→510).",
    240: "SEVERE: hard hat truncated ~127px (y=305→178); clipboard hand excluded (x=645→520).",
    270: "Hard hat truncated ~45px (y=237→192); extended forearm excluded (x=635→520).",
    300: "Hard hat truncated ~73px (y=268→195); forearm/elbow excluded (x=644→520).",
    330: "Hard hat truncated ~72px (y=274→202); writing arm excluded (x=642→520).",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def iou(a: list[int], b: list[int]) -> float:
    """Compute IoU for two [x, y, w, h] boxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def draw_box(img: np.ndarray, box: list[int], color: tuple[int, int, int], label: str, thickness: int = 3) -> np.ndarray:
    out = img.copy()
    x, y, w, h = box
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    # Semi-transparent label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(out, (x, y - th - 8), (x + tw + 6, y), color, -1)
    cv2.putText(out, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_3panel(raw: np.ndarray, submitted: list[int], proposed: list[int]) -> np.ndarray:
    """Create a 3-panel comparison image: raw | submitted (red) | proposed (green)."""
    panel_raw = raw.copy()
    panel_sub = draw_box(raw, submitted, (0, 0, 220), "SUBMITTED", thickness=3)
    panel_prop = draw_box(raw, proposed, (0, 180, 0), "PROPOSED", thickness=3)

    # Add reference grid (50px)
    h, w = raw.shape[:2]
    for panels in [panel_raw, panel_sub, panel_prop]:
        for gx in range(0, w, 50):
            cv2.line(panels, (gx, 0), (gx, h), (80, 80, 80), 1)
        for gy in range(0, h, 50):
            cv2.line(panels, (0, gy), (w, gy), (80, 80, 80), 1)

    separator = np.full((h, 4, 3), 200, dtype=np.uint8)
    return np.hstack([panel_raw, separator, panel_sub, separator, panel_prop])


def main() -> None:
    print("=" * 70)
    print("VIDEX Annotation Correction Visual Review Generator")
    print("STATUS: Generating CANDIDATE proposals only. NO fixtures modified.")
    print("=" * 70)

    # 1. Pre-run hash verification
    hashes_before = {
        "v1_fixture": sha256_file(V1_FIXTURE),
        "v2_fixture": sha256_file(V2_FIXTURE),
        "human_export": sha256_file(HUMAN_EXPORT),
    }
    print("\nPre-run hash check:")
    for k, v in hashes_before.items():
        print(f"  {k}: {v}")

    expected = {
        "v1_fixture":   "edd062d360c42850a55b7de94c5f8febd475683ef0fa53f2d9b5018d27af934b",
        "v2_fixture":   "8fcd58897e00b500e73a89e5ab7e91b2022fc3efbc3044687385e44ab526b4b7",
        "human_export": "60d75d0b403ca8f289790450dbc7f472968eaa11c3d2eaab2efbf0da4d4a02e1",
    }
    for k in expected:
        status = "OK" if hashes_before[k] == expected[k] else "MISMATCH"
        print(f"  {k}: {status}")
        if status == "MISMATCH":
            raise RuntimeError(f"Hash mismatch on {k}! Aborting to protect originals.")

    # 2. Load submitted human boxes
    with HUMAN_EXPORT.open() as f:
        export_data = json.load(f)

    submitted_boxes: dict[int, list[int]] = {}
    for det in export_data["videos"][0]["detections"]["items"]:
        fi = det["frame_index"]
        b = det["bbox"]
        submitted_boxes[fi] = [int(b["x"]), int(b["y"]), int(b["width"]), int(b["height"])]

    # 3. Generate per-frame panels
    frame_records: list[dict] = []
    frame_indices = sorted(PROPOSED_CORRECTIONS.keys())

    for fi in frame_indices:
        frame_path = FRAMES_DIR / f"frame_{fi:04d}.png"
        if not frame_path.exists():
            print(f"  [WARN] Raw frame not found: {frame_path}")
            continue

        raw = cv2.imread(str(frame_path))
        if raw is None:
            print(f"  [WARN] Could not read: {frame_path}")
            continue

        submitted = submitted_boxes.get(fi, [0, 0, 100, 100])
        proposed = PROPOSED_CORRECTIONS[fi]
        overlap = iou(submitted, proposed)

        panel = make_3panel(raw, submitted, proposed)
        out_path = OUTPUT_DIR / f"review_f{fi:04d}.png"
        cv2.imwrite(str(out_path), panel)

        record = {
            "frame_index": fi,
            "timestamp_pts": round(fi / 29.97003, 3),
            "submitted_box": submitted,
            "proposed_box": proposed,
            "submitted_extent": [submitted[0], submitted[1], submitted[0] + submitted[2], submitted[1] + submitted[3]],
            "proposed_extent": [proposed[0], proposed[1], proposed[0] + proposed[2], proposed[1] + proposed[3]],
            "iou_overlap": round(overlap, 3),
            "rationale": RATIONALE.get(fi, ""),
            "review_image": str(out_path),
            "status": "AWAITING_APPROVAL",
        }
        frame_records.append(record)
        print(f"  Frame {fi:3d}: submitted={submitted}  proposed={proposed}  IoU={overlap:.3f}")

    # 4. Audit trail JSON
    audit = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "Correction candidate review — awaiting human approval before any fixture is written",
        "policy": "Policy A: Visible Person Only (docs/evaluation/annotation_guidelines_ppe1_v1.md)",
        "sources": {
            "raw_frames": str(FRAMES_DIR),
            "human_export": str(HUMAN_EXPORT),
        },
        "fixture_hashes_before": hashes_before,
        "corrections_applied_to_fixture": False,
        "corrections_approved_by_reviewer": False,
        "frames": frame_records,
    }
    audit_path = OUTPUT_DIR / "audit_trail.json"
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    # 5. HTML summary
    html_rows = ""
    for rec in frame_records:
        fi = rec["frame_index"]
        sub = rec["submitted_box"]
        prop = rec["proposed_box"]
        iou_val = rec["iou_overlap"]
        rat = rec["rationale"]
        img_name = Path(rec["review_image"]).name
        html_rows += f"""
        <tr>
          <td><b>Frame {fi}</b><br><small>{rec['timestamp_pts']:.3f}s</small></td>
          <td><code>[{sub[0]}, {sub[1]}, {sub[2]}, {sub[3]}]</code></td>
          <td><code>[{prop[0]}, {prop[1]}, {prop[2]}, {prop[3]}]</code></td>
          <td>{iou_val:.3f}</td>
          <td style="font-size:12px">{rat}</td>
          <td><span style="background:#ff4444;color:#fff;padding:2px 6px;border-radius:4px">AWAITING APPROVAL</span></td>
        </tr>
        <tr>
          <td colspan="6">
            <img src="{img_name}" style="max-width:100%;border:1px solid #ccc;border-radius:4px"
                 alt="Frame {fi} correction review">
            <p style="font-size:11px;color:#666">
              Left: Raw Frame &nbsp;|&nbsp;
              <span style="color:#cc0000">■ Centre: Submitted (RED)</span> &nbsp;|&nbsp;
              <span style="color:#007700">■ Right: Proposed (GREEN)</span>
            </p>
          </td>
        </tr>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VIDEX — PPE-1 Annotation Correction Review</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1600px; margin: 0 auto; padding: 24px; background: #f8f8f8; }}
  h1 {{ color: #1a1a2e; }}
  .banner {{ background: #fffbe6; border-left: 5px solid #f0a500; padding: 12px 16px; margin: 16px 0; border-radius: 4px; }}
  .hash-box {{ background: #1e1e1e; color: #cefad0; font-family: monospace; font-size: 12px; padding: 12px; border-radius: 4px; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; border-radius: 6px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }}
  th {{ background: #1a1a2e; color: #fff; padding: 10px 12px; text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #eee; vertical-align: top; }}
  code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; font-size: 13px; }}
</style>
</head>
<body>
<h1>VIDEX — PPE-1 Annotation Correction Review</h1>
<div class="banner">
  <b>⚠ CANDIDATE REVIEW — NOT APPLIED</b><br>
  These proposed corrections have <b>NOT</b> been written to any fixture file.
  All originals (v1, v2, human export) are intact. Review the visual evidence below
  and approve or reject each frame before any corrected fixture is created.
</div>

<h2>Fixture Hash Verification (Pre-Run)</h2>
<div class="hash-box">
  v1  ppe1_annotations.json:                  edd062d360c42850a55b7de94c5f8febd475683ef0fa53f2d9b5018d27af934b  [OK]<br>
  v2  ppe1_annotations_v2.json:               8fcd58897e00b500e73a89e5ab7e91b2022fc3efbc3044687385e44ab526b4b7  [OK]<br>
  exp ppe1_independent_annotations_completed: 60d75d0b403ca8f289790450dbc7f472968eaa11c3d2eaab2efbf0da4d4a02e1  [OK]
</div>

<h2>Policy A Correction Proposals — Awaiting Your Approval</h2>
<p>
  Each row shows the submitted human box, the proposed Policy A–corrected box, and their IoU overlap.
  Below each row is a 3-panel image: <b>raw frame</b> | <b>submitted box (red)</b> | <b>proposed box (green)</b>.
</p>
<table>
  <thead>
    <tr>
      <th>Frame</th>
      <th>Submitted [x, y, w, h]</th>
      <th>Proposed [x, y, w, h]</th>
      <th>IoU</th>
      <th>Discrepancy Rationale</th>
      <th>Status</th>
    </tr>
  </thead>
  <tbody>
{html_rows}
  </tbody>
</table>

<h2>How to Proceed</h2>
<ol>
  <li>Review each frame's 3-panel image above.</li>
  <li>For each frame, confirm whether the green proposed box correctly encompasses:
    <ul>
      <li>The complete hard hat apex (yellow safety helmet top)</li>
      <li>All visibly extended limbs (raised arms, leaning forearms)</li>
      <li>Excluding separate tools, hanging lamp, countertop, and workpieces</li>
    </ul>
  </li>
  <li>If any proposed box needs further adjustment, note the required change.</li>
  <li>Once all 12 frames are approved, corrected fixture will be written to
      <code>tests/fixtures/evaluation/ppe1_annotations_v3_independent.json</code>.</li>
</ol>

<p style="font-size:12px;color:#888">
  Generated: {datetime.now(UTC).isoformat()} UTC<br>
  Audit trail: <code>audit_trail.json</code>
</p>
</body>
</html>
"""
    html_path = OUTPUT_DIR / "correction_review.html"
    with html_path.open("w", encoding="utf-8") as f:
        f.write(html)

    # 6. Post-run hash verification (confirm originals untouched)
    hashes_after = {
        "v1_fixture":   sha256_file(V1_FIXTURE),
        "v2_fixture":   sha256_file(V2_FIXTURE),
        "human_export": sha256_file(HUMAN_EXPORT),
    }
    for k in expected:
        if hashes_after[k] != expected[k]:
            raise RuntimeError(f"POST-RUN HASH MISMATCH on {k}! Original was modified!")

    print("\nPost-run hash check: ALL ORIGINALS INTACT [OK]")
    print(f"\nOutputs written to: {OUTPUT_DIR}/")
    print("  correction_review.html  — open in browser for visual review")
    print("  review_f<NNNN>.png      — per-frame 3-panel images")
    print("  audit_trail.json        — machine-readable audit record")
    print("\nSTATUS: Awaiting reviewer approval. No fixture files were modified.")


if __name__ == "__main__":
    main()
