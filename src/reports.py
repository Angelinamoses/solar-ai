"""
SolarGuard AI — Report Generation
===================================
Generates CSV and summary text reports for inspection results.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def detections_to_csv_bytes(
    detections: list[dict[str, Any]],
    filename: str,
    site_reference: str | None,
    timestamp: str | None = None,
) -> bytes:
    """
    Serialize detection results as UTF-8 CSV bytes (for Streamlit download).
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "filename",
            "site_reference",
            "timestamp",
            "class_name",
            "confidence",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "priority",
            "suggested_action",
            "review_interval_days",
        ]
    )
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for det in detections:
        bbox = det.get("bbox", [None, None, None, None])
        while len(bbox) < 4:
            bbox.append(None)
        writer.writerow(
            [
                filename,
                site_reference or "",
                ts,
                det.get("class_name", ""),
                f"{det.get('confidence', 0.0):.4f}",
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                det.get("priority", ""),
                det.get("suggested_action", ""),
                det.get("review_interval_days", ""),
            ]
        )
    return output.getvalue().encode("utf-8")


def inspections_to_csv_bytes(inspections: list[dict[str, Any]]) -> bytes:
    """Serialize a list of inspection records as CSV bytes."""
    if not inspections:
        return b"id,timestamp,filename,site_reference,model_name,conf_threshold,num_detections,overall_priority\n"

    output = io.StringIO()
    fields = [
        "id", "timestamp", "filename", "site_reference",
        "model_name", "conf_threshold", "num_detections", "overall_priority",
        "annotated_image", "report_path",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(inspections)
    return output.getvalue().encode("utf-8")


def save_report_to_disk(
    detections: list[dict[str, Any]],
    filename: str,
    site_reference: str | None,
    inspection_id: int | None = None,
) -> Path:
    """
    Save a detection CSV report to the reports/ directory.
    Returns the path of the saved file.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = Path(filename).stem[:40]
    fname = f"report_{ts}_{stem}.csv"
    if inspection_id is not None:
        fname = f"report_insp{inspection_id:05d}_{ts}_{stem}.csv"
    path = REPORTS_DIR / fname
    data = detections_to_csv_bytes(
        detections, filename, site_reference, ts
    )
    path.write_bytes(data)
    return path
