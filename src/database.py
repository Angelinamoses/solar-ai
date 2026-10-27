"""
SolarGuard AI — SQLite Database Layer
======================================
Handles schema creation, inspection record persistence, and retrieval.
The database is initialised automatically on first launch.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent.parent / "data" / "solarguard.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create tables if they do not exist."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT    NOT NULL,
                filename         TEXT    NOT NULL,
                site_reference   TEXT,
                model_name       TEXT    NOT NULL,
                conf_threshold   REAL    NOT NULL,
                num_detections   INTEGER NOT NULL DEFAULT 0,
                overall_priority TEXT,
                annotated_image  TEXT,
                report_path      TEXT
            );

            CREATE TABLE IF NOT EXISTS detections (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id   INTEGER NOT NULL REFERENCES inspections(id) ON DELETE CASCADE,
                class_name      TEXT    NOT NULL,
                confidence      REAL    NOT NULL,
                bbox_x1         REAL,
                bbox_y1         REAL,
                bbox_x2         REAL,
                bbox_y2         REAL,
                priority        TEXT,
                suggested_action TEXT,
                review_interval_days INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_inspections_ts
                ON inspections(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_inspections_ref
                ON inspections(site_reference);
            CREATE INDEX IF NOT EXISTS idx_detections_insp
                ON detections(inspection_id);
            """
        )


def save_inspection(
    *,
    filename: str,
    site_reference: str | None,
    model_name: str,
    conf_threshold: float,
    detections: list[dict[str, Any]],
    risk_report: dict[str, Any],
    annotated_image_path: str | None = None,
    report_path: str | None = None,
) -> int:
    """
    Persist an inspection and its detections.  Returns the new inspection ID.
    """
    init_db()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    overall_priority = risk_report.get("overall_priority", "None")

    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO inspections
                (timestamp, filename, site_reference, model_name, conf_threshold,
                 num_detections, overall_priority, annotated_image, report_path)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                timestamp,
                filename,
                site_reference or "",
                model_name,
                conf_threshold,
                len(detections),
                overall_priority,
                annotated_image_path or "",
                report_path or "",
            ),
        )
        insp_id = cur.lastrowid

        class_results_by_name: dict[str, dict] = {
            r["class_name"]: r for r in risk_report.get("class_results", [])
        }

        for det in detections:
            cls = det.get("class_name", "Unknown")
            rule = class_results_by_name.get(cls, {})
            bbox = det.get("bbox", [None, None, None, None])
            conn.execute(
                """
                INSERT INTO detections
                    (inspection_id, class_name, confidence,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                     priority, suggested_action, review_interval_days)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    insp_id,
                    cls,
                    float(det.get("confidence", 0.0)),
                    bbox[0] if len(bbox) > 0 else None,
                    bbox[1] if len(bbox) > 1 else None,
                    bbox[2] if len(bbox) > 2 else None,
                    bbox[3] if len(bbox) > 3 else None,
                    rule.get("priority", ""),
                    rule.get("suggested_action", ""),
                    rule.get("review_interval_days"),
                ),
            )

    return insp_id


def get_inspections(
    limit: int = 200,
    site_reference: str | None = None,
    priority: str | None = None,
) -> list[dict]:
    """Return a list of inspection records, newest first."""
    init_db()
    conditions = []
    params: list[Any] = []

    if site_reference:
        conditions.append("site_reference = ?")
        params.append(site_reference)
    if priority and priority != "All":
        conditions.append("overall_priority = ?")
        params.append(priority)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM inspections {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        ).fetchall()
    return [dict(r) for r in rows]


def get_inspection_by_id(inspection_id: int) -> dict | None:
    """Return a single inspection record with its detections."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM inspections WHERE id = ?", (inspection_id,)
        ).fetchone()
        if row is None:
            return None
        insp = dict(row)
        dets = conn.execute(
            "SELECT * FROM detections WHERE inspection_id = ?", (inspection_id,)
        ).fetchall()
        insp["detections"] = [dict(d) for d in dets]
    return insp


def get_summary_stats() -> dict[str, Any]:
    """Return aggregate statistics for the dashboard overview."""
    init_db()
    with _connect() as conn:
        total_inspections = conn.execute(
            "SELECT COUNT(*) FROM inspections"
        ).fetchone()[0]
        total_detections = conn.execute(
            "SELECT SUM(num_detections) FROM inspections"
        ).fetchone()[0] or 0
        high_priority = conn.execute(
            "SELECT COUNT(*) FROM inspections WHERE overall_priority IN ('Critical','High')"
        ).fetchone()[0]
        class_counts_rows = conn.execute(
            "SELECT class_name, COUNT(*) as cnt FROM detections GROUP BY class_name ORDER BY cnt DESC"
        ).fetchall()
        recent = conn.execute(
            "SELECT * FROM inspections ORDER BY timestamp DESC LIMIT 5"
        ).fetchall()

    return {
        "total_inspections": total_inspections,
        "total_detections": total_detections,
        "high_priority_count": high_priority,
        "class_distribution": [dict(r) for r in class_counts_rows],
        "recent_inspections": [dict(r) for r in recent],
    }


def delete_inspection(inspection_id: int) -> bool:
    """Delete an inspection and its detections. Returns True if found."""
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM inspections WHERE id = ?", (inspection_id,)
        )
    return cur.rowcount > 0


def get_all_site_references() -> list[str]:
    """Return sorted list of distinct non-empty site references."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT site_reference FROM inspections "
            "WHERE site_reference IS NOT NULL AND site_reference != '' "
            "ORDER BY site_reference"
        ).fetchall()
    return [r[0] for r in rows]
