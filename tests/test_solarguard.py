"""
Tests for SolarGuard AI.

Covers:
- Database operations (init, save, retrieve, stats)
- Risk rule evaluation
- Annotation validation logic

Does NOT require a trained model or GPU.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# Make sure the project root is on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ===========================================================================
# Database tests
# ===========================================================================

class TestDatabase:
    """Isolated DB tests using a temp directory to avoid touching production DB."""

    def setup_method(self):
        """Redirect DB_PATH to a temp file before each test."""
        import src.database as db_module
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_path = db_module.DB_PATH
        db_module.DB_PATH = Path(self._tmp.name) / "test.db"

    def teardown_method(self):
        """Restore DB_PATH and clean up temp dir.

        On Windows, SQLite WAL mode keeps the .db file open even after the
        connection context manager exits (WAL/SHM files may remain).  Use
        ignore_cleanup_errors so teardown never fails due to this OS-level
        lock; the temp directory will be reclaimed by the OS on process exit.
        """
        import gc
        import src.database as db_module
        db_module.DB_PATH = self._orig_path
        gc.collect()  # encourage Python to close any lingering sqlite3 Connection objects
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def _get_db(self):
        import src.database as db_module
        return db_module

    def test_init_creates_tables(self):
        db = self._get_db()
        db.init_db()
        with sqlite3.connect(str(db.DB_PATH)) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "inspections" in tables
        assert "detections" in tables

    def test_init_is_idempotent(self):
        db = self._get_db()
        db.init_db()
        db.init_db()  # second call must not raise

    def test_save_and_retrieve_inspection(self):
        db = self._get_db()
        db.init_db()
        detections = [
            {"class_name": "SingleHotSpot", "confidence": 0.85, "bbox": [10, 20, 50, 60]},
        ]
        risk_report = {
            "overall_priority": "High",
            "class_results": [
                {
                    "class_name": "SingleHotSpot",
                    "priority": "High",
                    "suggested_action": "Monitor",
                    "review_interval_days": 14,
                }
            ],
        }
        insp_id = db.save_inspection(
            filename="test_image.jpg",
            site_reference="Site-A",
            model_name="best.pt",
            conf_threshold=0.25,
            detections=detections,
            risk_report=risk_report,
        )
        assert isinstance(insp_id, int)
        assert insp_id >= 1

        record = db.get_inspection_by_id(insp_id)
        assert record is not None
        assert record["filename"] == "test_image.jpg"
        assert record["site_reference"] == "Site-A"
        assert record["num_detections"] == 1
        assert record["overall_priority"] == "High"
        assert len(record["detections"]) == 1
        assert record["detections"][0]["class_name"] == "SingleHotSpot"
        assert abs(record["detections"][0]["confidence"] - 0.85) < 1e-4

    def test_get_inspection_missing_id_returns_none(self):
        db = self._get_db()
        db.init_db()
        assert db.get_inspection_by_id(999999) is None

    def test_summary_stats_empty_db(self):
        db = self._get_db()
        db.init_db()
        stats = db.get_summary_stats()
        assert stats["total_inspections"] == 0
        assert stats["total_detections"] == 0
        assert stats["high_priority_count"] == 0
        assert stats["class_distribution"] == []
        assert stats["recent_inspections"] == []

    def test_summary_stats_with_data(self):
        db = self._get_db()
        db.init_db()
        for i in range(3):
            db.save_inspection(
                filename=f"img_{i}.jpg",
                site_reference="SiteX",
                model_name="best.pt",
                conf_threshold=0.25,
                detections=[{"class_name": "MultiHotSpot", "confidence": 0.9, "bbox": [0, 0, 100, 100]}],
                risk_report={
                    "overall_priority": "Critical",
                    "class_results": [{"class_name": "MultiHotSpot", "priority": "Critical",
                                       "suggested_action": "Inspect", "review_interval_days": 3}],
                },
            )
        stats = db.get_summary_stats()
        assert stats["total_inspections"] == 3
        assert stats["total_detections"] == 3
        assert stats["high_priority_count"] == 3

    def test_get_all_site_references(self):
        db = self._get_db()
        db.init_db()
        for ref in ["Beta", "Alpha", "Alpha"]:
            db.save_inspection(
                filename="x.jpg", site_reference=ref, model_name="best.pt",
                conf_threshold=0.25, detections=[], risk_report={"overall_priority": "None", "class_results": []}
            )
        refs = db.get_all_site_references()
        assert refs == ["Alpha", "Beta"]

    def test_filter_by_priority(self):
        db = self._get_db()
        db.init_db()
        db.save_inspection(
            filename="a.jpg", site_reference=None, model_name="best.pt",
            conf_threshold=0.25, detections=[], risk_report={"overall_priority": "High", "class_results": []}
        )
        db.save_inspection(
            filename="b.jpg", site_reference=None, model_name="best.pt",
            conf_threshold=0.25, detections=[], risk_report={"overall_priority": "Low", "class_results": []}
        )
        high = db.get_inspections(priority="High")
        assert len(high) == 1
        assert high[0]["filename"] == "a.jpg"

    def test_delete_inspection(self):
        db = self._get_db()
        db.init_db()
        iid = db.save_inspection(
            filename="del.jpg", site_reference=None, model_name="best.pt",
            conf_threshold=0.25, detections=[], risk_report={"overall_priority": "None", "class_results": []}
        )
        assert db.delete_inspection(iid) is True
        assert db.get_inspection_by_id(iid) is None
        assert db.delete_inspection(999999) is False


# ===========================================================================
# Risk tests
# ===========================================================================

class TestRisk:

    def test_no_detections_returns_no_detection(self):
        from src.risk import assess_inspection
        result = assess_inspection([])
        assert result["no_detection"] is True
        assert result["overall_priority"] == "None"
        assert result["class_results"] == []

    def test_single_critical_class(self):
        from src.risk import assess_inspection
        result = assess_inspection([
            {"class_name": "MultiHotSpot", "confidence": 0.9}
        ])
        assert result["no_detection"] is False
        assert result["overall_priority"] == "Critical"
        assert len(result["class_results"]) == 1
        assert result["class_results"][0]["class_name"] == "MultiHotSpot"

    def test_mixed_priorities_returns_highest(self):
        from src.risk import assess_inspection
        result = assess_inspection([
            {"class_name": "SingleDiode", "confidence": 0.6},   # Medium
            {"class_name": "StringOpenCircuit", "confidence": 0.8},  # Critical
        ])
        assert result["overall_priority"] == "Critical"

    def test_priority_order_critical_first(self):
        from src.risk import priority_rank
        assert priority_rank("Critical") < priority_rank("High")
        assert priority_rank("High") < priority_rank("Medium")
        assert priority_rank("Medium") < priority_rank("Low")

    def test_unknown_class_defaults_to_medium(self):
        from src.risk import assess_inspection
        result = assess_inspection([
            {"class_name": "UnknownFaultXYZ", "confidence": 0.5}
        ])
        assert result["overall_priority"] == "Medium"

    def test_priority_color_returns_string(self):
        from src.risk import priority_color
        for p in ["Critical", "High", "Medium", "Low", "None", "Unknown"]:
            color = priority_color(p)
            assert isinstance(color, str)
            assert color.startswith("#")

    def test_assess_inspection_review_interval(self):
        from src.risk import assess_inspection
        result = assess_inspection([
            {"class_name": "StringReversedPolarity", "confidence": 0.7}
        ])
        assert result["review_interval_days"] == 1  # as per config

    def test_deduplicated_classes(self):
        """Multiple detections of the same class should produce one class_result."""
        from src.risk import assess_inspection
        result = assess_inspection([
            {"class_name": "SingleHotSpot", "confidence": 0.9},
            {"class_name": "SingleHotSpot", "confidence": 0.7},
        ])
        assert sum(1 for cr in result["class_results"] if cr["class_name"] == "SingleHotSpot") == 1

    def test_disclaimer_present(self):
        from src.risk import assess_inspection
        result = assess_inspection([{"class_name": "SingleByPassed", "confidence": 0.5}])
        assert "disclaimer" in result
        assert len(result["disclaimer"]) > 10


# ===========================================================================
# Annotation validation tests
# ===========================================================================

class TestAnnotationValidation:
    """
    Tests the YOLO-line validation logic from scripts/prepare_dataset.py.
    Imported directly to avoid needing disk access.
    """

    def setup_method(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from prepare_dataset import _validate_yolo_line
        self.validate = _validate_yolo_line

    def test_valid_line(self):
        ok, reason = self.validate("2 0.5 0.5 0.2 0.3")
        assert ok is True
        assert reason == ""

    def test_wrong_field_count(self):
        ok, reason = self.validate("2 0.5 0.5 0.2")
        assert ok is False
        assert "4" in reason or "fields" in reason.lower()

    def test_non_integer_class(self):
        ok, reason = self.validate("a 0.5 0.5 0.2 0.3")
        assert ok is False

    def test_class_out_of_range_high(self):
        ok, reason = self.validate("8 0.5 0.5 0.2 0.3")
        assert ok is False

    def test_class_negative(self):
        ok, reason = self.validate("-1 0.5 0.5 0.2 0.3")
        assert ok is False

    def test_non_numeric_coordinate(self):
        ok, reason = self.validate("0 0.5 abc 0.2 0.3")
        assert ok is False

    def test_cx_out_of_range(self):
        ok, reason = self.validate("0 1.5 0.5 0.2 0.3")
        assert ok is False
        assert "cx" in reason.lower() or "cy" in reason.lower() or "0.1" in reason

    def test_zero_cx(self):
        ok, reason = self.validate("0 0.0 0.5 0.2 0.3")
        assert ok is False

    def test_width_out_of_range(self):
        ok, reason = self.validate("0 0.5 0.5 1.5 0.3")
        assert ok is False

    def test_all_eight_classes_valid(self):
        for cid in range(8):
            ok, reason = self.validate(f"{cid} 0.5 0.5 0.1 0.1")
            assert ok is True, f"Class {cid} should be valid but got: {reason}"

    def test_extra_whitespace_handled(self):
        ok, reason = self.validate("  3  0.4  0.6  0.15  0.2  ")
        assert ok is True


# ===========================================================================
# Report tests
# ===========================================================================

class TestReports:

    def test_detections_to_csv_bytes_empty(self):
        from src.reports import detections_to_csv_bytes
        data = detections_to_csv_bytes([], "test.jpg", None)
        assert isinstance(data, bytes)
        lines = data.decode("utf-8").splitlines()
        assert len(lines) == 1  # header only
        assert "class_name" in lines[0]

    def test_detections_to_csv_bytes_with_data(self):
        from src.reports import detections_to_csv_bytes
        dets = [
            {"class_name": "MultiHotSpot", "confidence": 0.88,
             "bbox": [10.0, 20.0, 50.0, 60.0],
             "priority": "Critical", "suggested_action": "Inspect", "review_interval_days": 3}
        ]
        data = detections_to_csv_bytes(dets, "img.jpg", "SiteA")
        lines = data.decode("utf-8").splitlines()
        assert len(lines) == 2  # header + 1 row
        assert "MultiHotSpot" in lines[1]
        assert "SiteA" in lines[1]

    def test_inspections_to_csv_bytes_empty(self):
        from src.reports import inspections_to_csv_bytes
        data = inspections_to_csv_bytes([])
        assert b"id" in data

    def test_inspections_to_csv_bytes_with_data(self):
        from src.reports import inspections_to_csv_bytes
        insp = [{"id": 1, "timestamp": "2024-01-01T00:00:00Z", "filename": "a.jpg",
                  "site_reference": "Site1", "model_name": "best.pt", "conf_threshold": 0.25,
                  "num_detections": 2, "overall_priority": "High"}]
        data = inspections_to_csv_bytes(insp)
        text = data.decode("utf-8")
        assert "Site1" in text
        assert "High" in text
