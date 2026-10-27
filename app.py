"""
SolarGuard AI — Streamlit Dashboard
======================================
Main entry point.  Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Internal modules
from src import database, detector, reports, risk

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SolarGuard AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — solar energy theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%);
        border-right: 1px solid #2d3748;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a2035 0%, #1e2d40 100%);
        border: 1px solid #2d4a6e;
        border-radius: 12px;
        padding: 12px 16px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #0f1117;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(245,158,11,0.4);
    }

    /* Priority badges */
    .badge-critical { background:#FF4B4B; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.8em; font-weight:600; }
    .badge-high     { background:#FF8C00; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.8em; font-weight:600; }
    .badge-medium   { background:#FFC107; color:#0f1117; padding:3px 10px; border-radius:20px; font-size:0.8em; font-weight:600; }
    .badge-low      { background:#4CAF50; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.8em; font-weight:600; }
    .badge-none     { background:#9E9E9E; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.8em; font-weight:600; }

    /* Section headers */
    .section-header {
        font-size: 1.1em; font-weight: 600; color: #f59e0b;
        border-bottom: 1px solid #2d3748; padding-bottom: 6px; margin-bottom: 12px;
    }

    /* Info boxes */
    .info-box {
        background: #1a2035; border-left: 3px solid #3b82f6;
        padding: 10px 14px; border-radius: 6px; margin: 8px 0;
        font-size: 0.9em; color: #94a3b8;
    }
    .warn-box {
        background: #2a1f0f; border-left: 3px solid #f59e0b;
        padding: 10px 14px; border-radius: 6px; margin: 8px 0;
        font-size: 0.9em; color: #d97706;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <div style="font-size:2.5em;">☀️</div>
            <div style="font-size:1.2em; font-weight:700; color:#f59e0b;">SolarGuard AI</div>
            <div style="font-size:0.75em; color:#64748b; margin-top:4px;">Fault Detection Dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    page = st.radio(
        "Navigation",
        options=[
            "📊 Dashboard Overview",
            "🔍 Inspect Image",
            "📋 Inspection History",
            "ℹ️ Model & Dataset Info",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    model_info = detector.get_model_info()
    if model_info["available"]:
        st.markdown(
            "<div style='color:#4CAF50; font-size:0.8em;'>✅ Custom model loaded</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Weights: `{Path(model_info['path']).name}`")
    else:
        st.markdown(
            "<div style='color:#FF4B4B; font-size:0.8em;'>⚠️ No custom model found</div>",
            unsafe_allow_html=True,
        )
        st.caption("Run `python train_model.py` to train")


# ---------------------------------------------------------------------------
# Initialise DB on every run (idempotent)
# ---------------------------------------------------------------------------
database.init_db()

# ---------------------------------------------------------------------------
# Helper: priority badge HTML
# ---------------------------------------------------------------------------
def _badge(priority: str) -> str:
    cls = {
        "Critical": "badge-critical",
        "High": "badge-high",
        "Medium": "badge-medium",
        "Low": "badge-low",
    }.get(priority, "badge-none")
    return f'<span class="{cls}">{priority}</span>'


# ===========================================================================
# Page A: Dashboard Overview
# ===========================================================================
if page == "📊 Dashboard Overview":
    st.title("📊 Dashboard Overview")
    st.caption("Statistics are calculated from real inspection records saved to the local database.")

    stats = database.get_summary_stats()

    # ---- KPI row -----------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Inspections", stats["total_inspections"])
    col2.metric("Total Detections", stats["total_detections"])
    col3.metric("High-Priority Reviews", stats["high_priority_count"])
    recent_count = len(stats["recent_inspections"])
    col4.metric("Recent (last 5)", recent_count)

    st.divider()

    if stats["total_inspections"] == 0:
        st.markdown(
            """
            <div class="info-box">
            <b>No inspections yet.</b><br>
            Go to <b>🔍 Inspect Image</b>, upload a solar panel image, and run the detector.
            Results will appear here automatically after saving.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown('<div class="section-header">Fault Class Distribution</div>', unsafe_allow_html=True)
            if stats["class_distribution"]:
                df_cls = pd.DataFrame(stats["class_distribution"]).rename(
                    columns={"class_name": "Fault Class", "cnt": "Detections"}
                )
                st.bar_chart(df_cls.set_index("Fault Class"), height=280)
            else:
                st.info("No detections recorded yet.")

        with col_b:
            st.markdown('<div class="section-header">Recent Inspections</div>', unsafe_allow_html=True)
            if stats["recent_inspections"]:
                for rec in stats["recent_inspections"]:
                    priority = rec.get("overall_priority", "None")
                    badge_html = _badge(priority)
                    ref = rec.get("site_reference") or "—"
                    st.markdown(
                        f"**#{rec['id']}** {Path(rec['filename']).name} &nbsp;"
                        f"{badge_html} &nbsp; `{ref}` &nbsp; "
                        f"<small style='color:#64748b'>{rec['timestamp']}</small>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No recent inspections.")

        st.divider()

        # Priority breakdown
        st.markdown('<div class="section-header">Priority Breakdown</div>', unsafe_allow_html=True)
        all_insp = database.get_inspections(limit=500)
        if all_insp:
            df_insp = pd.DataFrame(all_insp)
            priority_counts = df_insp["overall_priority"].value_counts().reset_index()
            priority_counts.columns = ["Priority", "Count"]
            st.dataframe(priority_counts, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="warn-box">
        <b>Disclaimer:</b> All priorities shown are provisional demonstration rules.
        They must be verified by qualified solar engineers before operational use.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# Page B: Inspect Image
# ===========================================================================
elif page == "🔍 Inspect Image":
    st.title("🔍 Inspect an Image")

    if not detector.is_model_available():
        st.error(
            "**No custom model weights found.**\n\n"
            "Train the model first:\n"
            "```bash\n"
            "# Quick prototype (default — ~2 epochs):\n"
            "python train_model.py\n\n"
            "# Full training:\n"
            "python train_model.py --mode full --epochs 30\n"
            "```\n\n"
            "After training, refresh this page."
        )
        st.stop()

    # ---- Upload & settings -------------------------------------------------
    col_upload, col_settings = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Upload a solar panel image",
            type=["jpg", "jpeg", "png"],
            help="Upload a JPG or PNG image. Grayscale images are supported.",
        )

    with col_settings:
        site_ref = st.text_input(
            "Site / Inspection Reference (optional)",
            placeholder="e.g. Site-A Block-3 Row-7",
            help="Optional user-entered label for grouping inspections. Not a verified panel ID.",
        )
        conf_threshold = st.slider(
            "Confidence Threshold", 0.10, 0.95, 0.25, 0.05,
            help="Minimum detection confidence. Lower = more detections (possibly more false positives).",
        )

    if uploaded is None:
        st.markdown(
            """
            <div class="info-box">
            Upload a solar panel image above to begin fault detection.
            The custom YOLOv8n model will detect anomalies and assign a review priority.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    # ---- Run inference -----------------------------------------------------
    try:
        img_bytes = uploaded.read()
        image = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    run_btn = st.button("🚀 Run Detection", use_container_width=False)
    if not run_btn:
        # Show original image while waiting
        st.image(image, caption="Uploaded image (awaiting detection)", use_container_width=True)
        st.stop()

    with st.spinner("Running custom YOLOv8n detector..."):
        result = detector.run_inference(image, conf_threshold=conf_threshold)

    if result["error"]:
        st.error(f"**Detection failed:**\n\n{result['error']}")
        st.stop()

    detections = result["detections"]
    annotated_img = result["annotated_image"]
    inference_ms = result["inference_time_ms"]

    # ---- Build risk report -------------------------------------------------
    risk_report = risk.assess_inspection(detections)

    # ---- Annotated image side-by-side view ---------------------------------
    st.subheader("Detection Results")
    img_col, ann_col = st.columns(2)
    with img_col:
        st.image(image, caption="Original Image", use_container_width=True)
    with ann_col:
        st.image(annotated_img, caption=f"Annotated — {len(detections)} detection(s)", use_container_width=True)

    # ---- Summary stats row -------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detections", len(detections))
    c2.metric("Inference Time", f"{inference_ms:.0f} ms")
    c3.metric("Confidence Threshold", f"{conf_threshold:.2f}")

    priority = risk_report["overall_priority"]
    priority_color_hex = risk.priority_color(priority)
    c4.markdown(
        f"**Review Priority**<br>"
        f'<span style="color:{priority_color_hex}; font-size:1.4em; font-weight:700;">{priority}</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- Risk details -------------------------------------------------------
    if risk_report["no_detection"]:
        st.info(
            f"**{risk_report['no_detection_message']}**\n\n"
            f"{risk_report['no_detection_note']}"
        )
    else:
        st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)
        for cr in risk_report["class_results"]:
            priority_emoji = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(
                cr["priority"], "⚪"
            )
            with st.expander(
                f"{priority_emoji} {cr['priority']} — {cr['class_name']}",
                expanded=True,
            ):
                st.markdown(
                    f"**Priority:** {_badge(cr['priority'])}",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Reason:** {cr['reason']}")
                st.markdown(f"**Suggested Action:** {cr['suggested_action']}")
                interval = cr.get("review_interval_days")
                if interval:
                    st.markdown(f"**Suggested Review Interval:** {interval} days *(provisional)*")
                if cr.get("severity_note"):
                    st.caption(f"⚠️ {cr['severity_note']}")

        st.markdown(
            f'<div class="warn-box">{risk_report["disclaimer"]}</div>',
            unsafe_allow_html=True,
        )

    # ---- Detection table ---------------------------------------------------
    if detections:
        st.markdown('<div class="section-header">Detection Table</div>', unsafe_allow_html=True)
        det_rows = []
        class_priority_map = {
            cr["class_name"]: cr["priority"] for cr in risk_report.get("class_results", [])
        }
        for d in detections:
            bbox = d.get("bbox", [None]*4)
            det_rows.append({
                "Class": d["class_name"],
                "Confidence": f"{d['confidence']:.3f}",
                "Priority": class_priority_map.get(d["class_name"], "—"),
                "x1": f"{bbox[0]:.1f}" if bbox[0] is not None else "—",
                "y1": f"{bbox[1]:.1f}" if bbox[1] is not None else "—",
                "x2": f"{bbox[2]:.1f}" if bbox[2] is not None else "—",
                "y2": f"{bbox[3]:.1f}" if bbox[3] is not None else "—",
            })
        st.dataframe(pd.DataFrame(det_rows), use_container_width=True, hide_index=True)

    # ---- Downloads ---------------------------------------------------------
    st.divider()
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    # Download annotated image
    ann_buf = io.BytesIO()
    annotated_img.save(ann_buf, format="PNG")
    ann_buf.seek(0)
    dl_col1.download_button(
        "⬇️ Download Annotated Image",
        data=ann_buf,
        file_name=f"annotated_{Path(uploaded.name).stem}.png",
        mime="image/png",
    )

    # Enrich detections with priority for CSV
    det_with_priority = []
    class_rule_map = {cr["class_name"]: cr for cr in risk_report.get("class_results", [])}
    for d in detections:
        rule = class_rule_map.get(d["class_name"], {})
        det_with_priority.append(
            {**d, "priority": rule.get("priority", ""), "suggested_action": rule.get("suggested_action", ""),
             "review_interval_days": rule.get("review_interval_days", "")}
        )

    csv_bytes = reports.detections_to_csv_bytes(
        det_with_priority, uploaded.name, site_ref or None
    )
    dl_col2.download_button(
        "⬇️ Download Detection CSV",
        data=csv_bytes,
        file_name=f"detections_{Path(uploaded.name).stem}.csv",
        mime="text/csv",
    )

    # Save to DB
    if dl_col3.button("💾 Save Inspection to Database"):
        # Save annotated image to outputs/
        outputs_dir = Path("data/outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        ann_path = outputs_dir / f"ann_{ts_str}_{Path(uploaded.name).stem}.png"
        annotated_img.save(str(ann_path))

        # Save report
        report_path = reports.save_report_to_disk(
            det_with_priority, uploaded.name, site_ref or None
        )

        insp_id = database.save_inspection(
            filename=uploaded.name,
            site_reference=site_ref or None,
            model_name=Path(result["model_path"]).name if result["model_path"] else "unknown",
            conf_threshold=conf_threshold,
            detections=detections,
            risk_report=risk_report,
            annotated_image_path=str(ann_path),
            report_path=str(report_path),
        )
        st.success(f"Inspection saved. Record ID: **#{insp_id}**")


# ===========================================================================
# Page C: Inspection History
# ===========================================================================
elif page == "📋 Inspection History":
    st.title("📋 Inspection History")
    st.caption(
        "Records are stored in a local SQLite database. "
        "Site references are user-entered grouping labels, not verified panel identities."
    )

    # ---- Filters -----------------------------------------------------------
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])

    with col_f1:
        all_refs = database.get_all_site_references()
        ref_options = ["All"] + all_refs
        selected_ref = col_f1.selectbox("Filter by Site Reference", ref_options)

    with col_f2:
        priority_filter = col_f2.selectbox(
            "Filter by Priority",
            ["All", "Critical", "High", "Medium", "Low", "None"],
        )

    with col_f3:
        limit = col_f3.number_input("Max records", 10, 500, 100, 10)

    inspections = database.get_inspections(
        limit=int(limit),
        site_reference=selected_ref if selected_ref != "All" else None,
        priority=priority_filter if priority_filter != "All" else None,
    )

    # ---- Export -----------------------------------------------------------
    if inspections:
        csv_data = reports.inspections_to_csv_bytes(inspections)
        st.download_button(
            "⬇️ Export All Records to CSV",
            data=csv_data,
            file_name="inspection_history.csv",
            mime="text/csv",
        )

    st.divider()

    if not inspections:
        st.info("No inspection records found matching the selected filters.")
    else:
        st.markdown(f"**{len(inspections)} record(s) shown**")

        # Table view
        df = pd.DataFrame(inspections)
        display_cols = [c for c in [
            "id", "timestamp", "filename", "site_reference",
            "num_detections", "overall_priority", "model_name", "conf_threshold"
        ] if c in df.columns]
        df_display = df[display_cols].copy()
        df_display["filename"] = df_display["filename"].apply(lambda x: Path(x).name)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.divider()

        # ---- Detail view ---------------------------------------------------
        st.markdown('<div class="section-header">Inspection Detail</div>', unsafe_allow_html=True)
        sel_id = st.number_input(
            "Enter Inspection ID to view details",
            min_value=1,
            step=1,
            value=inspections[0]["id"],
        )

        if st.button("View Inspection Details"):
            detail = database.get_inspection_by_id(int(sel_id))
            if detail is None:
                st.error(f"Inspection #{sel_id} not found.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Inspection ID", f"#{detail['id']}")
                c2.metric("Timestamp", detail["timestamp"])
                c3.metric("Filename", Path(detail["filename"]).name)

                c4, c5, c6 = st.columns(3)
                c4.metric("Site Reference", detail.get("site_reference") or "—")
                c5.metric("Detections", detail["num_detections"])
                c6.markdown(
                    f"**Priority:** {_badge(detail.get('overall_priority','None'))}",
                    unsafe_allow_html=True,
                )

                if detail.get("detections"):
                    st.markdown("**Detections:**")
                    det_df = pd.DataFrame(detail["detections"])
                    show_cols = [c for c in [
                        "class_name", "confidence", "priority",
                        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                        "suggested_action", "review_interval_days"
                    ] if c in det_df.columns]
                    st.dataframe(det_df[show_cols], use_container_width=True, hide_index=True)
                else:
                    st.info("No detections recorded for this inspection.")

                ann_path = detail.get("annotated_image", "")
                if ann_path and Path(ann_path).exists():
                    st.image(ann_path, caption="Saved Annotated Image", use_container_width=True)

        # ---- Comparison view -----------------------------------------------
        st.divider()
        st.markdown('<div class="section-header">Compare Two Inspections</div>', unsafe_allow_html=True)
        st.caption(
            "Compare model detections between two separate uploads. "
            "Differences in detections do not prove physical panel degradation."
        )

        cmp_col1, cmp_col2 = st.columns(2)
        cmp_id_a = cmp_col1.number_input("Inspection A (ID)", min_value=1, step=1, key="cmp_a", value=inspections[0]["id"])
        cmp_id_b = cmp_col2.number_input("Inspection B (ID)", min_value=1, step=1, key="cmp_b",
                                          value=inspections[1]["id"] if len(inspections) > 1 else inspections[0]["id"])

        if st.button("Compare"):
            rec_a = database.get_inspection_by_id(int(cmp_id_a))
            rec_b = database.get_inspection_by_id(int(cmp_id_b))
            if rec_a is None or rec_b is None:
                st.error("One or both inspection IDs not found.")
            else:
                col_a, col_b = st.columns(2)
                for col, rec, lbl in [(col_a, rec_a, "A"), (col_b, rec_b, "B")]:
                    with col:
                        st.markdown(f"**Inspection {lbl}: #{rec['id']}**")
                        st.write(f"File: `{Path(rec['filename']).name}`")
                        st.write(f"Ref: `{rec.get('site_reference') or '—'}`")
                        st.write(f"Timestamp: `{rec['timestamp']}`")
                        st.write(f"Detections: **{rec['num_detections']}**")
                        st.markdown(
                            f"Priority: {_badge(rec.get('overall_priority','None'))}",
                            unsafe_allow_html=True,
                        )
                        if rec.get("detections"):
                            classes = [d["class_name"] for d in rec["detections"]]
                            st.write("Classes detected:", ", ".join(classes) or "None")
                        ann = rec.get("annotated_image", "")
                        if ann and Path(ann).exists():
                            st.image(ann, caption=f"Inspection {lbl}", use_container_width=True)


# ===========================================================================
# Page D: Model & Dataset Info
# ===========================================================================
elif page == "ℹ️ Model & Dataset Info":
    st.title("ℹ️ Model & Dataset Information")

    # ---- Model status -------------------------------------------------------
    st.subheader("Custom Model")
    model_info = detector.get_model_info()

    if model_info["available"]:
        st.success(f"✅ Custom weights found: `{model_info['path']}`")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Weight File", Path(model_info["path"]).name)
        mc2.metric("Last Modified", model_info["modified"])
        mc3.metric("File Size", f"{model_info['size_mb']} MB")

        meta_path = Path("models/training_meta.json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            st.markdown("**Training Configuration:**")
            st.json(meta)
    else:
        st.error(
            "⚠️ No custom model weights found in `models/`.\n\n"
            "```bash\n"
            "# Quick training (~2 epochs, subset):\n"
            "python train_model.py\n\n"
            "# Full training:\n"
            "python train_model.py --mode full --epochs 30\n"
            "```"
        )

    st.divider()

    # ---- Class reference ---------------------------------------------------
    st.subheader("Supported Fault Classes")
    rules_path = Path("config/risk_rules.json")
    if rules_path.exists():
        with open(rules_path) as f:
            rules_data = json.load(f)
        class_rules = rules_data.get("class_rules", {})
        rows = []
        for i, cls in enumerate(detector.CLASS_NAMES):
            rule = class_rules.get(cls, {})
            rows.append({
                "ID": i,
                "Class Name": cls,
                "Default Priority": rule.get("priority", "—"),
                "Suggested Action": rule.get("suggested_action", "—"),
                "Provisional Review (days)": rule.get("review_interval_days", "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(rules_data.get("disclaimer", ""))
    else:
        for i, cls in enumerate(detector.CLASS_NAMES):
            st.write(f"**{i}** — {cls}")

    st.divider()

    # ---- Dataset summary ---------------------------------------------------
    st.subheader("Dataset Summary")
    summary_path = Path("data/processed/dataset_summary.json")
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        for split, stats in summary.get("splits", {}).items():
            with st.expander(f"**{split}** split"):
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Images", stats["images"])
                sc2.metric("Valid Annotations", stats["valid_annotation_lines"])
                sc3.metric("Malformed Lines", stats["malformed_annotation_lines"])
                sc4.metric("Empty Labels", stats["empty_label_files"])
        st.metric("Total Malformed Lines Excluded", summary.get("total_malformed_lines", "—"))
    else:
        st.info(
            "Dataset summary not found. Run the preparation script first:\n"
            "```bash\npython scripts/prepare_dataset.py\n```"
        )

    st.divider()

    # ---- Limitations -------------------------------------------------------
    st.subheader("Dataset Limitations")
    st.markdown(
        """
        <div class="warn-box">
        <b>Important limitations of this dataset (from the audit):</b>
        <ul>
        <li><b>Grayscale only</b> — thermal temperature data has been converted to a
            grayscale rendering. Calibrated temperature readings are not available.</li>
        <li><b>No temporal capability</b> — only two distinct flight dates (2023-08-09
            and 2024-04-11). No panel IDs link repeated inspections of the same panel.
            Temporal fault prediction is not supported.</li>
        <li><b>Severe class imbalance (67:1)</b> — SingleHotSpot (36.8%) vs.
            StringReversedPolarity (0.5%).</li>
        <li><b>Tiny test set</b> — 15 images, 5 of 8 classes. Not suitable for
            statistically meaningful final evaluation.</li>
        <li><b>No GPS or panel IDs</b> — cannot build geographic fault maps or link
            inspections to specific panels.</li>
        <li><b>Augmentation leakage risk</b> — each source image produces ~3 augmented
            siblings; verify split integrity before reporting final metrics.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">
        <b>What this application can demonstrate:</b><br>
        ✅ Custom YOLOv8n fault detection on user-uploaded images<br>
        ✅ Bounding box visualisation for 8 fault classes<br>
        ✅ Rule-based review priority assignment<br>
        ✅ Persistent inspection records across sessions<br>
        ✅ Detection comparison between separate uploads<br>
        <br>
        <b>What this application does not do:</b><br>
        ❌ Verify panel identity or GPS location<br>
        ❌ Predict future faults from historical trends<br>
        ❌ Provide calibrated temperature readings<br>
        ❌ Autonomously control drone re-inspection<br>
        ❌ Perform root-cause diagnosis<br>
        </div>
        """,
        unsafe_allow_html=True,
    )
