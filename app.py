"""Streamlit web application for the Hybrid Object Detection System."""

from __future__ import annotations

import os
import threading
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

import config
from fuzzy_system.inference import fuzzy_filter
from genetic_algorithm.ga_optimizer import optimize_threshold
from utils.plots import (
    draw_detections,
    plot_ga_history,
    plot_stage_comparison,
    stage_counts,
)
from yolo_module.detection import extract_detections
from yolo_module.yolov_model import YOLOv8Detector


# ------------------------- Page Configuration -------------------------

st.set_page_config(
    page_title="Hybrid Object Detection System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------- Styling -------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: #f7faf8;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #e9f6e8 0%,
            #f7faf8 100%
        );
    }

    .hero {
        padding: 28px 30px;
        border-radius: 20px;
        margin-bottom: 24px;
        background: linear-gradient(
            135deg,
            #dff3dd,
            #f5fbf3
        );
        border: 1px solid #cfe7cc;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.35rem;
        color: #123b19;
    }

    .hero p {
        margin: 8px 0 0;
        color: #45634b;
        font-size: 1.05rem;
    }

    .card {
        padding: 20px;
        border-radius: 16px;
        background: white;
        border: 1px solid #e2e9e3;
        box-shadow: 0 3px 14px rgba(0,0,0,.04);
        height: 100%;
    }

    .card h3 {
        margin-top: 0;
        color: #173c1d;
    }

    .step {
        padding: 14px 16px;
        border-left: 4px solid #75ad70;
        background: #f5faf4;
        margin: 8px 0;
        border-radius: 8px;
    }

    .badge {
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        background: #dff0dc;
        color: #245a2b;
        font-weight: 600;
        margin-right: 6px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e0e9e1;
        padding: 12px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------- YOLO Model -------------------------

@st.cache_resource
def load_detector():
    return YOLOv8Detector(config.MODEL_PATH)


# ------------------------- Processing -------------------------

def process_frame(
    detector: YOLOv8Detector,
    frame_bgr: np.ndarray
):
    result = detector.predict(frame_bgr)

    detections = extract_detections(result)

    confidences = [
        float(d.confidence)
        for d in detections
    ]

    fuzzy_scores, fuzzy_mask = fuzzy_filter(
        confidences
    )

    # ---------------- GA Optimization ----------------

    if len(confidences) > 0:

        ga_indices = list(
            range(len(confidences))
        )

        if len(ga_indices) > config.MAX_DETECTIONS_FOR_GA:

            ga_indices = sorted(
                ga_indices,
                key=lambda i: confidences[i],
                reverse=True
            )[
                :config.MAX_DETECTIONS_FOR_GA
            ]

        ga_confidences = [
            confidences[i]
            for i in ga_indices
        ]

        threshold, fitness, history = (
            optimize_threshold(
                ga_confidences
            )
        )

        hybrid_mask = [
            confidence >= threshold
            for confidence in confidences
        ]

    else:

        threshold = 1.0
        fitness = 0.0
        history = []

        hybrid_mask = [
            False
            for _ in confidences
        ]

    # ---------------- Drawing ----------------

    annotated = draw_detections(
        frame_bgr,
        detections,
        hybrid_mask,
        "Hybrid Detection"
    )

    counts = stage_counts(
        len(detections),
        sum(fuzzy_mask),
        sum(hybrid_mask)
    )

    metrics = {
        "counts": counts,
        "threshold": threshold,
        "fitness": fitness,
        "history": history,
        "fuzzy_scores": fuzzy_scores,
        "detections": detections,
        "hybrid_mask": hybrid_mask,
    }

    return annotated, metrics


# ------------------------- Image Conversion -------------------------

def bytes_to_rgb(
    uploaded_file
) -> Optional[np.ndarray]:

    if uploaded_file is None:
        return None

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        return np.array(image)

    except Exception as e:

        st.error(
            f"Could not read image: {e}"
        )

        return None


# ------------------------- Webcam Processor -------------------------

class LiveVideoProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.detector = load_detector()

        self.lock = threading.Lock()

        self.last_metrics = {
            "counts": {
                "YOLO": 0,
                "Fuzzy": 0,
                "Hybrid": 0
            },
            "threshold": 0.0,
            "fitness": 0.0,
            "history": [],
            "fuzzy_scores": [],
            "detections": [],
            "hybrid_mask": []
        }

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        annotated, metrics = process_frame(
            self.detector,
            img
        )

        with self.lock:
            self.last_metrics = metrics

        return frame.from_ndarray(
            annotated,
            format="bgr24"
        )


# ------------------------- Metrics -------------------------

def show_metrics(metrics):

    counts = metrics["counts"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "YOLO detections",
        counts["YOLO"]
    )

    c2.metric(
        "Fuzzy detections",
        counts["Fuzzy"]
    )

    c3.metric(
        "Hybrid detections",
        counts["Hybrid"]
    )

    c4.metric(
        "GA threshold",
        f"{metrics['threshold']:.3f}"
    )


# ------------------------- Image Detection -------------------------

def run_image_mode(
    image_rgb: np.ndarray
):

    detector = load_detector()

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    annotated_bgr, metrics = process_frame(
        detector,
        image_bgr
    )

    annotated_rgb = cv2.cvtColor(
        annotated_bgr,
        cv2.COLOR_BGR2RGB
    )

    show_metrics(metrics)

    st.subheader(
        "Detection Result"
    )

    left, right = st.columns(2)

    with left:

        st.image(
            image_rgb,
            caption="Input Image",
            use_container_width=True
        )

    with right:

        st.image(
            annotated_rgb,
            caption="Final Hybrid Output",
            use_container_width=True
        )

    # ---------------- Detected Objects ----------------

    st.subheader(
        "Detected Objects"
    )

    rows = []

    for det, keep, fuzzy in zip(
        metrics["detections"],
        metrics["hybrid_mask"],
        metrics["fuzzy_scores"]
    ):

        if keep:

            rows.append(
                {
                    "Object": det.class_name,
                    "YOLO Confidence":
                        round(
                            float(det.confidence),
                            4
                        ),
                    "Fuzzy Membership":
                        round(
                            float(fuzzy),
                            4
                        ),
                    "Final Status":
                        "Retained"
                }
            )

    if rows:

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No objects passed the final "
            "hybrid threshold for this image."
        )

    # ---------------- Pipeline Analysis ----------------

    st.subheader(
        "Pipeline Analysis"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.pyplot(
            plot_stage_comparison(
                metrics["counts"]
            ),
            clear_figure=True
        )

    with col2:

        if metrics["history"]:

            st.pyplot(
                plot_ga_history(
                    metrics["history"]
                ),
                clear_figure=True
            )

        else:

            st.info(
                "GA history is not available "
                "because no detections were found."
            )

    # ---------------- Explanation ----------------

    with st.expander(
        "How the hybrid pipeline works"
    ):

        st.markdown(
            f"""
            <div class='step'>
            <b>1. YOLOv8n:</b>
            generates candidate object detections.
            </div>

            <div class='step'>
            <b>2. Fuzzy Logic:</b>
            converts confidence into a membership
            score using the configured
            {config.LOW_CONF:.2f}–{config.HIGH_CONF:.2f}
            range.
            </div>

            <div class='step'>
            <b>3. Genetic Algorithm:</b>
            searches for a suitable final
            confidence threshold.
            </div>

            <div class='step'>
            <b>4. Hybrid Output:</b>
            detections at or above the optimized
            threshold are displayed.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(
            "GA fitness for this input: "
            f"**{metrics['fitness']:.4f}**"
        )


# ------------------------- Demo Images -------------------------

def demo_images_section():

    st.markdown(
        "## 🎬 Demo Images"
    )

    st.caption(
        "Use a built-in demo image for your "
        "presentation."
    )

    demo_dir = os.path.join(
        os.path.dirname(__file__),
        "demo_images"
    )

    demo_files = {

        "People & Vehicles":
            "people_vehicles.jpg",

        "Single Person":
            "person.jpg",

        "Cars":
            "cars.jpg"
    }

    available = []

    for label, filename in demo_files.items():

        path = os.path.join(
            demo_dir,
            filename
        )

        if os.path.exists(path):

            available.append(
                (label, filename)
            )

    if not available:

        st.info(
            "No demo images found. "
            "Please upload your own image."
        )

        return

    demo_cols = st.columns(
        len(available)
    )

    for col, (label, filename) in zip(
        demo_cols,
        available
    ):

        path = os.path.join(
            demo_dir,
            filename
        )

        with col:

            st.image(
                path,
                caption=label,
                use_container_width=True
            )

            if st.button(
                f"Use {label}",
                key=f"demo_{filename}"
            ):

                image_rgb = bytes_to_rgb(
                    Image.open(path)
                )

                if image_rgb is not None:

                    st.session_state[
                        "selected_demo_image"
                    ] = image_rgb

    if (
        "selected_demo_image"
        in st.session_state
    ):

        st.success(
            "Demo image selected."
        )

        run_image_mode(
            st.session_state[
                "selected_demo_image"
            ]
        )


# ------------------------- Model Comparison -------------------------

def comparison_page():

    st.markdown(
        "## 📊 Model Performance Comparison"
    )

    st.caption(
        "Reference comparison values used "
        "in the project presentation/report."
    )

    data = pd.DataFrame(
        [
            [
                "SSD",
                "81.20%",
                0.8120,
                0.8035,
                0.8077,
                0.8041
            ],

            [
                "Faster R-CNN",
                "85.65%",
                0.8563,
                0.8491,
                0.8526,
                0.8468
            ],

            [
                "YOLO (Only)",
                "88.10%",
                0.8847,
                0.8736,
                0.8774,
                0.8729
            ],

            [
                "Hybrid Model "
                "(YOLO + Fuzzy + GA)",
                "92.55%",
                0.9213,
                0.9183,
                0.9219,
                0.9194
            ]
        ],

        columns=[
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "Macro-F1",
            "mAP@0.5"
        ]
    )

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )

    chart = data.copy()

    chart["Accuracy"] = (
        chart["Accuracy"]
        .str.rstrip("%")
        .astype(float)
        / 100
    )

    st.bar_chart(
        chart.set_index("Model")[
            [
                "Accuracy",
                "Precision",
                "Recall",
                "Macro-F1",
                "mAP@0.5"
            ]
        ]
    )

    st.info(
        "These values should be described as "
        "project comparison/reference results "
        "unless obtained from a controlled "
        "labelled benchmark."
    )


# ------------------------- About Page -------------------------

def about_page():

    st.markdown(
        "## ℹ️ About the Project"
    )

    st.markdown(
        """
        <div class='card'>

        <h3>
        Hybrid Object Detection System
        </h3>

        <p>
        This academic prototype combines
        <b>YOLOv8n</b> object detection with
        <b>Fuzzy Logic</b> confidence filtering
        and a <b>Genetic Algorithm</b> for final
        threshold selection.
        </p>

        <p>
        <span class='badge'>Python</span>
        <span class='badge'>YOLOv8n</span>
        <span class='badge'>OpenCV</span>
        <span class='badge'>Fuzzy Logic</span>
        <span class='badge'>Genetic Algorithm</span>
        <span class='badge'>Streamlit</span>
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "### Processing Workflow"
    )

    steps = [

        (
            "Input",
            "Upload an image, capture a photo, "
            "or use the live webcam."
        ),

        (
            "YOLO Detection",
            "YOLOv8n produces candidate boxes, "
            "classes and confidence values."
        ),

        (
            "Fuzzy Filtering",
            "Confidence values are converted "
            "into fuzzy membership scores."
        ),

        (
            "GA Optimization",
            "The genetic algorithm searches "
            "candidate thresholds."
        ),

        (
            "Hybrid Result",
            "The selected threshold is applied "
            "and the final detections are visualized."
        )
    ]

    for title, text in steps:

        st.markdown(
            f"""
            <div class='step'>
            <b>{title}:</b> {text}
            </div>
            """,
            unsafe_allow_html=True
        )


# ------------------------- Detection Page -------------------------

def detection_page(mode):

    st.markdown(
        f"## {mode}"
    )

    # ---------------- Upload Image ----------------

    if mode == "Upload Image":

        demo_images_section()

        st.divider()

        st.subheader(
            "Upload Your Image"
        )

        uploaded = st.file_uploader(
            "Choose an image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ]
        )

        if uploaded is not None:

            image_rgb = bytes_to_rgb(
                uploaded
            )

            if image_rgb is not None:

                run_image_mode(
                    image_rgb
                )

    # ---------------- Capture Photo ----------------

    elif mode == "Capture Photo":

        captured = st.camera_input(
            "Take a photo"
        )

        if captured is not None:

            image_rgb = bytes_to_rgb(
                captured
            )

            if image_rgb is not None:

                run_image_mode(
                    image_rgb
                )

    # ---------------- Live Webcam ----------------

    elif mode == "Live Webcam":

        st.info(
            "Allow camera access in your browser, "
            "then start the webcam."
        )

        ctx = webrtc_streamer(

            key="hybrid-object-detection",

            mode=WebRtcMode.SENDRECV,

            video_processor_factory=
                LiveVideoProcessor,

            media_stream_constraints={
                "video": True,
                "audio": False
            },

            async_processing=True
        )

        if ctx.video_processor:

            with ctx.video_processor.lock:

                metrics = dict(
                    ctx.video_processor.last_metrics
                )

            show_metrics(
                metrics
            )


# ------------------------- Main Application -------------------------

def main():

    st.sidebar.markdown(
        "# 🎯 Hybrid AI"
    )

    st.sidebar.caption(
        "YOLOv8n + Fuzzy Logic + Genetic Algorithm"
    )

    page = st.sidebar.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "📤 Upload Image",
            "📷 Capture Photo",
            "🎥 Live Webcam",
            "📊 Model Comparison",
            "ℹ️ About"
        ]
    )

    st.sidebar.divider()

    st.sidebar.caption(
        "Academic Project Prototype"
    )

    # ---------------- Dashboard ----------------

    if page == "🏠 Dashboard":

        st.markdown(
            """
            <div class='hero'>

            <h1>
            🎯 Hybrid Object Detection System
            </h1>

            <p>
            Intelligent object detection using
            YOLOv8n, Fuzzy Logic and Genetic
            Algorithm optimization.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.markdown(
                """
                <div class='card'>

                <h3>
                🔎 YOLOv8n
                </h3>

                <p>
                Fast initial object detection
                with bounding boxes, class labels
                and confidence scores.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:

            st.markdown(
                """
                <div class='card'>

                <h3>
                🧠 Fuzzy Logic
                </h3>

                <p>
                Converts confidence values into
                gradual membership scores
                for filtering.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:

            st.markdown(
                """
                <div class='card'>

                <h3>
                🧬 Genetic Algorithm
                </h3>

                <p>
                Searches candidate confidence
                thresholds before producing the
                final hybrid output.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            "### 🚀 Start Detection"
        )

        st.write(
            "Choose **Upload Image**, "
            "**Capture Photo**, or "
            "**Live Webcam** from the sidebar "
            "to run the complete pipeline."
        )

        st.markdown(
            "### Workflow"
        )

        st.markdown(
            "`Input → YOLOv8n → Detection Extraction "
            "→ Fuzzy Filtering → GA Threshold "
            "Optimization → Hybrid Result`"
        )

    # ---------------- Comparison ----------------

    elif page == "📊 Model Comparison":

        comparison_page()

    # ---------------- About ----------------

    elif page == "ℹ️ About":

        about_page()

    # ---------------- Detection ----------------

    else:

        detection_page(
            page.split(" ", 1)[1]
        )


# ------------------------- Run -------------------------

if __name__ == "__main__":
    main()