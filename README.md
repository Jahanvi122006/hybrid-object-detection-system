# Hybrid Object Detection System

BCA project prototype implementing:

1. YOLOv8n object detection
2. Fuzzy confidence membership
3. Genetic Algorithm confidence-threshold optimization
4. Hybrid filtering and OpenCV visualization
5. Streamlit image upload, photo capture and live webcam modes

## Project structure

```text
hybrid_object_detection_system/
├── app.py
├── config.py
├── main.py
├── requirements.txt
├── yolo_module/
│   ├── __init__.py
│   ├── yolov_model.py
│   └── detection.py
├── fuzzy_system/
│   ├── __init__.py
│   └── inference.py
├── genetic_algorithm/
│   ├── __init__.py
│   └── ga_optimizer.py
└── utils/
    ├── __init__.py
    └── plots.py
```

## Installation

Use Python 3.10-3.12 in a fresh virtual environment.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

or:

```bash
python main.py
```

On the first run, Ultralytics will download `yolov8n.pt` if it is not already present. Internet access is therefore required for the first model download.

## Notes about the pipeline

YOLO inference uses a low candidate threshold (`YOLO_INFERENCE_CONF=0.01`) so the hybrid stages can work with a broad set of candidate detections. The final displayed detections are controlled by the GA threshold.

The fuzzy membership is:

- confidence <= 0.40: membership 0
- 0.40 < confidence < 0.70: linear membership
- confidence >= 0.70: membership 1

The displayed Fuzzy count keeps detections with membership >= 0.50. The Hybrid count keeps detections whose confidence is >= the GA-optimized threshold.

The GA fitness is:

`0.7 * average retained confidence + 0.3 * retention proportion`

The live webcam mode uses `streamlit-webrtc`, so browser camera permission is required. Running on localhost is the easiest setup for a college demonstration.
