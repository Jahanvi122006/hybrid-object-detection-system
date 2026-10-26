"""Project-wide configuration for the Hybrid Object Detection System."""

from pathlib import Path

# YOLO model
MODEL_PATH = "yolov8n.pt"
YOLO_IMAGE_SIZE = 640
# Use a low inference threshold so the hybrid pipeline receives enough
# candidate detections to optimize its final threshold.
YOLO_INFERENCE_CONF = 0.01
YOLO_IOU = 0.70

# Fuzzy membership parameters
LOW_CONF = 0.40
HIGH_CONF = 0.70

# Genetic Algorithm parameters
POP_SIZE = 10
GENERATIONS = 20
MUTATION = 0.10
TOURNAMENT_SIZE = 3
ELITISM = 2
THRESHOLD_MIN = LOW_CONF
THRESHOLD_MAX = HIGH_CONF
RANDOM_SEED = 42

# Fitness weights
CONFIDENCE_WEIGHT = 0.70
RETENTION_WEIGHT = 0.30

# Fuzzy-stage decision boundary used for the displayed Fuzzy count.
# membership >= 0.5 corresponds to confidence >= 0.55 for the linear rule.
FUZZY_MEMBERSHIP_THRESHOLD = 0.50

# UI / visualization
BOX_THICKNESS = 2
FONT_SCALE = 0.55
MAX_DETECTIONS_FOR_GA = 200

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
