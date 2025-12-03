"""
Configuration file for AI-Assisted Fraud Detection System
Contains all hyperparameters, paths, and model settings
"""

import os
from pathlib import Path

# ==================== PATHS ====================
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODEL_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==================== MODEL SETTINGS ====================
# Classification settings
NUM_CLASSES = 2  # Binary: 0=Genuine, 1=AI-Assisted
CLASS_NAMES = ['Genuine', 'AI-Assisted']
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for positive prediction

# ==================== FEATURE EXTRACTION ====================
# MediaPipe FaceMesh settings
FACE_DETECTION_CONFIDENCE = 0.5
FACE_TRACKING_CONFIDENCE = 0.5
MAX_NUM_FACES = 1  # Process single face per frame

# Eye landmark indices (MediaPipe FaceMesh)
LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 373, 374, 380]
IRIS_INDICES = [468, 469, 470, 471, 472]  # Left iris landmarks

# Head pose landmarks
HEAD_POSE_LANDMARKS = {
    'nose_tip': 1,
    'chin': 152,
    'left_eye': 33,
    'right_eye': 263,
    'left_mouth': 61,
    'right_mouth': 291
}

# Emotion detection settings (DeepFace)
EMOTION_MODELS = ['Emotion']  # Use emotion model
EMOTION_ACTIONS = ['emotion']
ENFORCE_DETECTION = False  # Don't fail on no face detected

# ==================== BEHAVIORAL FEATURES ====================
FEATURE_NAMES = [
    'eye_movement_freq',      # Gaze direction changes per second
    'eye_fixation_duration',  # Average fixation time (ms)
    'head_pose_variance',     # Yaw/pitch/roll standard deviation
    'head_stability',         # Movement smoothness score
    'response_delay',         # Time from question to answer (s)
    'emotion_stability',      # Emotion variance over time
    'emotion_intensity',      # Average emotion strength
    'micro_expression_count', # Brief involuntary expressions
    'blink_rate',            # Blinks per minute
    'speech_pause_frequency', # Pauses during response
    'gaze_dispersion',       # Spatial spread of gaze points
    'cognitive_load_score'   # Combined stress indicators
]

NUM_FEATURES = len(FEATURE_NAMES)

# Feature extraction window settings
WINDOW_SIZE_SECONDS = 5.0  # Analyze 5-second windows
OVERLAP_SECONDS = 2.5      # 50% overlap between windows

# Behavioral thresholds
BLINK_DURATION_THRESHOLD = 0.15  # Seconds
FIXATION_THRESHOLD = 0.3         # Seconds
GAZE_CHANGE_THRESHOLD = 15       # Degrees

# ==================== TRAINING SETTINGS ====================
# Data split
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

# Model training
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10

# Class balancing
USE_CLASS_WEIGHTS = True
CLASS_WEIGHT_RATIO = 1.5  # Weight for minority class

# Data augmentation
AUGMENTATION_PROB = 0.5
AUGMENTATION_PARAMS = {
    'brightness_range': (0.8, 1.2),
    'rotation_range': (-5, 5),
    'zoom_range': (0.95, 1.05),
    'flip_horizontal': False,  # Don't flip faces
    'add_noise': True,
    'blur': True
}

# ==================== ENSEMBLE SETTINGS ====================
ENSEMBLE_MODELS = {
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True
    },
    'neural_network': {
        'hidden_layers': [256, 128, 64],
        'dropout_rate': 0.3,
        'activation': 'relu',
        'optimizer': 'adam'
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8
    }
}

ENSEMBLE_VOTING = 'soft'  # Use probability averaging
ENSEMBLE_WEIGHTS = [0.4, 0.35, 0.25]  # RF, NN, GB

# ==================== EVALUATION SETTINGS ====================
METRICS_TO_COMPUTE = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'roc_auc',
    'confusion_matrix',
    'classification_report'
]

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8'  # Matplotlib style
FIGURE_SIZE = (12, 8)
DPI = 300
COLOR_PALETTE = 'husl'

# SHAP explainability
SHAP_BACKGROUND_SAMPLES = 100
SHAP_TEST_SAMPLES = 50

# ==================== STREAMLIT APP SETTINGS ====================
APP_TITLE = "🎯 AI-Assisted Fraud Detection"
APP_ICON = "🎯"
PAGE_CONFIG = {
    'page_title': 'AI Fraud Detection',
    'page_icon': '🎯',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Video input settings
MAX_VIDEO_SIZE_MB = 100
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
WEBCAM_RESOLUTION = (640, 480)
WEBCAM_FPS = 30

# Real-time processing
REALTIME_BUFFER_SIZE = 150  # Frames (5 seconds at 30fps)
PREDICTION_UPDATE_FREQUENCY = 30  # Update every 30 frames (1 second)

# ==================== LOGGING SETTINGS ====================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = LOGS_DIR / 'ai_fraud_detection.log'

# ==================== SYNTHETIC DATA GENERATION ====================
SYNTHETIC_DATA_PARAMS = {
    'genuine': {
        'eye_movement_freq': {'mean': 2.5, 'std': 0.8, 'min': 0.5, 'max': 5.0},
        'eye_fixation_duration': {'mean': 350, 'std': 100, 'min': 150, 'max': 800},
        'head_pose_variance': {'mean': 8.0, 'std': 3.0, 'min': 2.0, 'max': 20.0},
        'head_stability': {'mean': 0.75, 'std': 0.15, 'min': 0.3, 'max': 1.0},
        'response_delay': {'mean': 1.2, 'std': 0.5, 'min': 0.3, 'max': 3.0},
        'emotion_stability': {'mean': 0.65, 'std': 0.15, 'min': 0.2, 'max': 1.0},
        'emotion_intensity': {'mean': 0.7, 'std': 0.15, 'min': 0.3, 'max': 1.0},
        'micro_expression_count': {'mean': 8, 'std': 3, 'min': 2, 'max': 20},
        'blink_rate': {'mean': 17, 'std': 5, 'min': 8, 'max': 30},
        'speech_pause_frequency': {'mean': 3.5, 'std': 1.5, 'min': 0, 'max': 10},
        'gaze_dispersion': {'mean': 45, 'std': 15, 'min': 20, 'max': 100},
        'cognitive_load_score': {'mean': 0.6, 'std': 0.2, 'min': 0.2, 'max': 1.0}
    },
    'ai_assisted': {
        'eye_movement_freq': {'mean': 6.5, 'std': 1.5, 'min': 4.0, 'max': 12.0},
        'eye_fixation_duration': {'mean': 180, 'std': 60, 'min': 80, 'max': 350},
        'head_pose_variance': {'mean': 3.5, 'std': 1.5, 'min': 0.5, 'max': 8.0},
        'head_stability': {'mean': 0.9, 'std': 0.08, 'min': 0.7, 'max': 1.0},
        'response_delay': {'mean': 4.5, 'std': 1.5, 'min': 2.5, 'max': 10.0},
        'emotion_stability': {'mean': 0.85, 'std': 0.1, 'min': 0.6, 'max': 1.0},
        'emotion_intensity': {'mean': 0.45, 'std': 0.15, 'min': 0.15, 'max': 0.8},
        'micro_expression_count': {'mean': 3, 'std': 2, 'min': 0, 'max': 8},
        'blink_rate': {'mean': 12, 'std': 4, 'min': 5, 'max': 20},
        'speech_pause_frequency': {'mean': 2.0, 'std': 1.0, 'min': 0, 'max': 5},
        'gaze_dispersion': {'mean': 25, 'std': 8, 'min': 10, 'max': 45},
        'cognitive_load_score': {'mean': 0.35, 'std': 0.15, 'min': 0.1, 'max': 0.7}
    }
}

# ==================== DEVICE SETTINGS ====================
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 4  # For data loading
PIN_MEMORY = True if DEVICE == 'cuda' else False

# ==================== VERSION INFO ====================
VERSION = "1.0.0"
AUTHOR = "AI Fraud Detection Team"
LAST_UPDATED = "2025-10-07"

print(f"[CONFIG] Loaded configuration v{VERSION}")
print(f"[CONFIG] Using device: {DEVICE}")
print(f"[CONFIG] Base directory: {BASE_DIR}")
