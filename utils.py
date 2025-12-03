"""
Utility functions for AI-Assisted Fraud Detection System
Includes helpers for video processing, visualization, and data handling
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== VIDEO PROCESSING UTILITIES ====================

def load_video(video_path: Union[str, Path]) -> Tuple[List[np.ndarray], int, int]:
    """
    Load video and extract frames
    
    Args:
        video_path: Path to video file
        
    Returns:
        frames: List of frames as numpy arrays
        fps: Frame rate of video
        total_frames: Total number of frames
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    logger.info(f"Loaded {len(frames)} frames from {video_path} at {fps} FPS")
    return frames, fps, total_frames


def save_video(frames: List[np.ndarray], output_path: Union[str, Path], 
               fps: int = 30, codec: str = 'mp4v') -> None:
    """
    Save frames as video file
    
    Args:
        frames: List of frames as numpy arrays
        output_path: Output video path
        fps: Frame rate
        codec: Video codec
    """
    output_path = str(output_path)
    height, width = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    logger.info(f"Saved video to {output_path}")


def sample_frames(frames: List[np.ndarray], num_frames: int, 
                  method: str = 'uniform') -> List[np.ndarray]:
    """
    Sample frames from video
    
    Args:
        frames: List of all frames
        num_frames: Number of frames to sample
        method: Sampling method ('uniform', 'random', 'middle')
        
    Returns:
        Sampled frames
    """
    total_frames = len(frames)
    
    if total_frames <= num_frames:
        return frames
    
    if method == 'uniform':
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    elif method == 'random':
        indices = np.random.choice(total_frames, num_frames, replace=False)
        indices.sort()
    elif method == 'middle':
        start = (total_frames - num_frames) // 2
        indices = np.arange(start, start + num_frames)
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    return [frames[i] for i in indices]


def resize_frame(frame: np.ndarray, size: Tuple[int, int], 
                 maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize frame to target size
    
    Args:
        frame: Input frame
        size: Target (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized frame
    """
    if maintain_aspect:
        h, w = frame.shape[:2]
        target_w, target_h = size
        aspect = w / h
        
        if w > h:
            new_w = target_w
            new_h = int(target_w / aspect)
        else:
            new_h = target_h
            new_w = int(target_h * aspect)
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Pad to target size
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        padded = cv2.copyMakeBorder(
            resized, pad_h, target_h - new_h - pad_h, 
            pad_w, target_w - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return padded
    else:
        return cv2.resize(frame, size)


# ==================== GEOMETRIC UTILITIES ====================

def calculate_euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(point1 - point2)


def calculate_angle(point1: np.ndarray, point2: np.ndarray, 
                   point3: np.ndarray) -> float:
    """
    Calculate angle between three points (point2 is the vertex)
    
    Returns:
        Angle in degrees
    """
    vector1 = point1 - point2
    vector2 = point3 - point2
    
    cos_angle = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6
    )
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    return np.degrees(angle)


def rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles (yaw, pitch, roll)
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        yaw, pitch, roll in degrees
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return np.degrees(z), np.degrees(y), np.degrees(x)  # yaw, pitch, roll


# ==================== STATISTICAL UTILITIES ====================

def compute_variance(values: np.ndarray, axis: Optional[int] = None) -> float:
    """Compute variance with numerical stability"""
    return np.var(values, axis=axis, ddof=1)


def compute_statistics(values: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive statistics for a sequence
    
    Returns:
        Dictionary with mean, std, min, max, median, q1, q3
    """
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values),
        'q1': np.percentile(values, 25),
        'q3': np.percentile(values, 75),
        'iqr': np.percentile(values, 75) - np.percentile(values, 25)
    }


def detect_outliers(values: np.ndarray, method: str = 'iqr', 
                   threshold: float = 1.5) -> np.ndarray:
    """
    Detect outliers in data
    
    Args:
        values: Input data
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean mask of outliers
    """
    if method == 'iqr':
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (values < lower_bound) | (values > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((values - np.mean(values)) / (np.std(values) + 1e-6))
        return z_scores > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def smooth_signal(signal: np.ndarray, window_size: int = 5, 
                 method: str = 'moving_average') -> np.ndarray:
    """
    Smooth time series signal
    
    Args:
        signal: Input signal
        window_size: Window size for smoothing
        method: 'moving_average', 'gaussian', or 'median'
        
    Returns:
        Smoothed signal
    """
    if method == 'moving_average':
        kernel = np.ones(window_size) / window_size
        return np.convolve(signal, kernel, mode='same')
    
    elif method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(signal, sigma=window_size/4)
    
    elif method == 'median':
        from scipy.signal import medfilt
        return medfilt(signal, kernel_size=window_size)
    
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


# ==================== VISUALIZATION UTILITIES ====================

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         save_path: Optional[Path] = None) -> None:
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float,
                  save_path: Optional[Path] = None) -> None:
    """
    Plot ROC curve
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: Area under curve
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(feature_names: List[str], importances: np.ndarray,
                           top_n: int = 15, save_path: Optional[Path] = None) -> None:
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importances: Feature importance scores
        top_n: Number of top features to display
        save_path: Optional path to save figure
    """
    # Sort by importance
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[Path] = None) -> None:
    """
    Plot training and validation metrics over epochs
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ==================== DATA HANDLING UTILITIES ====================

def load_dataframe(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load data from CSV or Excel file"""
    file_path = Path(file_path)
    
    if file_path.suffix == '.csv':
        return pd.read_csv(file_path)
    elif file_path.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_dataframe(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """Save dataframe to CSV or Excel"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if file_path.suffix == '.csv':
        df.to_csv(file_path, index=False)
    elif file_path.suffix in ['.xlsx', '.xls']:
        df.to_excel(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    logger.info(f"Saved dataframe to {file_path}")


def normalize_features(features: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Dict]:
    """
    Normalize features
    
    Args:
        features: Feature array
        method: 'standard', 'minmax', or 'robust'
        
    Returns:
        Normalized features and normalization parameters
    """
    if method == 'standard':
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-8
        normalized = (features - mean) / std
        params = {'mean': mean, 'std': std, 'method': 'standard'}
    
    elif method == 'minmax':
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        range_val = max_val - min_val + 1e-8
        normalized = (features - min_val) / range_val
        params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
    
    elif method == 'robust':
        median = np.median(features, axis=0)
        q1 = np.percentile(features, 25, axis=0)
        q3 = np.percentile(features, 75, axis=0)
        iqr = q3 - q1 + 1e-8
        normalized = (features - median) / iqr
        params = {'median': median, 'iqr': iqr, 'method': 'robust'}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def apply_normalization(features: np.ndarray, params: Dict) -> np.ndarray:
    """Apply previously computed normalization"""
    method = params['method']
    
    if method == 'standard':
        return (features - params['mean']) / params['std']
    elif method == 'minmax':
        return (features - params['min']) / (params['max'] - params['min'])
    elif method == 'robust':
        return (features - params['median']) / params['iqr']
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ==================== TIME UTILITIES ====================

def get_timestamp() -> str:
    """Get current timestamp as formatted string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


# ==================== FILE UTILITIES ====================

def ensure_dir(directory: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_size(file_path: Union[str, Path]) -> str:
    """Get file size in human-readable format"""
    size_bytes = Path(file_path).stat().st_size
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} TB"


def save_json(data: Dict, file_path: Union[str, Path]) -> None:
    """Save dictionary to JSON file"""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved JSON to {file_path}")


def load_json(file_path: Union[str, Path]) -> Dict:
    """Load dictionary from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


# ==================== PROGRESS UTILITIES ====================

class ProgressTracker:
    """Simple progress tracker for long-running operations"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, n: int = 1) -> None:
        """Update progress by n steps"""
        self.current += n
        percent = (self.current / self.total) * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = format_duration(eta)
        else:
            eta_str = "calculating..."
        
        print(f"\r{self.description}: {self.current}/{self.total} ({percent:.1f}%) | ETA: {eta_str}", 
              end="", flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
    print(f"Logging level: {logger.level}")
