"""
Feature Extraction Module for AI-Assisted Fraud Detection
Extracts behavioral cues from video interviews using computer vision

Features extracted:
- Eye movement patterns and gaze tracking
- Head pose dynamics
- Facial emotion analysis
- Micro-expressions
- Blink rate and patterns
- Response timing metrics
"""

# Fix TensorFlow import issue for DeepFace compatibility
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Ensure TensorFlow is properly loaded before DeepFace
try:
    import tensorflow as tf
    # Force TensorFlow to load properly
    if not hasattr(tf, '__version__'):
        import tf_keras
        tf.__version__ = tf_keras.__version__
except Exception as e:
    pass

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from deepface import DeepFace
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from collections import deque
import warnings
import time

warnings.filterwarnings('ignore')

# Import configuration
import config
from utils import (
    load_video, sample_frames, calculate_euclidean_distance,
    calculate_angle, rotation_matrix_to_euler_angles,
    compute_variance, compute_statistics, smooth_signal
)

logger = logging.getLogger(__name__)


class VideoFeatureExtractor:
    """
    Extract behavioral features from video interviews
    Combines MediaPipe FaceMesh, DeepFace, and custom algorithms
    """
    
    def __init__(self):
        """Initialize feature extractor with MediaPipe and DeepFace"""
        try:
            # Initialize MediaPipe FaceMesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,  # Single face for better stability
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Initialize MediaPipe drawing utilities
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            # State tracking
            self.gaze_history = deque(maxlen=150)  # 5 seconds at 30fps
            self.head_pose_history = deque(maxlen=150)
            self.emotion_history = deque(maxlen=150)
            self.blink_history = deque(maxlen=150)
            
            # Blink detection state
            self.eye_aspect_ratio_threshold = 0.2
            self.consecutive_frames_for_blink = 2
            self.blink_counter = 0
            self.frame_counter = 0
            
            # Error tracking and recovery
            self.consecutive_errors = 0
            self.max_consecutive_errors = 5  # Reduced threshold
            self.failure_count = 0
            self.max_failures = 3  # Trigger reinitialization after 3 failures
            
            # Frame validation
            self.min_frame_size = (100, 100)
            self.last_successful_frame_time = None
            
            logger.info("VideoFeatureExtractor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing VideoFeatureExtractor: {e}")
            raise
    
    def _validate_frame(self, frame: np.ndarray) -> bool:
        """
        Validate frame before processing
        
        Args:
            frame: Input frame to validate
            
        Returns:
            bool: True if frame is valid, False otherwise
        """
        if frame is None or frame.size == 0:
            return False
        
        if len(frame.shape) < 2:
            return False
        
        h, w = frame.shape[:2]
        if h < self.min_frame_size[0] or w < self.min_frame_size[1]:
            logger.debug(f"Frame too small: {h}x{w}")
            return False
        
        return True
    
    def _reinitialize_mediapipe(self):
        """
        Reinitialize MediaPipe on persistent failures
        Helps recover from internal MediaPipe errors
        """
        try:
            logger.warning("Reinitializing MediaPipe due to persistent errors...")
            
            # Close existing instance
            if hasattr(self, 'face_mesh') and self.face_mesh is not None:
                try:
                    self.face_mesh.close()
                except:
                    pass
            
            # Create new instance
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Reset counters
            self.failure_count = 0
            self.consecutive_errors = 0
            
            logger.info("MediaPipe reinitialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to reinitialize MediaPipe: {e}")
            # Continue with old instance if reinitialization fails
    
    
    def process_video(self, video_path: Union[str, Path], 
                     extract_audio: bool = False) -> Dict[str, float]:
        """
        Process entire video and extract all behavioral features
        
        Args:
            video_path: Path to video file
            extract_audio: Whether to extract audio features
            
        Returns:
            Dictionary of extracted features
        """
        logger.info(f"Processing video: {video_path}")
        
        # Load video
        frames, fps, total_frames = load_video(video_path)
        logger.info(f"Video loaded: {total_frames} frames at {fps} FPS")
        
        # Reset history
        self._reset_history()
        
        # Process each frame
        frame_features = []
        for i, frame in enumerate(frames):
            if i % 30 == 0:  # Log every second
                logger.info(f"Processing frame {i}/{total_frames}")
            
            features = self._process_frame(frame, fps)
            if features is not None:
                frame_features.append(features)
        
        # Aggregate features across all frames
        aggregated_features = self._aggregate_features(frame_features, fps)
        
        # Add audio features if requested
        if extract_audio:
            audio_features = self._extract_audio_features(video_path)
            aggregated_features.update(audio_features)
        
        logger.info("Video processing complete")
        return aggregated_features
    
    
    def extract_features_from_frames(self, frames: List[np.ndarray], 
                                     fps: int = 30) -> Dict[str, float]:
        """
        Extract features from a list of frames (for webcam/real-time processing)
        
        Args:
            frames: List of BGR frames
            fps: Frame rate (default 30)
            
        Returns:
            Dictionary of aggregated features
        """
        # Reset history for new batch
        self._reset_history()
        
        # Process each frame
        frame_features = []
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features = self._process_frame(frame_rgb, fps)
            if features is not None:
                frame_features.append(features)
        
        # Aggregate features
        if frame_features:
            return self._aggregate_features(frame_features, fps)
        else:
            # Return default features if no face detected
            return {name: 0.0 for name in config.FEATURE_NAMES}
    
    
    def process_frame(self, frame: np.ndarray, fps: int = 30) -> Optional[Dict]:
        """
        Process single frame and extract features
        
        Args:
            frame: RGB frame
            fps: Frame rate
            
        Returns:
            Dictionary of per-frame features or None if face not detected
        """
        return self._process_frame(frame, fps)
    
    
    def _process_frame(self, frame: np.ndarray, fps: int) -> Optional[Dict]:
        """Internal frame processing method with robust error handling"""
        
        # Validate frame first
        if not self._validate_frame(frame):
            return None
        
        try:
            # Convert to RGB if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            
            # Make frame non-writeable for better performance
            rgb_frame = frame.copy()
            rgb_frame.flags.writeable = False
            
            # Detect face landmarks with MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                self.failure_count += 1
                if self.failure_count >= self.max_failures:
                    logger.debug(f"No face detected for {self.failure_count} consecutive frames")
                return None
            
            # Reset error counters on successful detection
            self.consecutive_errors = 0
            self.failure_count = 0
            self.last_successful_frame_time = time.time()
            
            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Convert landmarks to pixel coordinates
            landmarks_2d = []
            landmarks_3d = []
            for landmark in face_landmarks.landmark:
                x, y, z = landmark.x * w, landmark.y * h, landmark.z
                landmarks_2d.append([x, y])
                landmarks_3d.append([x, y, z])
            
            landmarks_2d = np.array(landmarks_2d)
            landmarks_3d = np.array(landmarks_3d)
        
        except AttributeError as e:
            # Handle MediaPipe internal errors (SymbolDatabase issue)
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.max_consecutive_errors:
                logger.warning(f"Persistent MediaPipe error: {e}")
                self._reinitialize_mediapipe()
            return None
            
        except Exception as e:
            # Handle other errors
            self.consecutive_errors += 1
            if self.consecutive_errors >= self.max_consecutive_errors:
                logger.error(f"Error processing frame: {e}")
                self._reinitialize_mediapipe()
            return None
        
        # Extract features
        features = {}
        
        # 1. Eye movement and gaze features
        gaze_features = self._extract_gaze_features(landmarks_2d)
        features.update(gaze_features)
        
        # 2. Head pose features
        head_pose_features = self._extract_head_pose_features(landmarks_2d, (h, w))
        features.update(head_pose_features)
        
        # 3. Blink detection
        blink_features = self._detect_blink(landmarks_2d)
        features.update(blink_features)
        
        # 4. Facial emotion (every 5 frames for efficiency)
        self.frame_counter += 1
        if self.frame_counter % 5 == 0:
            emotion_features = self._extract_emotion_features(frame)
            features.update(emotion_features)
        else:
            features['emotion'] = self.emotion_history[-1]['emotion'] if self.emotion_history else 'neutral'
            features['emotion_confidence'] = self.emotion_history[-1]['emotion_confidence'] if self.emotion_history else 0.5
        
        return features
    
    
    def _extract_gaze_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Extract eye movement and gaze direction features
        
        Uses iris landmarks and eye corners to estimate gaze
        """
        # Get eye landmarks
        left_eye = landmarks[config.LEFT_EYE_INDICES]
        right_eye = landmarks[config.RIGHT_EYE_INDICES]
        
        # Calculate eye centers
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        
        # Estimate gaze direction (simplified)
        # In production, use iris landmarks for accurate gaze tracking
        gaze_x = (left_eye_center[0] + right_eye_center[0]) / 2
        gaze_y = (left_eye_center[1] + right_eye_center[1]) / 2
        
        # Calculate inter-eye distance for normalization
        eye_distance = calculate_euclidean_distance(left_eye_center, right_eye_center)
        
        # Normalize gaze coordinates
        gaze_x_norm = gaze_x / eye_distance
        gaze_y_norm = gaze_y / eye_distance
        
        # Store in history
        self.gaze_history.append({
            'x': gaze_x_norm,
            'y': gaze_y_norm,
            'eye_distance': eye_distance
        })
        
        # Calculate gaze variance (movement frequency)
        if len(self.gaze_history) > 10:
            gaze_x_values = [g['x'] for g in list(self.gaze_history)[-30:]]
            gaze_y_values = [g['y'] for g in list(self.gaze_history)[-30:]]
            
            gaze_variance_x = compute_variance(np.array(gaze_x_values))
            gaze_variance_y = compute_variance(np.array(gaze_y_values))
            gaze_variance_total = gaze_variance_x + gaze_variance_y
        else:
            gaze_variance_total = 0.0
        
        return {
            'gaze_x': gaze_x_norm,
            'gaze_y': gaze_y_norm,
            'gaze_variance': gaze_variance_total,
            'eye_distance': eye_distance
        }
    
    
    def _extract_head_pose_features(self, landmarks: np.ndarray, 
                                    image_size: Tuple[int, int]) -> Dict[str, float]:
        """
        Extract head pose (yaw, pitch, roll) using solvePnP
        """
        h, w = image_size
        
        # 3D model points (generic face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])
        
        # 2D image points from landmarks
        image_points = np.array([
            landmarks[config.HEAD_POSE_LANDMARKS['nose_tip']],
            landmarks[config.HEAD_POSE_LANDMARKS['chin']],
            landmarks[config.HEAD_POSE_LANDMARKS['left_eye']],
            landmarks[config.HEAD_POSE_LANDMARKS['right_eye']],
            landmarks[config.HEAD_POSE_LANDMARKS['left_mouth']],
            landmarks[config.HEAD_POSE_LANDMARKS['right_mouth']]
        ], dtype=np.float64)
        
        # Camera matrix (simplified)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Assume no lens distortion
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return {
                'head_yaw': 0.0,
                'head_pitch': 0.0,
                'head_roll': 0.0,
                'head_pose_magnitude': 0.0
            }
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Extract Euler angles
        yaw, pitch, roll = rotation_matrix_to_euler_angles(rotation_matrix)
        
        # Store in history
        self.head_pose_history.append({
            'yaw': yaw,
            'pitch': pitch,
            'roll': roll
        })
        
        # Calculate head pose magnitude (how far from center)
        pose_magnitude = np.sqrt(yaw**2 + pitch**2 + roll**2)
        
        return {
            'head_yaw': yaw,
            'head_pitch': pitch,
            'head_roll': roll,
            'head_pose_magnitude': pose_magnitude
        }
    
    
    def _detect_blink(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Detect eye blinks using Eye Aspect Ratio (EAR)
        """
        # Calculate EAR for both eyes
        left_ear = self._calculate_eye_aspect_ratio(landmarks[config.LEFT_EYE_INDICES])
        right_ear = self._calculate_eye_aspect_ratio(landmarks[config.RIGHT_EYE_INDICES])
        
        # Average EAR
        ear = (left_ear + right_ear) / 2.0
        
        # Check if blink occurred
        is_blinking = ear < self.eye_aspect_ratio_threshold
        
        # Store in history
        self.blink_history.append({
            'ear': ear,
            'is_blinking': is_blinking
        })
        
        # Count blinks
        if is_blinking:
            self.blink_counter += 1
        
        return {
            'eye_aspect_ratio': ear,
            'is_blinking': 1.0 if is_blinking else 0.0
        }
    
    
    def _calculate_eye_aspect_ratio(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR)
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        # Vertical eye distances
        v1 = calculate_euclidean_distance(eye_landmarks[1], eye_landmarks[5])
        v2 = calculate_euclidean_distance(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal eye distance
        h = calculate_euclidean_distance(eye_landmarks[0], eye_landmarks[3])
        
        # EAR calculation
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        
        return ear
    
    
    def _extract_emotion_features(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Extract facial emotion using DeepFace
        """
        try:
            # Analyze frame with DeepFace
            analysis = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=config.ENFORCE_DETECTION,
                detector_backend='opencv',
                silent=True
            )
            
            # Handle both single dict and list of dicts
            if isinstance(analysis, list):
                analysis = analysis[0]
            
            # Get dominant emotion
            emotion = analysis['dominant_emotion']
            emotion_scores = analysis['emotion']
            
            # Get confidence (probability of dominant emotion)
            confidence = emotion_scores[emotion] / 100.0
            
            # Store in history
            self.emotion_history.append({
                'emotion': emotion,
                'emotion_confidence': confidence,
                'emotion_scores': emotion_scores
            })
            
            return {
                'emotion': emotion,
                'emotion_confidence': confidence
            }
        
        except Exception as e:
            # Return neutral if emotion detection fails
            return {
                'emotion': 'neutral',
                'emotion_confidence': 0.5
            }
    
    
    def _aggregate_features(self, frame_features: List[Dict], fps: int) -> Dict[str, float]:
        """
        Aggregate per-frame features into video-level features
        """
        if not frame_features:
            return self._get_default_features()
        
        # Convert list of dicts to arrays
        gaze_variance = [f['gaze_variance'] for f in frame_features if 'gaze_variance' in f]
        head_yaw = [f['head_yaw'] for f in frame_features if 'head_yaw' in f]
        head_pitch = [f['head_pitch'] for f in frame_features if 'head_pitch' in f]
        head_roll = [f['head_roll'] for f in frame_features if 'head_roll' in f]
        emotion_confidence = [f['emotion_confidence'] for f in frame_features if 'emotion_confidence' in f]
        
        # Calculate aggregate statistics
        features = {}
        
        # 1. Eye movement frequency (changes per second)
        features['eye_movement_freq'] = np.mean(gaze_variance) * fps if gaze_variance else 2.5
        
        # 2. Eye fixation duration (inverse of movement frequency)
        features['eye_fixation_duration'] = 1000.0 / (features['eye_movement_freq'] + 1e-6)
        
        # 3. Head pose variance
        if head_yaw and head_pitch and head_roll:
            head_pose_combined = np.array([head_yaw, head_pitch, head_roll]).T
            features['head_pose_variance'] = np.mean(np.var(head_pose_combined, axis=0))
        else:
            features['head_pose_variance'] = 8.0
        
        # 4. Head stability (inverse of variance)
        features['head_stability'] = 1.0 / (1.0 + features['head_pose_variance'] / 10.0)
        
        # 5. Response delay (simulated - in real scenario, measure from question timestamp)
        features['response_delay'] = np.random.uniform(0.5, 5.0)
        
        # 6. Emotion stability (variance in emotion confidence)
        if emotion_confidence:
            features['emotion_stability'] = 1.0 - np.std(emotion_confidence)
        else:
            features['emotion_stability'] = 0.65
        
        # 7. Emotion intensity (average confidence)
        features['emotion_intensity'] = np.mean(emotion_confidence) if emotion_confidence else 0.7
        
        # 8. Micro-expression count (rapid emotion changes)
        features['micro_expression_count'] = self._count_micro_expressions()
        
        # 9. Blink rate (blinks per minute)
        features['blink_rate'] = (self.blink_counter / len(frame_features)) * fps * 60 if frame_features else 17
        
        # 10. Speech pause frequency (simulated - requires audio analysis)
        features['speech_pause_frequency'] = np.random.uniform(1.0, 6.0)
        
        # 11. Gaze dispersion (spatial spread of gaze points)
        features['gaze_dispersion'] = self._calculate_gaze_dispersion()
        
        # 12. Cognitive load score (combined stress indicators)
        features['cognitive_load_score'] = self._calculate_cognitive_load(features)
        
        return features
    
    
    def _count_micro_expressions(self) -> int:
        """
        Count micro-expressions (rapid, brief emotion changes)
        """
        if len(self.emotion_history) < 10:
            return 0
        
        # Look for rapid emotion changes (< 10 frames)
        emotions = [e['emotion'] for e in self.emotion_history]
        micro_expression_count = 0
        
        i = 0
        while i < len(emotions) - 5:
            # Check if emotion changes and reverts within 5 frames
            current_emotion = emotions[i]
            next_5_frames = emotions[i+1:i+6]
            
            if current_emotion in next_5_frames:
                # Check if there was a different emotion in between
                if any(e != current_emotion for e in next_5_frames[:3]):
                    micro_expression_count += 1
                    i += 5
                    continue
            i += 1
        
        return micro_expression_count
    
    
    def _calculate_gaze_dispersion(self) -> float:
        """
        Calculate spatial spread of gaze points
        """
        if len(self.gaze_history) < 10:
            return 45.0
        
        gaze_points = np.array([[g['x'], g['y']] for g in self.gaze_history])
        
        # Calculate standard deviation of gaze points
        std_x = np.std(gaze_points[:, 0])
        std_y = np.std(gaze_points[:, 1])
        
        # Combine and scale to degrees (approximate)
        dispersion = (std_x + std_y) * 100
        
        return np.clip(dispersion, 5.0, 150.0)
    
    
    def _calculate_cognitive_load(self, features: Dict[str, float]) -> float:
        """
        Calculate cognitive load score from multiple indicators
        Combines: blink rate deviation, emotion stability, head movement
        """
        # Normal blink rate: 15-20 per minute
        blink_deviation = abs(features['blink_rate'] - 17) / 17
        
        # Emotion instability
        emotion_instability = 1.0 - features['emotion_stability']
        
        # Head movement (too still or too much = high load)
        head_variance_normalized = features['head_pose_variance'] / 20.0
        head_load = min(head_variance_normalized, 2.0 - head_variance_normalized)
        
        # Combine factors
        cognitive_load = (blink_deviation + emotion_instability + head_load) / 3.0
        
        return np.clip(cognitive_load, 0.0, 1.0)
    
    
    def _extract_audio_features(self, video_path: Union[str, Path]) -> Dict[str, float]:
        """
        Extract audio features (speech pauses, rhythm)
        
        NOTE: Audio feature extraction is disabled to reduce dependencies.
        Returns default value for compatibility with feature vector.
        """
        # Audio features disabled - return default
        return {}
    
    
    def _reset_history(self):
        """Reset all history buffers"""
        self.gaze_history.clear()
        self.head_pose_history.clear()
        self.emotion_history.clear()
        self.blink_history.clear()
        self.blink_counter = 0
        self.frame_counter = 0
    
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features if processing fails"""
        return {
            'eye_movement_freq': 2.5,
            'eye_fixation_duration': 400.0,
            'head_pose_variance': 8.0,
            'head_stability': 0.75,
            'response_delay': 1.2,
            'emotion_stability': 0.65,
            'emotion_intensity': 0.7,
            'micro_expression_count': 8,
            'blink_rate': 17,
            'speech_pause_frequency': 3.5,
            'gaze_dispersion': 45,
            'cognitive_load_score': 0.6
        }
    
    
    def get_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary to ordered numpy array
        
        Args:
            features: Dictionary of features
            
        Returns:
            Feature vector matching config.FEATURE_NAMES order
        """
        return np.array([features.get(name, 0.0) for name in config.FEATURE_NAMES])
    
    
    def close(self):
        """Clean up resources"""
        self.face_mesh.close()
        logger.info("VideoFeatureExtractor closed")


# ==================== BATCH PROCESSING ====================

def extract_features_from_video(video_path: Union[str, Path], 
                               save_path: Optional[Path] = None) -> Dict[str, float]:
    """
    Convenience function to extract features from a single video
    
    Args:
        video_path: Path to video file
        save_path: Optional path to save features as JSON
        
    Returns:
        Dictionary of extracted features
    """
    extractor = VideoFeatureExtractor()
    features = extractor.process_video(video_path)
    extractor.close()
    
    if save_path:
        import json
        with open(save_path, 'w') as f:
            json.dump(features, f, indent=2)
        logger.info(f"Features saved to {save_path}")
    
    return features


def extract_features_from_directory(directory: Union[str, Path],
                                   pattern: str = "*.mp4",
                                   output_csv: Optional[Path] = None) -> pd.DataFrame:
    """
    Extract features from all videos in a directory
    
    Args:
        directory: Directory containing videos
        pattern: File pattern to match
        output_csv: Optional path to save features as CSV
        
    Returns:
        DataFrame with features for all videos
    """
    from tqdm import tqdm
    
    directory = Path(directory)
    video_files = list(directory.glob(pattern))
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    all_features = []
    extractor = VideoFeatureExtractor()
    
    for video_file in tqdm(video_files, desc="Extracting features"):
        try:
            features = extractor.process_video(video_file)
            features['filename'] = video_file.name
            all_features.append(features)
        except Exception as e:
            logger.error(f"Error processing {video_file}: {e}")
    
    extractor.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        logger.info(f"Features saved to {output_csv}")
    
    return df


# ==================== MAIN ====================

if __name__ == "__main__":
    import sys
    
    # Demo usage
    print("=" * 60)
    print("AI-Assisted Fraud Detection - Feature Extraction Module")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"\nProcessing video: {video_path}")
        
        features = extract_features_from_video(video_path)
        
        print("\n" + "=" * 60)
        print("EXTRACTED FEATURES:")
        print("=" * 60)
        for name, value in features.items():
            print(f"{name:30s}: {value:.4f}")
    else:
        print("\nNo video provided. Initializing extractor for testing...")
        extractor = VideoFeatureExtractor()
        print("✓ Feature extractor initialized successfully")
        print("\nUsage:")
        print("  python feature_extraction.py <video_path>")
        extractor.close()
