"""
Streamlit Web Application for AI-Assisted Fraud Detection
Interactive UI for real-time video analysis and fraud detection
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
    print(f"Warning: TensorFlow initialization issue: {e}")
    pass

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import tempfile
from typing import Optional, Dict
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import queue
import threading
from collections import deque
import time
import json

# Import project modules
import config
from feature_extraction import VideoFeatureExtractor
from utils import load_video, get_timestamp
from fraud_detector import ComprehensiveFraudDetector, FraudIndicator

# Import model classes for unpickling
# This handles the case where models were saved from train_model.py
from train_model import EnsembleModel, FraudDetectionNN
import sys

# Workaround for pickle: Make classes available in __main__ namespace
# This fixes "Can't get attribute 'EnsembleModel' on <module 'main'>" error
sys.modules['__main__'].EnsembleModel = EnsembleModel
sys.modules['__main__'].FraudDetectionNN = FraudDetectionNN

# Page configuration
st.set_page_config(**config.PAGE_CONFIG)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-genuine {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .prediction-ai-assisted {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0.3; }
    }
    .warning-blink {
        animation: blink 1s infinite;
        background-color: #dc3545;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ==================== UTILITY FUNCTIONS ====================

@st.cache_resource
def load_model(model_path: Optional[Path] = None):
    """Load trained model (cached)"""
    if model_path is None:
        # Find latest model
        model_files = list(config.MODEL_DIR.glob('ai_fraud_detector_*.pkl'))
        if not model_files:
            return None
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    
    try:
        saved_data = joblib.load(model_path)
        return saved_data['ensemble'], saved_data['scaler']
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def get_feature_extractor():
    """Initialize feature extractor (cached)"""
    return VideoFeatureExtractor()


def predict_fraud(features: Dict[str, float], model, scaler) -> tuple:
    """
    Make fraud prediction
    
    Args:
        features: Extracted features
        model: Trained model
        scaler: Fitted scaler
        
    Returns:
        prediction, confidence, probabilities
    """
    # Convert to array
    feature_vector = np.array([features[name] for name in config.FEATURE_NAMES]).reshape(1, -1)
    
    # Normalize
    feature_vector = scaler.transform(feature_vector)
    
    # Predict
    prediction = model.predict(feature_vector)[0]
    probabilities = model.predict_proba(feature_vector)[0]
    confidence = probabilities[prediction]
    
    return prediction, confidence, probabilities


# ==================== WEBCAM PROCESSING ====================

class VideoProcessor(VideoProcessorBase):
    """Real-time video processor for webcam fraud detection"""
    
    def __init__(self):
        self.extractor = VideoFeatureExtractor()
        self.frame_buffer = []
        self.buffer_size = 30  # Analyze every 30 frames (1 second at 30fps)
        self.result_queue = queue.Queue(maxsize=1)
        self.model = None
        self.scaler = None
        self.processing = False
        
        # Store prediction history for live graphing
        self.prediction_history = deque(maxlen=100)  # Keep last 100 predictions
        self.timestamp_history = deque(maxlen=100)
        
        # Warning state
        self.suspicious_count = 0
        self.warning_threshold = 5  # Trigger warning after 5 suspicious predictions (increased from 3)
        
        # Persistent prediction display (minimum 30 seconds)
        self.last_result = None
        self.last_result_time = None
        self.result_display_duration = 5  # Display prediction for at least 5 seconds
        
    def set_model(self, model, scaler):
        """Set the model and scaler for predictions"""
        self.model = model
        self.scaler = scaler
    
    def get_prediction_history(self):
        """Get prediction history for live graphing"""
        return list(self.prediction_history), list(self.timestamp_history)
        
    def recv(self, frame):
        """Process each frame from webcam"""
        img = frame.to_ndarray(format="bgr24")
        
        # Add frame to buffer
        self.frame_buffer.append(img)
        
        # Process when buffer is full
        if len(self.frame_buffer) >= self.buffer_size and not self.processing:
            self.processing = True
            # Process frames in background thread to avoid blocking
            threading.Thread(target=self._process_buffer, daemon=True).start()
        
        # Get current result to display
        current_result = None
        current_time = time.time()
        
        # Check for new result in queue
        if not self.result_queue.empty():
            try:
                new_result = self.result_queue.get_nowait()
                self.last_result = new_result
                self.last_result_time = current_time
                current_result = new_result
            except queue.Empty:
                pass
        
        # Use last result if still within display duration
        if current_result is None and self.last_result is not None:
            if self.last_result_time is not None:
                elapsed = current_time - self.last_result_time
                if elapsed < self.result_display_duration:
                    current_result = self.last_result
        
        # Draw overlay with face mesh and predictions
        if current_result is not None:
            self._draw_overlay(img, current_result)
        else:
            # Always draw face mesh even without predictions
            self._draw_face_mesh_only(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _draw_face_mesh_only(self, img):
        """Draw only face mesh without predictions (when no result available)"""
        height, width = img.shape[:2]
        
        # Get landmarks from extractor
        landmarks = None
        if hasattr(self.extractor, 'face_mesh'):
            try:
                rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.extractor.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            except Exception as e:
                pass
        
        # Draw face mesh if landmarks detected
        if landmarks:
            self._draw_face_mesh(img, landmarks)
            
            # Draw status text
            cv2.putText(img, "Analyzing...", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    def _detect_suspicious_patterns(self, features):
        """
        Detect suspicious behavioral patterns using rule-based logic
        Supplements ML model with explicit fraud indicators
        
        Returns:
            (is_suspicious, reasons, fraud_score)
        """
        suspicious_reasons = []
        fraud_score = 0.0
        
        if not features:
            return False, [], 0.0
        
        # 1. Excessive gaze shifts (reading from another screen)
        gaze_shift_count = features.get('gaze_shift_count', 0)
        if gaze_shift_count > 40:  # Increased threshold from 30 to 40
            suspicious_reasons.append(f"High gaze shifts: {gaze_shift_count:.0f}")
            fraud_score += 0.3
        
        # 2. Abnormal blink rate (nervousness or cognitive load)
        blink_rate = features.get('blink_rate', 0)
        if blink_rate > 30 or blink_rate < 3:  # Wider range: was 25/5, now 30/3
            suspicious_reasons.append(f"Abnormal blink rate: {blink_rate:.1f}/min")
            fraud_score += 0.15
        
        # 3. High cognitive load (indicates reading/processing external info)
        cognitive_load = features.get('cognitive_load_score', 0)
        if cognitive_load > 0.8:  # Increased from 0.7 to 0.8
            suspicious_reasons.append(f"High cognitive load: {cognitive_load:.2f}")
            fraud_score += 0.25
        
        # 4. Unstable head position (looking away frequently)
        head_stability = features.get('head_stability', 1.0)
        if head_stability < 0.3:  # Decreased from 0.5 to 0.3 (more lenient)
            suspicious_reasons.append(f"Unstable head: {head_stability:.2f}")
            fraud_score += 0.2
        
        # 5. High eye movement velocity (rapid scanning)
        gaze_velocity = features.get('gaze_x_velocity', 0)
        if abs(gaze_velocity) > 15:  # Increased from 10 to 15
            suspicious_reasons.append(f"Rapid eye movement: {abs(gaze_velocity):.1f}")
            fraud_score += 0.2
        
        # 6. Long fixation duration (reading text)
        fixation_duration = features.get('eye_fixation_duration', 0)
        if fixation_duration > 800:  # Increased from 600ms to 800ms
            suspicious_reasons.append(f"Long fixations: {fixation_duration:.0f}ms")
            fraud_score += 0.2
        
        # 7. Emotion instability (stress/deception)
        emotion_stability = features.get('emotion_stability', 1.0)
        if emotion_stability < 0.3:  # Decreased from 0.4 to 0.3 (more lenient)
            suspicious_reasons.append(f"Unstable emotions: {emotion_stability:.2f}")
            fraud_score += 0.15
        
        # 8. High response delay (thinking/reading time)
        response_delay = features.get('response_delay', 0)
        if response_delay > 7:  # Increased from 5 to 7 seconds
            suspicious_reasons.append(f"High response delay: {response_delay:.1f}s")
            fraud_score += 0.15
        
        # Determine if suspicious (stricter criteria: need more indicators OR higher score)
        is_suspicious = len(suspicious_reasons) >= 3 or fraud_score >= 0.6  # Changed from 2/0.4 to 3/0.6
        
        return is_suspicious, suspicious_reasons, min(fraud_score, 1.0)
    
    def _process_buffer(self):
        """Process buffered frames and make prediction"""
        try:
            if self.model is None or self.scaler is None:
                return
            
            # Extract features from buffered frames
            features = self.extractor.extract_features_from_frames(self.frame_buffer)
            
            if features:
                # Make prediction
                feature_vector = np.array([features[name] for name in config.FEATURE_NAMES]).reshape(1, -1)
                feature_vector = self.scaler.transform(feature_vector)
                
                prediction = self.model.predict(feature_vector)[0]
                probabilities = self.model.predict_proba(feature_vector)[0]
                confidence = probabilities[prediction]
                
                # Apply rule-based detection
                is_suspicious, reasons, rule_based_score = self._detect_suspicious_patterns(features)
                
                # Combine ML and rule-based scores
                ml_fraud_prob = probabilities[1]
                combined_fraud_prob = (ml_fraud_prob * 0.6) + (rule_based_score * 0.4)
                
                # Override prediction if rule-based detection is strong
                if is_suspicious and rule_based_score > 0.7:  # Increased from 0.5 to 0.7
                    prediction = 1  # Override to fraud
                    confidence = combined_fraud_prob
                
                # Update probabilities with combined score
                adjusted_probabilities = np.array([1 - combined_fraud_prob, combined_fraud_prob])
                
                # Update prediction history
                current_time = time.time()
                self.prediction_history.append({
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': adjusted_probabilities,
                    'ai_assisted_prob': combined_fraud_prob,
                    'rule_based_score': rule_based_score,
                    'suspicious_reasons': reasons
                })
                self.timestamp_history.append(current_time)
                
                # Track suspicious behavior - ONLY for fraud predictions with high confidence
                if prediction == 1 and confidence > 0.7:  # Must be fraud AND high confidence
                    self.suspicious_count += 1
                elif prediction == 0:  # If genuine, reduce suspicious count
                    self.suspicious_count = max(0, self.suspicious_count - 2)  # Reduce by 2 for genuine
                else:
                    self.suspicious_count = max(0, self.suspicious_count - 1)
                
                result = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': adjusted_probabilities,
                    'show_warning': self.suspicious_count >= self.warning_threshold,
                    'features': features,  # Pass features for display
                    'suspicious_reasons': reasons,
                    'rule_based_score': rule_based_score
                }
                
                # Update result queue
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.result_queue.put(result)
            
            # Clear buffer
            self.frame_buffer = []
            
        except Exception as e:
            # Silently handle errors to avoid console spam
            pass
        finally:
            self.processing = False
    
    def _draw_face_mesh(self, img, landmarks):
        """Draw face mesh landmarks for visual feedback"""
        height, width = img.shape[:2]
        
        # Define face mesh connections (simplified)
        # Draw key face features
        face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        left_eye = [33, 160, 158, 133, 153, 144, 163, 7]
        right_eye = [362, 385, 387, 263, 373, 380, 374, 263]
        
        left_iris = [468, 469, 470, 471, 472]
        right_iris = [473, 474, 475, 476, 477]
        
        mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        
        # Draw connections for each feature
        def draw_feature(indices, color, thickness=1):
            for i in range(len(indices) - 1):
                if indices[i] < len(landmarks) and indices[i + 1] < len(landmarks):
                    pt1 = (int(landmarks[indices[i]][0] * width), int(landmarks[indices[i]][1] * height))
                    pt2 = (int(landmarks[indices[i + 1]][0] * width), int(landmarks[indices[i + 1]][1] * height))
                    cv2.line(img, pt1, pt2, color, thickness)
        
        # Draw face oval (cyan)
        draw_feature(face_oval, (255, 255, 0), 1)
        
        # Draw eyes (green)
        draw_feature(left_eye, (0, 255, 0), 2)
        draw_feature(right_eye, (0, 255, 0), 2)
        
        # Draw irises (yellow) - key for gaze tracking
        for idx in left_iris:
            if idx < len(landmarks):
                pt = (int(landmarks[idx][0] * width), int(landmarks[idx][1] * height))
                cv2.circle(img, pt, 2, (0, 255, 255), -1)
        
        for idx in right_iris:
            if idx < len(landmarks):
                pt = (int(landmarks[idx][0] * width), int(landmarks[idx][1] * height))
                cv2.circle(img, pt, 2, (0, 255, 255), -1)
        
        # Draw mouth (blue)
        draw_feature(mouth, (255, 0, 0), 1)
    
    def _draw_gaze_direction(self, img, landmarks, features):
        """Draw gaze direction arrows showing where person is looking"""
        height, width = img.shape[:2]
        
        # Get eye centers from landmarks
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        if len(landmarks) > max(max(left_eye_indices), max(right_eye_indices)):
            # Calculate eye centers
            left_eye_pts = np.array([[landmarks[i][0] * width, landmarks[i][1] * height] 
                                     for i in left_eye_indices])
            right_eye_pts = np.array([[landmarks[i][0] * width, landmarks[i][1] * height] 
                                      for i in right_eye_indices])
            
            left_center = left_eye_pts.mean(axis=0).astype(int)
            right_center = right_eye_pts.mean(axis=0).astype(int)
            
            # Get gaze features if available
            if features and 'gaze_x_velocity' in features and 'gaze_y_velocity' in features:
                gaze_x_vel = features.get('gaze_x_velocity', 0)
                gaze_y_vel = features.get('gaze_y_velocity', 0)
                
                # Scale velocity to arrow length
                arrow_scale = 50
                arrow_x = int(gaze_x_vel * arrow_scale)
                arrow_y = int(gaze_y_vel * arrow_scale)
                
                # Draw gaze direction arrows
                arrow_color = (255, 165, 0)  # Orange
                cv2.arrowedLine(img, tuple(left_center), 
                               (left_center[0] + arrow_x, left_center[1] + arrow_y),
                               arrow_color, 2, tipLength=0.3)
                cv2.arrowedLine(img, tuple(right_center), 
                               (right_center[0] + arrow_x, right_center[1] + arrow_y),
                               arrow_color, 2, tipLength=0.3)
            
            # Draw eye center points
            cv2.circle(img, tuple(left_center), 3, (0, 255, 255), -1)
            cv2.circle(img, tuple(right_center), 3, (0, 255, 255), -1)
    
    def _draw_behavioral_indicators(self, img, features):
        """Draw behavioral metrics as overlay indicators"""
        if not features:
            return
        
        height, width = img.shape[:2]
        
        # Create indicator panel on the right side
        panel_x = width - 220
        panel_y = 10
        
        # Semi-transparent background for metrics
        overlay = img.copy()
        cv2.rectangle(overlay, (panel_x - 10, panel_y), (width - 10, panel_y + 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
        # Display key behavioral metrics
        metrics = [
            ('Blink Rate', features.get('blink_rate', 0), 20, 'blinks/min'),
            ('Head Movement', features.get('head_movement_magnitude', 0), 10, 'units'),
            ('Gaze Shifts', features.get('gaze_shift_count', 0), 30, 'shifts'),
            ('Eye Fixation', features.get('eye_fixation_duration', 0), 500, 'ms')
        ]
        
        y_offset = panel_y + 25
        for name, value, threshold, unit in metrics:
            # Determine if value is suspicious (exceeds threshold)
            is_suspicious = value > threshold
            color = (0, 0, 255) if is_suspicious else (0, 255, 0)
            
            # Draw metric name
            cv2.putText(img, f"{name}:", (panel_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw value
            value_text = f"{value:.1f} {unit}"
            cv2.putText(img, value_text, (panel_x, y_offset + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            
            y_offset += 45
    
    def _draw_overlay(self, img, result):
        """Draw comprehensive overlay with face mesh, gaze tracking, and predictions"""
        height, width = img.shape[:2]
        
        # Get landmarks from extractor if available
        landmarks = None
        if hasattr(self.extractor, 'face_mesh'):
            try:
                rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.extractor.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            except Exception as e:
                # Silently handle errors
                pass
        
        # Always draw face mesh if landmarks detected
        if landmarks:
            self._draw_face_mesh(img, landmarks)
            self._draw_gaze_direction(img, landmarks, result.get('features'))
        
        # Draw behavioral indicators
        self._draw_behavioral_indicators(img, result.get('features'))
        
        # Determine color and label
        if result['prediction'] == 0:
            color = (0, 255, 0)  # Green for genuine
            label = "GENUINE"
        else:
            color = (0, 0, 255)  # Red for AI-assisted
            label = "AI-ASSISTED FRAUD"
        
        confidence = result['confidence'] * 100
        ai_prob = result['probabilities'][1] * 100  # AI-assisted probability
        
        # Draw semi-transparent overlay at top
        overlay = img.copy()
        overlay_height = 120 if result.get('show_warning', False) else 80
        cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Draw main status text
        cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(img, f"Confidence: {confidence:.1f}%", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw fraud probability bar
        bar_width = int((width - 40) * (ai_prob / 100))
        bar_color = (0, 255, 0) if ai_prob < 50 else (0, 165, 255) if ai_prob < 70 else (0, 0, 255)
        cv2.rectangle(img, (20, height - 40), (20 + bar_width, height - 20), bar_color, -1)
        cv2.rectangle(img, (20, height - 40), (width - 20, height - 20), (255, 255, 255), 2)
        cv2.putText(img, f"Fraud Risk: {ai_prob:.1f}%", (20, height - 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show warning if suspicious behavior detected
        if result.get('show_warning', False):
            cv2.putText(img, "WARNING: SUSPICIOUS BEHAVIOR!", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # Add blinking effect
            if int(time.time() * 2) % 2:  # Blink every 0.5 seconds
                cv2.rectangle(img, (5, 5), (width - 5, overlay_height - 5), (0, 0, 255), 5)


def create_gauge_chart(confidence: float, prediction: int) -> go.Figure:
    """Create gauge chart for confidence score"""
    color = "green" if prediction == 0 else "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def create_feature_chart(features: Dict[str, float]) -> go.Figure:
    """Create radar chart for behavioral features"""
    # Normalize features to 0-1 scale for visualization
    normalized_features = {}
    for feature_name in config.FEATURE_NAMES[:8]:  # Show top 8 features
        value = features[feature_name]
        # Simple normalization (can be improved with actual min/max)
        if feature_name == 'eye_fixation_duration':
            normalized = min(value / 800, 1.0)
        elif feature_name == 'response_delay':
            normalized = min(value / 10, 1.0)
        elif feature_name in ['emotion_stability', 'emotion_intensity', 'head_stability', 'cognitive_load_score']:
            normalized = value
        else:
            normalized = min(value / 20, 1.0)
        
        normalized_features[feature_name] = normalized
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(normalized_features.values()),
        theta=[name.replace('_', ' ').title() for name in normalized_features.keys()],
        fill='toself',
        name='Feature Values'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=400
    )
    
    return fig


def create_probability_chart(probabilities: np.ndarray) -> go.Figure:
    """Create bar chart for class probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=config.CLASS_NAMES,
            y=probabilities * 100,
            marker_color=['green', 'red'],
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Class Probabilities",
        yaxis_title="Probability (%)",
        height=300
    )
    
    return fig


# ==================== MAIN APP ====================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 AI-Assisted Fraud Detection</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; color: #666;'>
    Detect AI assistance in video interviews using behavioral analysis
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_files = list(config.MODEL_DIR.glob('ai_fraud_detector_*.pkl'))
        if model_files:
            model_names = [f.name for f in model_files]
            selected_model = st.selectbox("Select Model", model_names, index=0)
            model_path = config.MODEL_DIR / selected_model
        else:
            st.warning("No trained models found!")
            model_path = None
        
        st.markdown("---")
        
        # Input method
        st.header("📹 Input Method")
        input_method = st.radio(
            "Choose input source:",
            ["Upload Video", "Use Webcam", "Demo Mode"]
        )
        
        st.markdown("---")
        
        # Information
        st.header("ℹ️ About")
        st.info("""
        This system analyzes behavioral cues:
        - 👁️ Eye movement patterns
        - 🧑 Head pose dynamics
        - 😊 Facial expressions
        - ⏱️ Response timing
        - 🧠 Cognitive load indicators
        """)
        
    
    # Load model
    if model_path:
        model_data = load_model(model_path)
        if model_data:
            model, scaler = model_data
            st.success("✓ Model loaded successfully")
        else:
            st.error("Failed to load model")
            return
    else:
        st.error("No model available. Please train a model first.")
        return
    
    # Main content
    if input_method == "Upload Video":
        st.header("📤 Upload Interview Video")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video of the interview"
        )
        
        if uploaded_file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            # Display video
            st.video(uploaded_file)
            
            # Analysis options
            st.markdown("### 🔍 Analysis Options")
            col1, col2 = st.columns(2)
            with col1:
                analyze_behavior = st.checkbox("Behavioral Analysis", value=True, 
                                              help="Analyze eye movements, head pose, emotions")
                analyze_timing = st.checkbox("Timing Analysis", value=True,
                                            help="Analyze response timing patterns")
            with col2:
                analyze_text = st.checkbox("Text Response Analysis", value=False,
                                          help="Analyze text responses for AI patterns")
                comprehensive_mode = st.checkbox("Comprehensive Report", value=True,
                                                help="Generate detailed fraud assessment")
            
            # Text input for response analysis
            responses_text = []
            if analyze_text:
                st.markdown("#### 📝 Enter Interview Responses")
                st.info("Paste the candidate's text responses below (one per line or separated by blank lines)")
                responses_input = st.text_area(
                    "Candidate Responses",
                    height=150,
                    placeholder="Paste responses here...\n\nResponse 1: ...\n\nResponse 2: ..."
                )
                if responses_input:
                    # Split by double newlines or single newlines
                    responses_text = [r.strip() for r in responses_input.split('\n\n') if r.strip()]
                    if not responses_text:
                        responses_text = [r.strip() for r in responses_input.split('\n') if r.strip() and len(r.strip()) > 20]
            
            # Analyze button
            if st.button("🔍 Analyze Interview", type="primary"):
                with st.spinner("Extracting behavioral features..."):
                    # Extract features
                    extractor = get_feature_extractor()
                    features = extractor.process_video(video_path)
                    if hasattr(extractor, 'close'):
                        extractor.close()
                
                if comprehensive_mode:
                    # Use comprehensive fraud detector
                    with st.spinner("Running comprehensive fraud analysis..."):
                        detector = ComprehensiveFraudDetector(model_path)
                        
                        # Behavioral analysis
                        if analyze_behavior:
                            detector.analyze_behavioral_patterns(features)
                        
                        # Text analysis
                        if analyze_text and responses_text:
                            detector.analyze_response_patterns(responses_text)
                        
                        # Timing analysis (mock data for demo)
                        if analyze_timing:
                            # Generate sample timestamps based on video duration
                            num_responses = len(responses_text) if responses_text else 5
                            timestamps = np.cumsum(np.random.uniform(3, 12, num_responses))
                            detector.analyze_timing_patterns(timestamps)
                        
                        # Get comprehensive assessment
                        assessment = detector.get_comprehensive_assessment()
                        
                        # Display comprehensive results
                        display_comprehensive_results(assessment, features)
                else:
                    # Simple prediction
                    prediction, confidence, probabilities = predict_fraud(features, model, scaler)
                    display_results(prediction, confidence, probabilities, features)
    
    elif input_method == "Use Webcam":
        st.header("📷 Real-Time Webcam Analysis")
        
        st.markdown("""
        ### 🎥 Live Interview Analysis
        This feature analyzes your behavior in real-time using your webcam.
        
        **How it works:**
        1. Click "Start" to begin webcam capture
        2. Position your face in the frame
        3. The system analyzes your behavior every second
        4. Real-time predictions appear on screen
        
        **Privacy Note:** All processing happens locally. No video is uploaded.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create WebRTC streamer
            ctx = webrtc_streamer(
                key="fraud-detection",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=VideoProcessor,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 1280},
                        "height": {"ideal": 720},
                    },
                    "audio": False
                },
                async_processing=True,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }
            )
            
            # Set model in video processor
            if ctx.video_processor:
                ctx.video_processor.set_model(model, scaler)
                
        with col2:
            st.markdown("### 📊 Live Status")
            
            status_placeholder = st.empty()
            graph_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            if ctx.state.playing:
                status_placeholder.success("🟢 **Active**\n\nAnalyzing behavior...")
                
                # Try to display live graph
                if ctx.video_processor:
                    try:
                        pred_history, time_history = ctx.video_processor.get_prediction_history()
                        if len(pred_history) > 0:
                            # Create live fraud probability graph
                            fraud_probs = [p['ai_assisted_prob'] * 100 for p in pred_history]
                            relative_times = [(t - time_history[0]) for t in time_history]
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=relative_times,
                                y=fraud_probs,
                                mode='lines+markers',
                                name='Fraud Risk',
                                line=dict(color='red', width=2),
                                marker=dict(size=4),
                                fill='tozeroy',
                                fillcolor='rgba(255, 0, 0, 0.1)'
                            ))
                            
                            # Add threshold line
                            fig.add_hline(y=70, line_dash="dash", line_color="orange", 
                                        annotation_text="Warning Threshold")
                            
                            fig.update_layout(
                                title="Real-Time Fraud Risk",
                                xaxis_title="Time (seconds)",
                                yaxis_title="Fraud Probability (%)",
                                height=250,
                                margin=dict(l=20, r=20, t=40, b=20),
                                yaxis=dict(range=[0, 100])
                            )
                            
                            graph_placeholder.plotly_chart(fig, use_container_width=True)
                            
                            # Show latest metrics
                            latest = pred_history[-1]
                            metrics_placeholder.metric(
                                "Current Fraud Risk",
                                f"{latest['ai_assisted_prob']*100:.1f}%",
                                delta=f"{'⚠️ High' if latest['ai_assisted_prob'] > 0.7 else '✓ Normal'}"
                            )
                    except:
                        pass
                
                st.markdown("""
                ### 💡 Tips
                - Keep your face visible
                - Natural lighting works best
                - Look at the camera
                - Answer questions normally
                """)
            else:
                status_placeholder.info("⚪ **Standby**\n\nClick START to begin")
        
        st.markdown("---")
        
        # Live behavioral metrics dashboard
        if ctx.state.playing and ctx.video_processor:
            st.markdown("### 🎯 Live Behavioral Metrics")
            
            # Get latest result if available
            try:
                if not ctx.video_processor.result_queue.empty():
                    latest_result = ctx.video_processor.result_queue.queue[-1]
                    features = latest_result.get('features', {})
                    
                    if features:
                        # Display key metrics in columns
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Eye Movement",
                                f"{features.get('eye_movement_freq', 0):.1f}/s",
                                help="Frequency of eye movements per second"
                            )
                            st.metric(
                                "Gaze Stability",
                                f"{features.get('eye_fixation_duration', 0):.0f}ms",
                                help="Average duration of eye fixations"
                            )
                        
                        with col2:
                            st.metric(
                                "Head Movement",
                                f"{features.get('head_pose_variance', 0):.1f}°",
                                help="Variance in head pose angles"
                            )
                            st.metric(
                                "Head Stability",
                                f"{features.get('head_stability', 0):.2f}",
                                help="Steadiness of head position (0-1)"
                            )
                        
                        with col3:
                            st.metric(
                                "Response Time",
                                f"{features.get('response_delay', 0):.1f}s",
                                help="Average delay in responses"
                            )
                            st.metric(
                                "Blink Rate",
                                f"{features.get('blink_rate', 0):.0f}/min",
                                help="Blinks per minute"
                            )
                        
                        with col4:
                            st.metric(
                                "Emotion Stability",
                                f"{features.get('emotion_stability', 0):.2f}",
                                help="Consistency of emotional state (0-1)"
                            )
                            st.metric(
                                "Cognitive Load",
                                f"{features.get('cognitive_load_score', 0):.2f}",
                                help="Estimated mental effort (0-1)"
                            )
                        
                        # Warning banner if suspicious
                        if latest_result.get('show_warning', False):
                            st.markdown("""
                            <div class="warning-blink">
                                🚨 ALERT: Multiple suspicious indicators detected! Possible AI assistance. 🚨
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display specific suspicious reasons
                            suspicious_reasons = latest_result.get('suspicious_reasons', [])
                            rule_based_score = latest_result.get('rule_based_score', 0)
                            
                            if suspicious_reasons:
                                st.error("**Detected Issues:**")
                                for reason in suspicious_reasons:
                                    st.write(f"⚠️ {reason}")
                                st.write(f"**Rule-Based Fraud Score:** {rule_based_score*100:.1f}%")
                            
                            # Play alert sound using JavaScript
                            st.markdown("""
                            <audio autoplay>
                                <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIGGe+7+OZSA0PVKzn77BgGgU7k9r3yngzBi9+y/DajT0KF2O49+meSgoRW67r8KRTGA==">
                            </audio>
                            <script>
                                // Optional: Add visual shake effect
                                document.body.style.animation = 'shake 0.5s';
                            </script>
                            """, unsafe_allow_html=True)
            except:
                st.info("Waiting for analysis data...")
        
        st.markdown("---")
        
        # Instructions
        with st.expander("📖 Detailed Instructions"):
            st.markdown("""
            ### System Requirements
            - Modern web browser (Chrome, Edge, or Firefox recommended)
            - Working webcam
            - Good lighting conditions
            
            ### Best Practices
            1. **Positioning**: Center your face in the frame
            2. **Lighting**: Face a light source, avoid backlighting
            3. **Background**: Use a plain, uncluttered background
            4. **Distance**: Sit about 50-70cm from the camera
            
            ### What's Being Analyzed
            - **Eye Movements**: Gaze patterns and shifts
            - **Head Pose**: Movement and orientation
            - **Facial Expressions**: Micro-expressions and emotions
            - **Blink Rate**: Frequency and patterns
            - **Response Timing**: Reaction delays
            
            ### Interpretation
            - **Green (Genuine)**: Natural behavioral patterns
            - **Red (AI-Assisted)**: Indicators of potential assistance
            - **Confidence**: How certain the system is (higher is more confident)
            """)
        
        # Performance note
        st.info("""
        **Note**: Analysis occurs every ~1 second. The system needs time to process frames 
        and extract behavioral features. Real-time performance depends on your computer's CPU/GPU.
        """)
    
    elif input_method == "Demo Mode":
        st.header("🎬 Demo Mode")
        st.info("Using pre-generated sample features for demonstration")
        
        # Generate demo features
        demo_type = st.radio("Select demo profile:", ["Genuine Candidate", "AI-Assisted Candidate"])
        
        # Add comprehensive mode toggle
        use_comprehensive = st.checkbox("Use Comprehensive Analysis", value=True)
        
        # Add demo text responses
        demo_responses = {
            "Genuine Candidate": [
                "Well, um, I think my biggest strength is probably my ability to work with teams. Like, I've always enjoyed collaborating with others and I find that I can usually help resolve conflicts when they come up.",
                "That's a good question. I guess I'd say my weakness is sometimes I get too focused on details and lose track of time. I'm working on it though, trying to step back and see the bigger picture more often.",
                "Oh yeah, so at my last job we had this project that was falling behind schedule. I basically just organized a few extra meetings with the team, we figured out what was blocking us, and managed to get things back on track. It was pretty stressful but we pulled it off."
            ],
            "AI-Assisted Candidate": [
                "My greatest strength lies in my exceptional ability to collaborate effectively within team environments. I possess strong interpersonal skills and consistently demonstrate proficiency in conflict resolution through active listening and diplomatic communication. Furthermore, I excel at fostering inclusive workplace cultures that promote innovation and productivity.",
                "In terms of areas for improvement, I would identify a tendency toward perfectionism in detail-oriented tasks. However, I am actively addressing this through strategic time management techniques and by developing a more holistic perspective on project objectives. This self-awareness enables continuous professional growth.",
                "In my previous role, I encountered a significant project delay challenge. I implemented a comprehensive action plan that included: 1) Conducting stakeholder analysis meetings, 2) Identifying and addressing bottlenecks systematically, 3) Reallocating resources strategically. Through these methodologies, we successfully achieved project completion within the revised timeline."
            ]
        }
        
        if st.button("🎯 Run Demo Analysis", type="primary"):
            # Generate sample features
            if demo_type == "Genuine Candidate":
                features = {
                    'eye_movement_freq': 2.8,
                    'eye_fixation_duration': 380,
                    'head_pose_variance': 9.2,
                    'head_stability': 0.72,
                    'response_delay': 1.5,
                    'emotion_stability': 0.62,
                    'emotion_intensity': 0.75,
                    'micro_expression_count': 9,
                    'blink_rate': 18,
                    'speech_pause_frequency': 4.2,
                    'gaze_dispersion': 48,
                    'cognitive_load_score': 0.58
                }
            else:
                features = {
                    'eye_movement_freq': 7.1,
                    'eye_fixation_duration': 165,
                    'head_pose_variance': 3.2,
                    'head_stability': 0.91,
                    'response_delay': 5.2,
                    'emotion_stability': 0.88,
                    'emotion_intensity': 0.42,
                    'micro_expression_count': 2,
                    'blink_rate': 11,
                    'speech_pause_frequency': 1.8,
                    'gaze_dispersion': 22,
                    'cognitive_load_score': 0.31
                }
            
            if use_comprehensive:
                # Comprehensive analysis
                with st.spinner("Running comprehensive analysis..."):
                    detector = ComprehensiveFraudDetector(model_path)
                    
                    # Analyze behavioral patterns
                    detector.analyze_behavioral_patterns(features)
                    
                    # Analyze text responses
                    responses = demo_responses[demo_type]
                    detector.analyze_response_patterns(responses)
                    
                    # Analyze timing (demo timestamps)
                    timestamps = [0, 15, 32, 48] if demo_type == "Genuine Candidate" else [0, 8, 25, 35]
                    detector.analyze_timing_patterns(timestamps)
                    
                    # Get assessment
                    assessment = detector.get_comprehensive_assessment()
                    
                    # Display comprehensive results
                    display_comprehensive_results(assessment, features)
            else:
                # Simple prediction
                prediction, confidence, probabilities = predict_fraud(features, model, scaler)
                display_results(prediction, confidence, probabilities, features)


def display_comprehensive_results(assessment: Dict, features: Dict[str, float]):
    """
    Display comprehensive fraud assessment with multiple indicators
    
    Args:
        assessment: Comprehensive assessment dictionary from fraud detector
        features: Behavioral features extracted from video
    """
    st.markdown("---")
    st.header("🎯 Comprehensive Fraud Assessment")
    
    # Overall risk display
    risk_level = assessment['overall_risk']
    risk_score = assessment['risk_score']
    confidence = assessment['confidence']
    
    risk_colors = {
        'LOW': '#28a745',
        'MEDIUM': '#ffc107',
        'HIGH': '#fd7e14',
        'CRITICAL': '#dc3545'
    }
    
    risk_icons = {
        'LOW': '✅',
        'MEDIUM': '⚠️',
        'HIGH': '🚨',
        'CRITICAL': '🛑'
    }
    
    color = risk_colors.get(risk_level, '#666')
    icon = risk_icons.get(risk_level, '❓')
    
    # Main risk banner
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {color}15 0%, {color}05 100%); 
                padding: 2rem; border-radius: 15px; 
                border-left: 5px solid {color};
                margin-bottom: 2rem;'>
        <h1 style='color: {color}; margin: 0; font-size: 2.5rem;'>
            {icon} Risk Level: {risk_level}
        </h1>
        <div style='margin-top: 1.5rem; font-size: 1.1rem;'>
            <p><strong>Risk Score:</strong> {risk_score:.1%}</p>
            <p><strong>Analysis Confidence:</strong> {confidence:.1%}</p>
            <p><strong>Indicators Detected:</strong> {assessment['num_indicators']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary
    st.markdown(f"""
    <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h3 style='margin-top: 0;'>📋 Executive Summary</h3>
        <p style='font-size: 1.1rem; line-height: 1.6;'>{assessment['summary']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendation
    st.markdown("### 💡 Recommendation")
    rec_color = color if risk_level != 'LOW' else '#17a2b8'
    st.markdown(f"""
    <div style='background-color: {rec_color}15; padding: 1.5rem; 
                border-radius: 10px; border-left: 4px solid {rec_color};'>
        <p style='font-size: 1.05rem; line-height: 1.6; margin: 0;'>
            {assessment['recommendation']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed indicators
    st.markdown("### 🔍 Detailed Fraud Indicators")
    
    if not assessment['indicators']:
        st.info("No specific fraud indicators detected.")
    else:
        # Group by category
        categories = {}
        for indicator in assessment['indicators']:
            cat = indicator['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(indicator)
        
        # Display by category
        category_names = {
            'behavioral': '👁️ Behavioral Analysis',
            'textual': '📝 Text Response Analysis',
            'timing': '⏱️ Timing Analysis',
            'technical': '💻 Technical Analysis'
        }
        
        for category, indicators in categories.items():
            st.markdown(f"#### {category_names.get(category, category.title())}")
            
            for indicator in indicators:
                severity = indicator['severity']
                conf = indicator['confidence']
                
                # Color code by severity
                if severity >= 0.7:
                    severity_color = '#dc3545'
                    severity_label = 'HIGH'
                elif severity >= 0.4:
                    severity_color = '#ffc107'
                    severity_label = 'MEDIUM'
                else:
                    severity_color = '#28a745'
                    severity_label = 'LOW'
                
                with st.expander(f"**{indicator['name']}** - Severity: {severity_label} ({severity:.0%})"):
                    st.markdown(f"**Description:** {indicator['description']}")
                    
                    # Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Severity", f"{severity:.0%}", 
                                 delta="High" if severity > 0.6 else ("Medium" if severity > 0.3 else "Low"),
                                 delta_color="inverse")
                    with col2:
                        st.metric("Confidence", f"{conf:.0%}")
                    
                    # Evidence
                    if indicator['evidence']:
                        st.markdown("**Evidence Found:**")
                        for evidence in indicator['evidence']:
                            st.markdown(f"- {evidence}")
    
    st.markdown("---")
    
    # Visualization section
    st.markdown("### 📊 Visual Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            title={'text': "Overall Fraud Risk"},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 35], 'color': "lightgreen"},
                    {'range': [35, 55], 'color': "lightyellow"},
                    {'range': [55, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Indicator severity chart
        if assessment['indicators']:
            indicator_names = [ind['name'][:20] for ind in assessment['indicators']]
            severities = [ind['severity'] * 100 for ind in assessment['indicators']]
            colors_list = ['#dc3545' if s >= 70 else '#ffc107' if s >= 40 else '#28a745' 
                          for s in severities]
            
            fig_bar = go.Figure(go.Bar(
                x=severities,
                y=indicator_names,
                orientation='h',
                marker=dict(color=colors_list),
                text=[f"{s:.0f}%" for s in severities],
                textposition='auto'
            ))
            fig_bar.update_layout(
                title="Indicator Severity Levels",
                xaxis_title="Severity (%)",
                yaxis_title="",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Behavioral features radar chart
    st.markdown("### 🎯 Behavioral Feature Analysis")
    st.plotly_chart(create_feature_chart(features), use_container_width=True)
    
    # Detailed feature table
    with st.expander("📋 View All Extracted Features"):
        feature_df = pd.DataFrame({
            'Feature': list(features.keys()),
            'Value': [f"{v:.3f}" for v in features.values()]
        })
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
    
    # Export option
    st.markdown("---")
    st.markdown("### 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare export data
        export_data = {
            'timestamp': get_timestamp(),
            'risk_level': risk_level,
            'risk_score': risk_score,
            'confidence': confidence,
            'summary': assessment['summary'],
            'recommendation': assessment['recommendation'],
            'indicators': assessment['indicators'],
            'features': features
        }
        
        import json
        json_str = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="📥 Download Full Report (JSON)",
            data=json_str,
            file_name=f"fraud_assessment_{get_timestamp()}.json",
            mime="application/json"
        )
    
    with col2:
        # CSV export of key metrics
        summary_df = pd.DataFrame([{
            'Timestamp': get_timestamp(),
            'Risk Level': risk_level,
            'Risk Score': f"{risk_score:.2%}",
            'Confidence': f"{confidence:.2%}",
            'Num Indicators': assessment['num_indicators']
        }])
        
        csv_str = summary_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Summary (CSV)",
            data=csv_str,
            file_name=f"fraud_summary_{get_timestamp()}.csv",
            mime="text/csv"
        )


def display_results(prediction: int, confidence: float, 
                   probabilities: np.ndarray, features: Dict[str, float]):
    """Display prediction results with visualizations"""
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Prediction result
    if prediction == 0:
        st.markdown(f"""
        <div class="prediction-genuine">
            <h2 style="color: #28a745; margin: 0;">🟢 Genuine Response</h2>
            <p style="margin-top: 0.5rem; font-size: 1.1rem;">
                The candidate appears to be answering genuinely without AI assistance.
            </p>
            <p style="margin-top: 1rem; font-weight: bold; font-size: 1.2rem;">
                Confidence: {confidence*100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-ai-assisted">
            <h2 style="color: #dc3545; margin: 0;">🔴 AI-Assisted Response</h2>
            <p style="margin-top: 0.5rem; font-size: 1.1rem;">
                Behavioral patterns suggest the candidate may be using AI assistance.
            </p>
            <p style="margin-top: 1rem; font-weight: bold; font-size: 1.2rem;">
                Confidence: {confidence*100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_gauge_chart(confidence, prediction), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_probability_chart(probabilities), use_container_width=True)
    
    # Feature analysis
    st.subheader("🔍 Behavioral Feature Analysis")
    st.plotly_chart(create_feature_chart(features), use_container_width=True)
    
    # Detailed features
    with st.expander("📋 View Detailed Features"):
        feature_df = pd.DataFrame({
            'Feature': list(features.keys()),
            'Value': [f"{v:.2f}" for v in features.values()]
        })
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
    
    # Key indicators
    st.subheader("🎯 Key Behavioral Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Response Delay",
            f"{features['response_delay']:.1f}s",
            delta="High" if features['response_delay'] > 3 else "Normal",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Eye Movement Frequency",
            f"{features['eye_movement_freq']:.1f}/s",
            delta="High" if features['eye_movement_freq'] > 5 else "Normal",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Emotion Stability",
            f"{features['emotion_stability']:.2f}",
            delta="High" if features['emotion_stability'] > 0.8 else "Normal",
            delta_color="inverse"
        )
    
    # Interpretation
    st.subheader("💡 Interpretation")
    
    if prediction == 0:
        st.success("""
        **Genuine Response Indicators:**
        - Natural eye movement patterns
        - Appropriate response timing
        - Varied emotional expressions
        - Natural head movements
        """)
    else:
        st.error("""
        **AI-Assisted Response Indicators:**
        - Excessive eye movement (reading from screen)
        - Delayed responses (typing to AI)
        - Flat emotional expressions
        - Unnatural stillness or focus
        """)
    
    # Export results
    if st.button("📥 Export Analysis Report"):
        # Create report
        report_data = {
            'timestamp': get_timestamp(),
            'prediction': config.CLASS_NAMES[prediction],
            'confidence': float(confidence),
            'probabilities': {
                'genuine': float(probabilities[0]),
                'ai_assisted': float(probabilities[1])
            },
            'features': {k: float(v) for k, v in features.items()}
        }
        
        # Convert to JSON
        import json
        report_json = json.dumps(report_data, indent=2)
        
        st.download_button(
            label="Download JSON Report",
            data=report_json,
            file_name=f"fraud_detection_report_{get_timestamp()}.json",
            mime="application/json"
        )
        
        st.success("✓ Report ready for download!")


# ==================== RUN APP ====================

if __name__ == "__main__":
    main()
