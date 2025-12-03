"""
Test Script for Visual Tracking Features
Validates the new face mesh, gaze tracking, and detection improvements
"""

import cv2
import numpy as np
from feature_extraction import VideoFeatureExtractor
from pathlib import Path

def test_visual_tracking():
    """Test the visual tracking features"""
    print("🧪 Testing Visual Tracking Features...")
    print("=" * 50)
    
    # 1. Test VideoFeatureExtractor initialization
    print("\n1️⃣ Testing VideoFeatureExtractor initialization...")
    try:
        extractor = VideoFeatureExtractor()
        print("   ✅ VideoFeatureExtractor initialized successfully")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # 2. Test MediaPipe face mesh availability
    print("\n2️⃣ Testing MediaPipe face mesh...")
    try:
        if hasattr(extractor, 'face_mesh') and extractor.face_mesh is not None:
            print("   ✅ MediaPipe face mesh available")
        else:
            print("   ⚠️ MediaPipe face mesh not found")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Test frame processing with dummy frame
    print("\n3️⃣ Testing frame processing...")
    try:
        # Create a test frame (blank image)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Try to process it
        features = extractor._process_frame(test_frame)
        
        if features is None:
            print("   ℹ️ No face detected (expected for blank frame)")
        else:
            print(f"   ✅ Frame processed, extracted {len(features)} features")
    except Exception as e:
        print(f"   ⚠️ Warning: {e}")
    
    # 4. Test suspicious pattern detection
    print("\n4️⃣ Testing suspicious pattern detection...")
    try:
        # Import app module to test detection function
        import sys
        sys.path.append(str(Path(__file__).parent))
        from app import VideoProcessor
        
        processor = VideoProcessor()
        
        # Test with suspicious features
        suspicious_features = {
            'gaze_shift_count': 35,  # High
            'blink_rate': 30,  # High
            'cognitive_load_score': 0.8,  # High
            'head_stability': 0.3,  # Low (unstable)
            'gaze_x_velocity': 15,  # High
            'eye_fixation_duration': 700,  # High
            'emotion_stability': 0.3,  # Low
            'response_delay': 6  # High
        }
        
        is_suspicious, reasons, score = processor._detect_suspicious_patterns(suspicious_features)
        
        print(f"   Suspicious: {is_suspicious}")
        print(f"   Fraud Score: {score:.2f}")
        print(f"   Reasons: {reasons}")
        
        if is_suspicious and len(reasons) > 0:
            print("   ✅ Suspicious pattern detection working")
        else:
            print("   ⚠️ Detection may need tuning")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 5. Test with normal features
    print("\n5️⃣ Testing normal behavior detection...")
    try:
        normal_features = {
            'gaze_shift_count': 15,  # Normal
            'blink_rate': 18,  # Normal
            'cognitive_load_score': 0.4,  # Normal
            'head_stability': 0.8,  # Stable
            'gaze_x_velocity': 3,  # Normal
            'eye_fixation_duration': 300,  # Normal
            'emotion_stability': 0.7,  # Stable
            'response_delay': 2  # Normal
        }
        
        is_suspicious, reasons, score = processor._detect_suspicious_patterns(normal_features)
        
        print(f"   Suspicious: {is_suspicious}")
        print(f"   Fraud Score: {score:.2f}")
        print(f"   Reasons: {reasons}")
        
        if not is_suspicious:
            print("   ✅ Normal behavior correctly identified")
        else:
            print("   ⚠️ False positive detected")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 6. Test visualization components
    print("\n6️⃣ Testing visualization components...")
    try:
        # Test if drawing functions exist
        if hasattr(processor, '_draw_face_mesh'):
            print("   ✅ Face mesh drawing function available")
        if hasattr(processor, '_draw_gaze_direction'):
            print("   ✅ Gaze direction drawing function available")
        if hasattr(processor, '_draw_behavioral_indicators'):
            print("   ✅ Behavioral indicators function available")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Testing complete!")
    print("\n📚 Next Steps:")
    print("   1. Run the Streamlit app: streamlit run app.py")
    print("   2. Click 'AI-Assisted Fraud Detection' in sidebar")
    print("   3. Click START to begin webcam analysis")
    print("   4. You should see:")
    print("      - Cyan face mesh overlay")
    print("      - Green eye regions")
    print("      - Yellow iris dots")
    print("      - Orange gaze arrows")
    print("      - Right panel with behavioral metrics")
    print("      - Bottom fraud risk bar")
    print("\n📖 For more info, see VISUAL_TRACKING_GUIDE.md")


if __name__ == "__main__":
    test_visual_tracking()
