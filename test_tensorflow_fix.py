"""
Quick test to verify TensorFlow and DeepFace imports work correctly
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Testing TensorFlow import...")
try:
    import tensorflow as tf
    if not hasattr(tf, '__version__'):
        import tf_keras
        tf.__version__ = tf_keras.__version__
    print(f"✓ TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"✗ TensorFlow error: {e}")
    exit(1)

print("\nTesting DeepFace import...")
try:
    from deepface import DeepFace
    print("✓ DeepFace imported successfully")
except Exception as e:
    print(f"✗ DeepFace error: {e}")
    exit(1)

print("\nTesting MediaPipe import...")
try:
    import mediapipe as mp
    print("✓ MediaPipe imported successfully")
except Exception as e:
    print(f"✗ MediaPipe error: {e}")
    exit(1)

print("\n" + "="*50)
print("✓ ALL IMPORTS SUCCESSFUL!")
print("="*50)
print("\nYou can now run: python -m streamlit run app.py")
