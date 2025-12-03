"""
AI-Assisted Fraud Detection System
Complete End-to-End Machine Learning Project

This is the main initialization file that can be used to verify all components
"""

__version__ = "1.0.0"
__author__ = "AI Fraud Detection Team"
__description__ = "AI-Assisted Fraud Detection in Online Interviews"

# Version info
VERSION_INFO = {
    'version': __version__,
    'author': __author__,
    'description': __description__,
    'python_requires': '>=3.8',
    'status': 'Production Ready'
}

# Module exports
__all__ = [
    'config',
    'utils',
    'feature_extraction',
    'data_generator',
    'train_model',
    'evaluate_model',
    'batch_predict'
]

def print_project_info():
    """Print project information"""
    print("\n" + "=" * 70)
    print(f" {__description__}")
    print("=" * 70)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Status: Production Ready")
    print("=" * 70)
    print("\nKey Features:")
    print("  ✓ Real-time behavioral analysis")
    print("  ✓ Multi-modal feature extraction")
    print("  ✓ Ensemble learning (RF + NN + GB)")
    print("  ✓ Model explainability with SHAP")
    print("  ✓ Interactive Streamlit UI")
    print("  ✓ Comprehensive evaluation tools")
    print("\nQuick Start:")
    print("  1. Run: python data_generator.py --samples 1000")
    print("  2. Run: python train_model.py --data data/synthetic_dataset.csv")
    print("  3. Run: streamlit run app.py")
    print("\nFor detailed instructions, see README.md and QUICKSTART.md")
    print("=" * 70 + "\n")


def verify_installation():
    """Verify all required packages are installed"""
    import sys
    
    required_packages = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'deepface': 'deepface',
        'torch': 'pytorch',
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'streamlit': 'streamlit',
        'shap': 'shap',
        'joblib': 'joblib'
    }
    
    print("\n" + "=" * 70)
    print(" VERIFYING INSTALLATION")
    print("=" * 70)
    
    all_installed = True
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package:30s} [OK]")
        except ImportError:
            print(f"  ✗ {package:30s} [MISSING]")
            all_installed = False
    
    print("=" * 70)
    
    if all_installed:
        print("\n✓ All packages installed successfully!")
        print("\nSystem Information:")
        print(f"  Python version: {sys.version.split()[0]}")
        
        try:
            import torch
            print(f"  PyTorch version: {torch.__version__}")
            print(f"  CUDA available: {torch.cuda.is_available()}")
        except:
            pass
        
        return True
    else:
        print("\n✗ Some packages are missing!")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    print_project_info()
    verify_installation()
