"""
Quick Installation Verification Script
Run this after installing requirements to verify everything works
"""

import sys
import importlib

print("\n" + "=" * 70)
print(" 🔍 VERIFYING INSTALLATION")
print("=" * 70)

# Required packages with import names
PACKAGES = {
    # Core ML/DL
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'sklearn': 'Scikit-Learn',
    
    # Computer Vision
    'cv2': 'OpenCV',
    'mediapipe': 'MediaPipe',
    'deepface': 'DeepFace',
    'PIL': 'Pillow',
    
    # Data Science
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'scipy': 'SciPy',
    
    # Visualization
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'plotly': 'Plotly',
    
    # ML Utilities
    'shap': 'SHAP',
    'joblib': 'Joblib',
    'tqdm': 'TQDM',
    
    # Web Application
    'streamlit': 'Streamlit',
    'streamlit_webrtc': 'Streamlit-WebRTC',
    
    # Utilities
    'dotenv': 'Python-Dotenv',
}

# Check each package
missing = []
installed = []

for module, name in PACKAGES.items():
    try:
        importlib.import_module(module)
        print(f"  ✅ {name:30s} [OK]")
        installed.append(name)
    except ImportError:
        print(f"  ❌ {name:30s} [MISSING]")
        missing.append(name)

print("=" * 70)

# Summary
print(f"\n📊 Summary:")
print(f"  ✅ Installed: {len(installed)}/{len(PACKAGES)}")
print(f"  ❌ Missing: {len(missing)}/{len(PACKAGES)}")

# System info
print(f"\n💻 System Information:")
print(f"  Python version: {sys.version.split()[0]}")
print(f"  Platform: {sys.platform}")

try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
except:
    pass

# Result
print("\n" + "=" * 70)

if not missing:
    print("\n🎉 SUCCESS! All packages installed correctly!")
    print("\n✅ Next Steps:")
    print("  1. Run demo: streamlit run app.py")
    print("  2. Generate data: python data_generator.py --samples 100")
    print("  3. See examples: python example_usage.py")
    print("\n📚 Documentation:")
    print("  - QUICKSTART.md - Get started in 5 minutes")
    print("  - INSTALL_GUIDE.md - Detailed installation")
    print("  - README.md - Full documentation")
else:
    print("\n⚠️ WARNING! Some packages are missing!")
    print("\n❌ Missing packages:")
    for pkg in missing:
        print(f"  - {pkg}")
    print("\n🔧 To fix, run:")
    print("  pip install -r requirements.txt")

print("=" * 70 + "\n")
