# AI Interview Authenticity Detector

A machine learning project that detects whether a candidate is using AI assistance during video interviews.

## What This Project Does

This system analyzes video interviews to detect if someone is using AI help. It looks at:
- Eye movement patterns
- Head position and movements  
- Facial expressions and emotions
- Response timing and delays

Based on these patterns, it classifies the candidate as "Genuine" or "AI-Assisted".

## Technologies Used

- **Python** - Programming language
- **OpenCV** - Video processing
- **MediaPipe** - Face detection
- **Scikit-learn** - Machine learning models
- **Streamlit** - Web interface

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Training Data
```bash
python data_generator.py
```

### 3. Train the Model
```bash
python train_model.py
```

### 4. Test the Model

Option A - Command line:
```bash
python batch_predict.py --input videos/ --output results.csv
```

Option B - Web interface:
```bash
streamlit run app.py
```

## Project Structure

```
├── data/                      # Training and test data
├── models/                    # Saved trained models
├── config.py                  # Settings
├── train_model.py             # Training script
├── batch_predict.py           # Batch prediction
├── feature_extraction.py      # Feature extraction
├── app.py                     # Streamlit web interface
├── requirements.txt           # Python dependencies
└── README.md
```
