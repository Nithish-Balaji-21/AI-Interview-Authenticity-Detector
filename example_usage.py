"""
Example Usage Script for AI-Assisted Fraud Detection System
Demonstrates all major features and use cases
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print(" AI-ASSISTED FRAUD DETECTION - EXAMPLE USAGE")
print("=" * 70)
print()

# ==================== EXAMPLE 1: GENERATE SYNTHETIC DATA ====================
print("EXAMPLE 1: Generate Synthetic Training Data")
print("-" * 70)

from data_generator import SyntheticDataGenerator

# Create generator
generator = SyntheticDataGenerator(random_seed=42)

# Generate 500 samples
df = generator.generate_balanced_dataset(
    n_samples=500,
    include_edge_cases=True,
    add_temporal=False
)

print(f"✓ Generated {len(df)} samples")
print(f"  - Genuine: {len(df[df['label']==0])} samples")
print(f"  - AI-Assisted: {len(df[df['label']==1])} samples")
print()

# Save dataset
output_path = Path('data/processed/example_dataset.csv')
df.to_csv(output_path, index=False)
print(f"✓ Dataset saved to: {output_path}")
print()


# ==================== EXAMPLE 2: TRAIN MODEL ====================
print("\nEXAMPLE 2: Train Fraud Detection Model")
print("-" * 70)

from train_model import prepare_data, train_random_forest
from sklearn.preprocessing import StandardScaler

# Prepare data
X_train, X_test, y_train, y_test, scaler = prepare_data(df, test_size=0.2)

print(f"✓ Data prepared:")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Test samples: {len(X_test)}")
print()

# Train a simple Random Forest (faster than full ensemble for demo)
print("Training Random Forest classifier...")

# Further split for validation
from sklearn.model_selection import train_test_split
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

model = train_random_forest(X_train_split, y_train_split, X_val, y_val)
print("✓ Model trained successfully")
print()


# ==================== EXAMPLE 3: MAKE PREDICTIONS ====================
print("\nEXAMPLE 3: Make Predictions on Test Set")
print("-" * 70)

# Predict on test set
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Performance:")
print(f"  Accuracy:  {accuracy:.3f}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f}")
print(f"  F1-Score:  {f1:.3f}")
print()


# ==================== EXAMPLE 4: SINGLE SAMPLE PREDICTION ====================
print("\nEXAMPLE 4: Predict Single Sample")
print("-" * 70)

# Get a random test sample
sample_idx = 0
sample_features = X_test[sample_idx:sample_idx+1]
true_label = y_test[sample_idx]

# Predict
prediction = model.predict(sample_features)[0]
confidence = model.predict_proba(sample_features)[0]

print(f"Sample #{sample_idx}:")
print(f"  True Label:      {['Genuine', 'AI-Assisted'][true_label]}")
print(f"  Predicted:       {['Genuine', 'AI-Assisted'][prediction]}")
print(f"  Confidence:      {confidence[prediction]*100:.1f}%")
print(f"  Probabilities:   Genuine={confidence[0]:.3f}, AI-Assisted={confidence[1]:.3f}")
print(f"  Correct:         {'✓' if prediction == true_label else '✗'}")
print()


# ==================== EXAMPLE 5: FEATURE IMPORTANCE ====================
print("\nEXAMPLE 5: Analyze Feature Importance")
print("-" * 70)

import config

# Get feature importance
importances = model.feature_importances_
feature_names = config.FEATURE_NAMES

# Sort by importance
indices = np.argsort(importances)[::-1]

print("Top 5 Most Important Features:")
for i in range(min(5, len(feature_names))):
    idx = indices[i]
    print(f"  {i+1}. {feature_names[idx]:30s}: {importances[idx]:.4f}")
print()


# ==================== EXAMPLE 6: CONFUSION MATRIX ====================
print("\nEXAMPLE 6: Confusion Matrix")
print("-" * 70)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(f"                 Predicted")
print(f"               Genuine  AI-Assisted")
print(f"Actual Genuine    {cm[0,0]:3d}       {cm[0,1]:3d}")
print(f"    AI-Assisted   {cm[1,0]:3d}       {cm[1,1]:3d}")
print()


# ==================== EXAMPLE 7: SAVE MODEL ====================
print("\nEXAMPLE 7: Save Trained Model")
print("-" * 70)

import joblib

# Save model and scaler
model_data = {
    'ensemble': model,  # In real case, this would be the ensemble
    'scaler': scaler
}

model_path = Path('models/example_model.pkl')
joblib.dump(model_data, model_path)

print(f"✓ Model saved to: {model_path}")
print()


# ==================== EXAMPLE 8: LOAD AND USE SAVED MODEL ====================
print("\nEXAMPLE 8: Load Saved Model and Predict")
print("-" * 70)

# Load model
loaded_data = joblib.load(model_path)
loaded_model = loaded_data['ensemble']
loaded_scaler = loaded_data['scaler']

print("✓ Model loaded successfully")

# Create new sample features (simulated)
new_sample = {
    'eye_movement_freq': 6.8,
    'eye_fixation_duration': 170,
    'head_pose_variance': 3.5,
    'head_stability': 0.89,
    'response_delay': 4.8,
    'emotion_stability': 0.86,
    'emotion_intensity': 0.44,
    'micro_expression_count': 2,
    'blink_rate': 12,
    'speech_pause_frequency': 2.1,
    'gaze_dispersion': 24,
    'cognitive_load_score': 0.33
}

# Convert to array
feature_vector = np.array([[new_sample[name] for name in config.FEATURE_NAMES]])

# Normalize
feature_vector = loaded_scaler.transform(feature_vector)

# Predict
prediction = loaded_model.predict(feature_vector)[0]
probabilities = loaded_model.predict_proba(feature_vector)[0]

print("\nNew Sample Prediction:")
print(f"  Features: {new_sample}")
print(f"  Prediction:  {['Genuine', 'AI-Assisted'][prediction]}")
print(f"  Confidence:  {probabilities[prediction]*100:.1f}%")
print()


# ==================== EXAMPLE 9: BATCH PREDICTION ====================
print("\nEXAMPLE 9: Batch Predictions")
print("-" * 70)

# Generate 10 new samples
test_samples = generator.generate_dataset(n_samples=10, class_balance=0.5)

# Extract features
X_batch = test_samples[config.FEATURE_NAMES].values
y_batch = test_samples['label'].values

# Normalize
X_batch = scaler.transform(X_batch)

# Predict
y_batch_pred = model.predict(X_batch)
y_batch_proba = model.predict_proba(X_batch)

print("Batch Prediction Results:")
print(f"  Total samples: {len(X_batch)}")

for i in range(len(X_batch)):
    pred_label = ['Genuine', 'AI-Assisted'][y_batch_pred[i]]
    true_label = ['Genuine', 'AI-Assisted'][y_batch[i]]
    confidence = y_batch_proba[i, y_batch_pred[i]]
    
    status = "✓" if y_batch_pred[i] == y_batch[i] else "✗"
    
    print(f"  Sample {i+1}: {status} Predicted={pred_label:12s} "
          f"(Conf={confidence:.2f}) | True={true_label}")
print()


# ==================== EXAMPLE 10: EVALUATE WITH METRICS ====================
print("\nEXAMPLE 10: Comprehensive Evaluation")
print("-" * 70)

from sklearn.metrics import classification_report

# Generate classification report
report = classification_report(
    y_test, y_pred,
    target_names=['Genuine', 'AI-Assisted'],
    digits=3
)

print("Classification Report:")
print(report)


# ==================== SUMMARY ====================
print("\n" + "=" * 70)
print(" EXAMPLE USAGE COMPLETE!")
print("=" * 70)
print()
print("What we demonstrated:")
print("  ✓ Generated synthetic training data")
print("  ✓ Trained a Random Forest classifier")
print("  ✓ Made predictions on test set")
print("  ✓ Analyzed single sample prediction")
print("  ✓ Examined feature importance")
print("  ✓ Generated confusion matrix")
print("  ✓ Saved and loaded model")
print("  ✓ Used loaded model for new predictions")
print("  ✓ Performed batch predictions")
print("  ✓ Generated comprehensive evaluation")
print()
print("Next steps:")
print("  1. Run full training: python train_model.py --data data/processed/example_dataset.csv")
print("  2. Launch web app: streamlit run app.py")
print("  3. Explore evaluation: python evaluate_model.py --model models/example_model.pkl --data data/processed/example_dataset.csv")
print()
print("=" * 70)
