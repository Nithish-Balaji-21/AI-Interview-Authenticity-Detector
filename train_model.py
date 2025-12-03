"""
Model Training Pipeline for AI-Assisted Fraud Detection
Implements ensemble learning with Random Forest, Neural Network, and Gradient Boosting
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import argparse
import logging
from datetime import datetime
import joblib

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import project modules
import config
from utils import (
    load_dataframe, save_json, get_timestamp, 
    format_duration, ProgressTracker, plot_training_history
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== NEURAL NETWORK MODEL ====================

class FraudDetectionNN(nn.Module):
    """Neural Network for fraud classification"""
    
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout_rate: float = 0.3):
        super(FraudDetectionNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class FraudDataset(Dataset):
    """PyTorch dataset for fraud detection"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ==================== TRAINING UTILITIES ====================

def prepare_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
    """
    Prepare data for training
    
    Args:
        df: Input dataframe
        test_size: Proportion of test set
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    logger.info("Preparing data for training...")
    
    # Extract features and labels
    feature_cols = [col for col in df.columns if col in config.FEATURE_NAMES]
    X = df[feature_cols].values
    y = df['label'].values
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Labels shape: {y.shape}")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=config.RANDOM_SEED, stratify=y
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, scaler


def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Calculate class weights for imbalanced data"""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {int(cls): total / (len(unique) * count) for cls, count in zip(unique, counts)}
    logger.info(f"Class weights: {weights}")
    return weights


# ==================== MODEL TRAINING FUNCTIONS ====================

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> RandomForestClassifier:
    """Train Random Forest classifier"""
    logger.info("\n" + "=" * 60)
    logger.info("Training Random Forest Classifier")
    logger.info("=" * 60)
    
    rf_params = config.ENSEMBLE_MODELS['random_forest']
    
    # Calculate class weights
    if config.USE_CLASS_WEIGHTS:
        class_weights = calculate_class_weights(y_train)
    else:
        class_weights = None
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=rf_params['n_estimators'],
        max_depth=rf_params['max_depth'],
        min_samples_split=rf_params['min_samples_split'],
        min_samples_leaf=rf_params['min_samples_leaf'],
        max_features=rf_params['max_features'],
        bootstrap=rf_params['bootstrap'],
        class_weight=class_weights,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        verbose=1
    )
    
    # Train
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Evaluate on validation set
    y_pred_val = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    
    logger.info(f"Training time: {format_duration(training_time)}")
    logger.info(f"Validation accuracy: {val_accuracy:.4f}")
    logger.info(f"Feature importances calculated")
    
    return model


def train_gradient_boosting(X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> GradientBoostingClassifier:
    """Train Gradient Boosting classifier"""
    logger.info("\n" + "=" * 60)
    logger.info("Training Gradient Boosting Classifier")
    logger.info("=" * 60)
    
    gb_params = config.ENSEMBLE_MODELS['gradient_boosting']
    
    # Initialize model
    model = GradientBoostingClassifier(
        n_estimators=gb_params['n_estimators'],
        learning_rate=gb_params['learning_rate'],
        max_depth=gb_params['max_depth'],
        subsample=gb_params['subsample'],
        random_state=config.RANDOM_SEED,
        verbose=1
    )
    
    # Train
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Evaluate
    y_pred_val = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred_val)
    
    logger.info(f"Training time: {format_duration(training_time)}")
    logger.info(f"Validation accuracy: {val_accuracy:.4f}")
    
    return model


def train_neural_network(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        epochs: int = config.NUM_EPOCHS) -> FraudDetectionNN:
    """Train Neural Network classifier"""
    logger.info("\n" + "=" * 60)
    logger.info("Training Neural Network Classifier")
    logger.info("=" * 60)
    
    nn_params = config.ENSEMBLE_MODELS['neural_network']
    
    # Create datasets
    train_dataset = FraudDataset(X_train, y_train)
    val_dataset = FraudDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=0
    )
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = FraudDetectionNN(
        input_dim=input_dim,
        hidden_layers=nn_params['hidden_layers'],
        dropout_rate=nn_params['dropout_rate']
    )
    model = model.to(config.DEVICE)
    
    # Loss and optimizer
    if config.USE_CLASS_WEIGHTS:
        class_weights = calculate_class_weights(y_train)
        weights = torch.FloatTensor([class_weights[0], class_weights[1]]).to(config.DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    start_time = datetime.now()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training time: {format_duration(training_time)}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save training history plot
    plot_path = config.MODEL_DIR / f'training_history_{get_timestamp()}.png'
    plot_training_history(history, plot_path)
    
    return model, history


# ==================== ENSEMBLE MODEL ====================

class EnsembleModel:
    """Ensemble of multiple classifiers with soft voting"""
    
    def __init__(self, models: List, weights: Optional[List[float]] = None):
        """
        Initialize ensemble
        
        Args:
            models: List of trained models
            weights: Optional voting weights for each model
        """
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        logger.info(f"Ensemble initialized with {len(models)} models")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities using weighted voting"""
        probas = []
        
        for i, model in enumerate(self.models):
            if isinstance(model, FraudDetectionNN):
                # Neural network prediction
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(config.DEVICE)
                    outputs = model(X_tensor)
                    proba = torch.softmax(outputs, dim=1).cpu().numpy()
            else:
                # Scikit-learn model
                proba = model.predict_proba(X)
            
            probas.append(proba * self.weights[i])
        
        # Average probabilities
        ensemble_proba = np.sum(probas, axis=0) / np.sum(self.weights)
        return ensemble_proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


# ==================== MAIN TRAINING PIPELINE ====================

def train_full_pipeline(data_path: Union[str, Path],
                       save_models: bool = True) -> Dict:
    """
    Complete training pipeline
    
    Args:
        data_path: Path to training data CSV
        save_models: Whether to save trained models
        
    Returns:
        Dictionary with trained models and metadata
    """
    logger.info("\n" + "=" * 70)
    logger.info("AI-ASSISTED FRAUD DETECTION - TRAINING PIPELINE")
    logger.info("=" * 70)
    
    # Load data
    df = load_dataframe(data_path)
    logger.info(f"Loaded {len(df)} samples from {data_path}")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Further split training into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y_train
    )
    
    # Train individual models
    rf_model = train_random_forest(X_train, y_train, X_val, y_val)
    gb_model = train_gradient_boosting(X_train, y_train, X_val, y_val)
    nn_model, history = train_neural_network(X_train, y_train, X_val, y_val)
    
    # Create ensemble
    logger.info("\n" + "=" * 60)
    logger.info("Creating Ensemble Model")
    logger.info("=" * 60)
    
    ensemble = EnsembleModel(
        models=[rf_model, nn_model, gb_model],
        weights=config.ENSEMBLE_WEIGHTS
    )
    
    # Evaluate ensemble on test set
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Accuracy:  {test_accuracy:.4f}")
    logger.info(f"  Precision: {test_precision:.4f}")
    logger.info(f"  Recall:    {test_recall:.4f}")
    logger.info(f"  F1-Score:  {test_f1:.4f}")
    
    # Save models
    if save_models:
        timestamp = get_timestamp()
        
        # Save ensemble and scaler
        model_path = config.MODEL_DIR / f'ai_fraud_detector_{timestamp}.pkl'
        scaler_path = config.MODEL_DIR / f'feature_scaler_{timestamp}.pkl'
        
        joblib.dump({'ensemble': ensemble, 'scaler': scaler}, model_path)
        logger.info(f"✓ Model saved to: {model_path}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'training_samples': len(X_train) + len(X_val),
            'test_samples': len(X_test),
            'features': config.FEATURE_NAMES,
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1': float(test_f1),
            'model_path': str(model_path),
            'scaler_path': str(scaler_path)
        }
        
        metadata_path = config.MODEL_DIR / f'metadata_{timestamp}.json'
        save_json(metadata, metadata_path)
        logger.info(f"✓ Metadata saved to: {metadata_path}")
    
    return {
        'ensemble': ensemble,
        'scaler': scaler,
        'rf_model': rf_model,
        'gb_model': gb_model,
        'nn_model': nn_model,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'history': history
    }


# ==================== CLI ====================

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Train AI fraud detection model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Training epochs')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save trained models')
    
    args = parser.parse_args()
    
    # Update config if needed
    if args.epochs != config.NUM_EPOCHS:
        config.NUM_EPOCHS = args.epochs
    
    # Train
    results = train_full_pipeline(args.data, save_models=not args.no_save)
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
