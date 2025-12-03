"""
Model Evaluation and Explainability Module
Comprehensive evaluation with metrics, visualizations, and SHAP explanations
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
from typing import Dict, Optional, Union, List, Tuple
import joblib

# ML and visualization
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Import project modules
import config
from utils import (
    load_dataframe, plot_confusion_matrix, plot_roc_curve,
    plot_feature_importance, save_json, get_timestamp
)
from train_model import EnsembleModel, FraudDetectionNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== EVALUATION FUNCTIONS ====================

def evaluate_model(model, scaler, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model (ensemble or single)
        scaler: Fitted scaler
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary of metrics and predictions
    """
    logger.info("\n" + "=" * 70)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 70)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary'),
        'roc_auc': roc_auc_score(y_test, y_proba[:, 1])
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(
        y_test, y_pred,
        target_names=config.CLASS_NAMES,
        output_dict=True
    )
    
    # Print results
    logger.info("\nPerformance Metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"  TN: {cm[0,0]:4d} | FP: {cm[0,1]:4d}")
    logger.info(f"  FN: {cm[1,0]:4d} | TP: {cm[1,1]:4d}")
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def create_evaluation_report(results: Dict, save_dir: Path) -> None:
    """
    Create comprehensive evaluation report with visualizations
    
    Args:
        results: Results from evaluate_model
        save_dir: Directory to save visualizations
    """
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING EVALUATION REPORT")
    logger.info("=" * 70)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    
    # 1. Confusion Matrix
    logger.info("Creating confusion matrix plot...")
    cm_path = save_dir / f'confusion_matrix_{timestamp}.png'
    plot_confusion_matrix(
        results['confusion_matrix'],
        config.CLASS_NAMES,
        cm_path
    )
    
    # 2. ROC Curve
    logger.info("Creating ROC curve plot...")
    fpr, tpr, _ = roc_curve(results['y_test'], results['y_proba'][:, 1])
    roc_path = save_dir / f'roc_curve_{timestamp}.png'
    plot_roc_curve(fpr, tpr, results['metrics']['roc_auc'], roc_path)
    
    # 3. Metrics Summary
    logger.info("Creating metrics summary plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    values = [results['metrics'][m] for m in metrics_to_plot]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']
    
    bars = ax.bar(range(len(metrics_to_plot)), values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(metrics_to_plot)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
    ax.set_ylabel('Score')
    ax.set_ylim([0, 1.0])
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    metrics_path = save_dir / f'metrics_summary_{timestamp}.png'
    plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Evaluation report saved to {save_dir}")


def explain_predictions_shap(model, X_test: np.ndarray, 
                            feature_names: List[str],
                            save_dir: Path,
                            n_samples: int = 100) -> None:
    """
    Generate SHAP explanations for model predictions
    
    Args:
        model: Trained model
        X_test: Test features
        feature_names: List of feature names
        save_dir: Directory to save plots
        n_samples: Number of samples to explain
    """
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING SHAP EXPLANATIONS")
    logger.info("=" * 70)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    
    try:
        # Select random samples
        sample_idx = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        X_sample = X_test[sample_idx]
        
        logger.info(f"Computing SHAP values for {len(X_sample)} samples...")
        
        # For ensemble models, we'll use the Random Forest component
        # In production, implement custom SHAP for ensemble
        if hasattr(model, 'models'):
            # Use Random Forest from ensemble
            rf_model = model.models[0]
            explainer = shap.TreeExplainer(rf_model)
        else:
            explainer = shap.TreeExplainer(model)
        
        shap_values = explainer.shap_values(X_sample)
        
        # If binary classification with 2 outputs, use positive class
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        
        # 1. Summary plot (bar)
        logger.info("Creating SHAP summary plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample, feature_names=feature_names,
            plot_type="bar", show=False
        )
        plt.tight_layout()
        summary_path = save_dir / f'shap_summary_{timestamp}.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Summary plot (dot)
        logger.info("Creating SHAP importance plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_sample, feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        importance_path = save_dir / f'shap_importance_{timestamp}.png'
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Waterfall plot for first sample
        logger.info("Creating SHAP waterfall plot...")
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[1],
                data=X_sample[0],
                feature_names=feature_names
            ),
            show=False
        )
        plt.tight_layout()
        waterfall_path = save_dir / f'shap_waterfall_{timestamp}.png'
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ SHAP explanations saved to {save_dir}")
        
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {e}")
        logger.warning("Continuing without SHAP analysis")


def analyze_feature_importance(model, feature_names: List[str], 
                               save_dir: Path) -> None:
    """
    Analyze and visualize feature importance
    
    Args:
        model: Trained model
        feature_names: List of feature names
        save_dir: Directory to save plots
    """
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE IMPORTANCE ANALYSIS")
    logger.info("=" * 70)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    
    try:
        # Get feature importance from Random Forest
        if hasattr(model, 'models'):
            rf_model = model.models[0]
        else:
            rf_model = model
        
        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            # Print top features
            logger.info("\nTop 10 Most Important Features:")
            for i in range(min(10, len(feature_names))):
                idx = indices[i]
                logger.info(f"  {i+1}. {feature_names[idx]:30s}: {importances[idx]:.4f}")
            
            # Plot
            plot_feature_importance(
                feature_names, importances,
                top_n=len(feature_names),
                save_path=save_dir / f'feature_importance_{timestamp}.png'
            )
            
            # Save to JSON
            importance_dict = {
                feature_names[i]: float(importances[i])
                for i in range(len(feature_names))
            }
            save_json(
                importance_dict,
                save_dir / f'feature_importance_{timestamp}.json'
            )
        
    except Exception as e:
        logger.error(f"Error analyzing feature importance: {e}")


def generate_prediction_examples(model, scaler, X_test: np.ndarray,
                                 y_test: np.ndarray, feature_names: List[str],
                                 n_examples: int = 5) -> pd.DataFrame:
    """
    Generate example predictions with explanations
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X_test: Test features
        y_test: Test labels
        feature_names: Feature names
        n_examples: Number of examples per class
        
    Returns:
        DataFrame with example predictions
    """
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING PREDICTION EXAMPLES")
    logger.info("=" * 70)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    examples = []
    
    # Get examples for each class
    for true_class in [0, 1]:
        class_name = config.CLASS_NAMES[true_class]
        
        # Get correctly classified samples
        correct_idx = np.where((y_test == true_class) & (y_pred == true_class))[0]
        
        if len(correct_idx) >= n_examples:
            sample_idx = np.random.choice(correct_idx, n_examples, replace=False)
            
            for idx in sample_idx:
                example = {
                    'true_label': class_name,
                    'predicted_label': config.CLASS_NAMES[y_pred[idx]],
                    'confidence': y_proba[idx, y_pred[idx]],
                    'correct': True
                }
                
                # Add top 3 features
                for i, feat in enumerate(feature_names):
                    example[feat] = X_test[idx, i]
                
                examples.append(example)
    
    df = pd.DataFrame(examples)
    logger.info(f"Generated {len(examples)} example predictions")
    
    return df


# ==================== MAIN EVALUATION PIPELINE ====================

def evaluate_full_pipeline(model_path: Union[str, Path],
                          test_data_path: Union[str, Path],
                          output_dir: Optional[Path] = None) -> None:
    """
    Complete evaluation pipeline
    
    Args:
        model_path: Path to saved model
        test_data_path: Path to test data
        output_dir: Output directory for reports
    """
    logger.info("\n" + "=" * 70)
    logger.info("AI-ASSISTED FRAUD DETECTION - EVALUATION PIPELINE")
    logger.info("=" * 70)
    
    # Setup output directory
    if output_dir is None:
        output_dir = config.MODEL_DIR / 'evaluation_results'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    saved_data = joblib.load(model_path)
    model = saved_data['ensemble']
    scaler = saved_data['scaler']
    
    # Load test data
    logger.info(f"Loading test data from {test_data_path}")
    df_test = load_dataframe(test_data_path)
    
    # Prepare test data
    feature_cols = [col for col in df_test.columns if col in config.FEATURE_NAMES]
    X_test = df_test[feature_cols].values
    y_test = df_test['label'].values
    
    # Normalize
    X_test = scaler.transform(X_test)
    
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Evaluate model
    results = evaluate_model(model, scaler, X_test, y_test)
    results['y_test'] = y_test  # Add for later use
    
    # Create evaluation report
    create_evaluation_report(results, output_dir)
    
    # Feature importance analysis
    analyze_feature_importance(model, config.FEATURE_NAMES, output_dir)
    
    # SHAP explanations
    explain_predictions_shap(
        model, X_test, config.FEATURE_NAMES,
        output_dir, n_samples=100
    )
    
    # Generate prediction examples
    examples_df = generate_prediction_examples(
        model, scaler, X_test, y_test,
        config.FEATURE_NAMES, n_examples=5
    )
    examples_path = output_dir / f'prediction_examples_{get_timestamp()}.csv'
    examples_df.to_csv(examples_path, index=False)
    logger.info(f"✓ Prediction examples saved to {examples_path}")
    
    # Save complete evaluation results
    timestamp = get_timestamp()
    results_json = {
        'timestamp': timestamp,
        'model_path': str(model_path),
        'test_data_path': str(test_data_path),
        'metrics': {k: float(v) for k, v in results['metrics'].items()},
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': results['classification_report']
    }
    
    results_path = output_dir / f'evaluation_results_{timestamp}.json'
    save_json(results_json, results_path)
    logger.info(f"✓ Evaluation results saved to {results_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ EVALUATION COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"\nAll results saved to: {output_dir}")


# ==================== CLI ====================

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Evaluate AI fraud detection model')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model')
    parser.add_argument('--data', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Evaluate
    evaluate_full_pipeline(
        model_path=args.model,
        test_data_path=args.data,
        output_dir=Path(args.output) if args.output else None
    )


if __name__ == "__main__":
    main()
