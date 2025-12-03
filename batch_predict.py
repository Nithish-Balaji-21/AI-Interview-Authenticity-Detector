"""
Batch prediction script for processing multiple videos
Useful for evaluating model on a dataset of videos
"""

import argparse
from pathlib import Path
import pandas as pd
import joblib
from tqdm import tqdm
import logging

from feature_extraction import VideoFeatureExtractor
import config
from utils import ensure_dir, get_timestamp, save_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def batch_predict(video_dir: Path, model_path: Path, output_path: Path):
    """
    Process multiple videos and generate predictions
    
    Args:
        video_dir: Directory containing videos
        model_path: Path to trained model
        output_path: Path to save results CSV
    """
    logger.info("=" * 70)
    logger.info("BATCH PREDICTION")
    logger.info("=" * 70)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    saved_data = joblib.load(model_path)
    model = saved_data['ensemble']
    scaler = saved_data['scaler']
    
    # Find all videos
    video_files = []
    for ext in config.SUPPORTED_VIDEO_FORMATS:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    if len(video_files) == 0:
        logger.error("No videos found!")
        return
    
    # Initialize feature extractor
    extractor = VideoFeatureExtractor()
    
    # Process each video
    results = []
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        try:
            # Extract features
            features = extractor.process_video(video_file)
            
            # Prepare for prediction
            feature_vector = [features[name] for name in config.FEATURE_NAMES]
            feature_vector = scaler.transform([feature_vector])
            
            # Predict
            prediction = model.predict(feature_vector)[0]
            probabilities = model.predict_proba(feature_vector)[0]
            
            # Store results
            result = {
                'filename': video_file.name,
                'prediction': config.CLASS_NAMES[prediction],
                'confidence': probabilities[prediction],
                'genuine_prob': probabilities[0],
                'ai_assisted_prob': probabilities[1]
            }
            
            # Add features
            for name, value in features.items():
                result[f'feature_{name}'] = value
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {video_file.name}: {e}")
            results.append({
                'filename': video_file.name,
                'prediction': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            })
    
    # Clean up
    extractor.close()
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    save_dataframe(df, output_path)
    logger.info(f"✓ Results saved to {output_path}")
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total videos processed: {len(video_files)}")
    
    if 'prediction' in df.columns:
        prediction_counts = df['prediction'].value_counts()
        for pred, count in prediction_counts.items():
            percentage = (count / len(df)) * 100
            logger.info(f"  {pred}: {count} ({percentage:.1f}%)")
    
    logger.info("=" * 70)


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Batch predict fraud detection on multiple videos'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Directory containing videos'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output CSV path (default: results_<timestamp>.csv)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    video_dir = Path(args.input)
    model_path = Path(args.model)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = config.PROCESSED_DATA_DIR / f'batch_results_{get_timestamp()}.csv'
    
    # Validate inputs
    if not video_dir.exists():
        logger.error(f"Video directory not found: {video_dir}")
        return
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # Run batch prediction
    batch_predict(video_dir, model_path, output_path)


if __name__ == "__main__":
    main()
