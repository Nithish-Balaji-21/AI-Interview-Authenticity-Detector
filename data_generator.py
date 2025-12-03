"""
Synthetic Data Generator for AI-Assisted Fraud Detection
Generates realistic training data based on behavioral profiles

Creates samples with proper statistical distributions matching:
- Genuine candidates (natural behavioral patterns)
- AI-Assisted candidates (reading/typing patterns)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from scipy import stats

import config
from utils import save_dataframe, get_timestamp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generate synthetic behavioral features for training
    Based on research on deception detection and AI-assisted cheating patterns
    """
    
    def __init__(self, random_seed: int = config.RANDOM_SEED):
        """
        Initialize generator with statistical parameters
        
        Args:
            random_seed: Seed for reproducibility
        """
        np.random.seed(random_seed)
        self.params = config.SYNTHETIC_DATA_PARAMS
        self.feature_names = config.FEATURE_NAMES
        logger.info(f"SyntheticDataGenerator initialized (seed={random_seed})")
    
    
    def generate_dataset(self, n_samples: int, 
                        class_balance: float = 0.5,
                        add_noise: bool = True,
                        noise_level: float = 0.05) -> pd.DataFrame:
        """
        Generate complete synthetic dataset
        
        Args:
            n_samples: Total number of samples to generate
            class_balance: Proportion of AI-assisted samples (0 to 1)
            add_noise: Whether to add realistic noise
            noise_level: Amount of noise to add (0 to 1)
            
        Returns:
            DataFrame with features and labels
        """
        logger.info(f"Generating {n_samples} samples (class_balance={class_balance})")
        
        # Calculate samples per class
        n_ai_assisted = int(n_samples * class_balance)
        n_genuine = n_samples - n_ai_assisted
        
        logger.info(f"  - Genuine: {n_genuine} samples")
        logger.info(f"  - AI-Assisted: {n_ai_assisted} samples")
        
        # Generate samples for each class
        genuine_samples = self._generate_class_samples('genuine', n_genuine)
        ai_assisted_samples = self._generate_class_samples('ai_assisted', n_ai_assisted)
        
        # Combine and shuffle
        all_samples = np.vstack([genuine_samples, ai_assisted_samples])
        labels = np.array([0] * n_genuine + [1] * n_ai_assisted)
        
        # Shuffle
        shuffle_idx = np.random.permutation(n_samples)
        all_samples = all_samples[shuffle_idx]
        labels = labels[shuffle_idx]
        
        # Add realistic noise
        if add_noise:
            all_samples = self._add_noise(all_samples, noise_level)
        
        # Create DataFrame
        df = pd.DataFrame(all_samples, columns=self.feature_names)
        df['label'] = labels
        
        # Add metadata
        df['sample_id'] = [f'sample_{i:06d}' for i in range(n_samples)]
        df['class_name'] = df['label'].map({0: 'Genuine', 1: 'AI-Assisted'})
        
        logger.info("Dataset generation complete")
        return df
    
    
    def _generate_class_samples(self, class_name: str, n_samples: int) -> np.ndarray:
        """
        Generate samples for a specific class
        
        Args:
            class_name: 'genuine' or 'ai_assisted'
            n_samples: Number of samples to generate
            
        Returns:
            Array of shape (n_samples, n_features)
        """
        class_params = self.params[class_name]
        samples = np.zeros((n_samples, len(self.feature_names)))
        
        for i, feature_name in enumerate(self.feature_names):
            if feature_name not in class_params:
                logger.warning(f"No parameters for {feature_name}, using defaults")
                samples[:, i] = np.random.randn(n_samples)
                continue
            
            params = class_params[feature_name]
            
            # Generate from truncated normal distribution
            samples[:, i] = self._generate_truncated_normal(
                mean=params['mean'],
                std=params['std'],
                min_val=params['min'],
                max_val=params['max'],
                size=n_samples
            )
        
        return samples
    
    
    def _generate_truncated_normal(self, mean: float, std: float,
                                   min_val: float, max_val: float,
                                   size: int) -> np.ndarray:
        """
        Generate samples from truncated normal distribution
        
        Args:
            mean: Distribution mean
            std: Standard deviation
            min_val: Minimum value
            max_val: Maximum value
            size: Number of samples
            
        Returns:
            Array of samples
        """
        # Calculate truncation parameters
        a = (min_val - mean) / std
        b = (max_val - mean) / std
        
        # Generate from truncated normal
        samples = stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
        
        return samples
    
    
    def _add_noise(self, samples: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add realistic noise to samples to simulate measurement uncertainty
        
        Args:
            samples: Input samples
            noise_level: Noise intensity (0 to 1)
            
        Returns:
            Noisy samples
        """
        # Calculate feature-specific noise based on scale
        feature_scales = np.std(samples, axis=0)
        
        # Generate proportional noise
        noise = np.random.randn(*samples.shape) * feature_scales * noise_level
        
        # Add noise
        noisy_samples = samples + noise
        
        # Ensure non-negative values where appropriate
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in ['eye_movement_freq', 'eye_fixation_duration', 
                               'blink_rate', 'micro_expression_count']:
                noisy_samples[:, i] = np.maximum(noisy_samples[:, i], 0.0)
        
        # Clip values to valid ranges
        noisy_samples = np.clip(noisy_samples, 0.0, None)
        
        return noisy_samples
    
    
    def generate_correlated_features(self, base_df: pd.DataFrame,
                                    correlations: Dict[Tuple[str, str], float]) -> pd.DataFrame:
        """
        Add realistic correlations between features
        
        Args:
            base_df: Base dataset
            correlations: Dictionary mapping feature pairs to correlation coefficients
            
        Returns:
            DataFrame with correlated features
        """
        df = base_df.copy()
        
        for (feature1, feature2), corr in correlations.items():
            if feature1 in df.columns and feature2 in df.columns:
                # Add correlation by blending features
                df[feature2] = (
                    corr * df[feature1] + 
                    (1 - abs(corr)) * df[feature2]
                )
        
        return df
    
    
    def add_temporal_patterns(self, df: pd.DataFrame, 
                             session_length: int = 10) -> pd.DataFrame:
        """
        Add temporal patterns (e.g., fatigue, adaptation over time)
        
        Args:
            df: Input dataset
            session_length: Number of samples per session
            
        Returns:
            DataFrame with temporal features added
        """
        df = df.copy()
        
        # Add session IDs
        n_samples = len(df)
        n_sessions = n_samples // session_length
        df['session_id'] = np.repeat(range(n_sessions), session_length)[:n_samples]
        
        # Add time within session
        df['time_in_session'] = df.groupby('session_id').cumcount()
        
        # Simulate fatigue (some features change over time)
        fatigue_factor = 1.0 + 0.1 * (df['time_in_session'] / session_length)
        
        # Apply fatigue to relevant features
        df['blink_rate'] *= fatigue_factor
        df['cognitive_load_score'] *= fatigue_factor
        
        return df
    
    
    def create_edge_cases(self, n_samples: int = 50) -> pd.DataFrame:
        """
        Generate edge cases for robust model training
        
        Args:
            n_samples: Number of edge cases to generate
            
        Returns:
            DataFrame with edge case samples
        """
        edge_cases = []
        
        # 1. Nervous genuine candidate (high movement, high cognitive load)
        nervous_genuine = self._generate_class_samples('genuine', n_samples // 5)
        nervous_genuine[:, self.feature_names.index('cognitive_load_score')] *= 1.5
        nervous_genuine[:, self.feature_names.index('eye_movement_freq')] *= 1.3
        
        # 2. Confident AI-assisted (lower response delay)
        confident_cheater = self._generate_class_samples('ai_assisted', n_samples // 5)
        confident_cheater[:, self.feature_names.index('response_delay')] *= 0.7
        
        # 3. Distracted genuine (high gaze dispersion)
        distracted = self._generate_class_samples('genuine', n_samples // 5)
        distracted[:, self.feature_names.index('gaze_dispersion')] *= 1.4
        
        # 4. Well-prepared cheater (more natural emotions)
        prepared_cheater = self._generate_class_samples('ai_assisted', n_samples // 5)
        prepared_cheater[:, self.feature_names.index('emotion_stability')] *= 0.8
        prepared_cheater[:, self.feature_names.index('emotion_intensity')] *= 1.2
        
        # 5. Uncertain genuine (high response delay)
        uncertain = self._generate_class_samples('genuine', n_samples // 5)
        uncertain[:, self.feature_names.index('response_delay')] *= 1.8
        
        # Combine edge cases
        all_edge_cases = np.vstack([
            nervous_genuine, confident_cheater, distracted, 
            prepared_cheater, uncertain
        ])
        
        # Labels
        labels = np.array(
            [0] * (n_samples // 5) + [1] * (n_samples // 5) + [0] * (n_samples // 5) + 
            [1] * (n_samples // 5) + [0] * (n_samples // 5)
        )
        
        # Create DataFrame
        df = pd.DataFrame(all_edge_cases, columns=self.feature_names)
        df['label'] = labels
        df['sample_id'] = [f'edge_{i:04d}' for i in range(len(df))]
        df['class_name'] = df['label'].map({0: 'Genuine', 1: 'AI-Assisted'})
        df['is_edge_case'] = True
        
        return df
    
    
    def generate_balanced_dataset(self, n_samples: int,
                                 include_edge_cases: bool = True,
                                 add_temporal: bool = False) -> pd.DataFrame:
        """
        Generate a complete, balanced dataset with all enhancements
        
        Args:
            n_samples: Total number of samples
            include_edge_cases: Whether to include edge cases
            add_temporal: Whether to add temporal patterns
            
        Returns:
            Complete dataset ready for training
        """
        # Generate main dataset
        df = self.generate_dataset(n_samples, class_balance=0.5)
        
        # Add edge cases
        if include_edge_cases:
            edge_cases = self.create_edge_cases(n_samples // 10)
            df = pd.concat([df, edge_cases], ignore_index=True)
            logger.info(f"Added {len(edge_cases)} edge cases")
        
        # Add temporal patterns
        if add_temporal:
            df = self.add_temporal_patterns(df)
            logger.info("Added temporal patterns")
        
        # Add realistic correlations
        correlations = {
            ('response_delay', 'cognitive_load_score'): 0.4,
            ('emotion_stability', 'emotion_intensity'): -0.3,
            ('eye_movement_freq', 'gaze_dispersion'): 0.5,
            ('head_pose_variance', 'head_stability'): -0.7,
        }
        df = self.generate_correlated_features(df, correlations)
        logger.info("Added feature correlations")
        
        # Shuffle
        df = df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
        
        return df


# ==================== VISUALIZATION ====================

def visualize_dataset(df: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Create visualization of generated dataset
    
    Args:
        df: Generated dataset
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    feature_cols = [col for col in df.columns if col in config.FEATURE_NAMES]
    
    for i, feature in enumerate(feature_cols):
        ax = axes[i]
        
        # Plot distributions for both classes
        genuine_data = df[df['label'] == 0][feature]
        ai_assisted_data = df[df['label'] == 1][feature]
        
        ax.hist(genuine_data, bins=30, alpha=0.6, label='Genuine', color='blue')
        ax.hist(ai_assisted_data, bins=30, alpha=0.6, label='AI-Assisted', color='red')
        
        ax.set_title(feature.replace('_', ' ').title(), fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    plt.show()


def print_dataset_statistics(df: pd.DataFrame):
    """
    Print comprehensive statistics about generated dataset
    
    Args:
        df: Generated dataset
    """
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    print(f"\nTotal Samples: {len(df)}")
    print(f"Features: {len([c for c in df.columns if c in config.FEATURE_NAMES])}")
    
    print("\nClass Distribution:")
    class_counts = df['label'].value_counts()
    for label, count in class_counts.items():
        class_name = 'Genuine' if label == 0 else 'AI-Assisted'
        percentage = (count / len(df)) * 100
        print(f"  {class_name:15s}: {count:5d} ({percentage:5.2f}%)")
    
    print("\nFeature Statistics:")
    print("-" * 70)
    
    feature_cols = [col for col in df.columns if col in config.FEATURE_NAMES]
    
    for feature in feature_cols:
        genuine_mean = df[df['label'] == 0][feature].mean()
        ai_assisted_mean = df[df['label'] == 1][feature].mean()
        diff_percent = ((ai_assisted_mean - genuine_mean) / genuine_mean) * 100
        
        print(f"{feature:30s}: Genuine={genuine_mean:7.2f} | "
              f"AI-Assisted={ai_assisted_mean:7.2f} | Diff={diff_percent:+6.1f}%")
    
    print("=" * 70 + "\n")


# ==================== MAIN ====================

def main():
    """Command-line interface for data generation"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic dataset for AI fraud detection'
    )
    parser.add_argument(
        '--samples', type=int, default=1000,
        help='Number of samples to generate (default: 1000)'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output CSV file path (default: data/synthetic_dataset_TIMESTAMP.csv)'
    )
    parser.add_argument(
        '--balance', type=float, default=0.5,
        help='Proportion of AI-assisted samples (default: 0.5)'
    )
    parser.add_argument(
        '--edge-cases', action='store_true',
        help='Include edge cases in dataset'
    )
    parser.add_argument(
        '--temporal', action='store_true',
        help='Add temporal patterns'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Create visualization of dataset'
    )
    parser.add_argument(
        '--seed', type=int, default=config.RANDOM_SEED,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticDataGenerator(random_seed=args.seed)
    
    # Generate dataset
    print(f"\nGenerating {args.samples} samples...")
    df = generator.generate_balanced_dataset(
        n_samples=args.samples,
        include_edge_cases=args.edge_cases,
        add_temporal=args.temporal
    )
    
    # Print statistics
    print_dataset_statistics(df)
    
    # Save dataset
    if args.output is None:
        output_path = config.PROCESSED_DATA_DIR / f'synthetic_dataset_{get_timestamp()}.csv'
    else:
        output_path = Path(args.output)
    
    save_dataframe(df, output_path)
    print(f"✓ Dataset saved to: {output_path}")
    
    # Visualize if requested
    if args.visualize:
        viz_path = output_path.parent / f'{output_path.stem}_visualization.png'
        visualize_dataset(df, viz_path)
    
    print("\n✓ Data generation complete!\n")


if __name__ == "__main__":
    main()
