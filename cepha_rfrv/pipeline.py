import numpy as np
from pathlib import Path
import pickle
import time

class RFRVPipeline:
    """Complete RFRV pipeline for training and detection."""
    
    def __init__(self, annotations_path, images_path, reference_width=800):
        from .data_loader import CephalometricDataLoader
        from .preprocessor import ImagePreprocessor
        from .feature_extractor import HaarFeatureExtractor
        
        self.data_loader = CephalometricDataLoader(annotations_path, images_path)
        self.preprocessor = ImagePreprocessor(reference_width)
        self.feature_extractor = None
        self.trainer = None
        self.detector = None
        
    def setup_feature_extractor(self, patch_size=24, num_features=100):
        """Initialize feature extractor."""
        from .feature_extractor import HaarFeatureExtractor
        self.feature_extractor = HaarFeatureExtractor(patch_size, num_features)
        return self.feature_extractor
    
    def train(self, num_samples=None, landmarks_to_train=None, 
              test_split=0.2, max_displacement=15, 
              num_samples_per_image=50, n_trees=10):
        """Train the complete pipeline."""
        from .trainer import RFRVTrainer
        
        # Load data
        print("Loading data...")
        all_data = self.data_loader.load_data(max_samples=num_samples)
        
        # Split data
        split_idx = int(len(all_data) * (1 - test_split))
        train_data = all_data[:split_idx]
        test_data = all_data[split_idx:]
        
        print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
        
        # Preprocess
        print("Preprocessing...")
        self.train_preprocessed = self.preprocessor.preprocess_dataset(train_data)
        self.test_preprocessed = self.preprocessor.preprocess_dataset(test_data)
        
        # Setup feature extractor if not already done
        if self.feature_extractor is None:
            self.setup_feature_extractor()
        
        # Train
        print("Training models...")
        self.trainer = RFRVTrainer(
            self.feature_extractor, 
            max_displacement=max_displacement,
            num_samples_per_image=num_samples_per_image
        )
        
        self.trainer.train_all_landmarks(
            self.train_preprocessed,
            landmark_keys=landmarks_to_train
        )
        
        return self.trainer
    
    def detect(self, image=None, use_optimized=True, step_size=5):
        """Detect landmarks in image."""
        from .detector import RFRVDetector
        from .detector_optimized import OptimizedRFRVDetector
        
        if self.trainer is None:
            raise ValueError("No trained models. Run train() first or load models.")
        
        # Create detector if needed
        if self.detector is None:
            if use_optimized:
                self.detector = OptimizedRFRVDetector(
                    self.trainer, self.feature_extractor
                )
            else:
                self.detector = RFRVDetector(
                    self.trainer, self.feature_extractor
                )
        
        # Use test image if none provided
        if image is None and hasattr(self, 'test_preprocessed') and self.test_preprocessed:
            image = self.test_preprocessed[0]['enhanced_image']
        
        # Detect
        if use_optimized:
            results = {}
            for landmark_key in self.trainer.rf_models.keys():
                pos, vote_map = self.detector.coarse_to_fine_detection(
                    image, landmark_key
                )
                results[landmark_key] = {'position': pos, 'vote_map': vote_map}
        else:
            results = self.detector.detect_all_landmarks(image, step_size)
        
        return results
    
    def evaluate(self):
        """Evaluate on test set."""
        if not hasattr(self, 'test_preprocessed'):
            raise ValueError("No test data available. Run train() first.")
        
        return self.detector.evaluate_accuracy(self.test_preprocessed)
    
    def save(self, filepath):
        """Save complete pipeline."""
        save_dict = {
            'trainer_models': self.trainer.rf_models if self.trainer else None,
            'preprocessor_mean_shape': self.preprocessor.mean_shape,
            'preprocessor_landmark_keys': self.preprocessor.landmark_keys,
            'feature_extractor_features': self.feature_extractor.features if self.feature_extractor else None,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        print(f"Pipeline saved to {filepath}")
    
    def load(self, filepath):
        """Load saved pipeline."""
        from .trainer import RFRVTrainer
        from .feature_extractor import HaarFeatureExtractor
        
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Restore components
        if save_dict['trainer_models']:
            self.trainer = RFRVTrainer(None, 15, 50)  # Default values
            self.trainer.rf_models = save_dict['trainer_models']
        
        self.preprocessor.mean_shape = save_dict['preprocessor_mean_shape']
        self.preprocessor.landmark_keys = save_dict['preprocessor_landmark_keys']
        
        if save_dict['feature_extractor_features']:
            self.feature_extractor = HaarFeatureExtractor()
            self.feature_extractor.features = save_dict['feature_extractor_features']
        
        print(f"Pipeline loaded from {filepath}")