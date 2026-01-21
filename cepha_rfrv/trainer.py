import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import matplotlib.pyplot as plt

class RFRVTrainer:
    """
    Implements the Random Forest Regression Voting training process.
    Following the paper: "To train the detector for a single feature point we 
    generate samples by extracting features at a set of random displacements 
    from the true position."
    """
    
    def __init__(self, feature_extractor, max_displacement=100, num_samples_per_image=100):
        """
        Initialize the RFRV trainer.
        
        Args:
            feature_extractor: HaarFeatureExtractor instance
            max_displacement: Maximum displacement for training samples (pixels)
                            Paper uses 5-15 pixels depending on stage
            num_samples_per_image: Number of random samples per landmark per image
        """
        self.feature_extractor = feature_extractor
        self.max_displacement = max_displacement
        self.num_samples_per_image = num_samples_per_image
        self.rf_models = {}  # Store one RF model per landmark
        
    def generate_training_samples(self, preprocessed_data, landmark_key):
        """
        Generate training samples for a specific landmark.
        
        For each image and landmark:
        1. Sample random points around the true landmark position
        2. Extract features at these points
        3. Record the displacement vector from sample point to true landmark
        
        This teaches the model: "When you see these features, the landmark 
        is probably this distance and direction away"
        """
        X_train = []  # Features
        y_train = []  # Displacement vectors (dx, dy)
        
        print(f"Generating training samples for landmark '{landmark_key}'...")
        
        for sample in tqdm(preprocessed_data):
            if landmark_key not in sample['normalized_landmarks']:
                continue
                
            image = sample['enhanced_image']
            true_pos = sample['normalized_landmarks'][landmark_key]
            
            # Generate random displacements
            for _ in range(self.num_samples_per_image):
                # Random displacement within [-max_displacement, +max_displacement]
                dx = np.random.uniform(-self.max_displacement, self.max_displacement)
                dy = np.random.uniform(-self.max_displacement, self.max_displacement)
                
                # Sample point (where we extract features)
                sample_x = true_pos[0] + dx
                sample_y = true_pos[1] + dy
                
                # Skip if sample point is outside image bounds
                h, w = image.shape
                if sample_x < 0 or sample_x >= w or sample_y < 0 or sample_y >= h:
                    continue
                
                # Extract features at sample point
                features = self.feature_extractor.extract_features(image, sample_x, sample_y)
                
                # The target is the displacement FROM the sample point TO the true landmark
                # This is what the model needs to predict: "landmark is here relative to current point"
                target_displacement = np.array([-dx, -dy])  # Negative because we want vector TO landmark
                
                X_train.append(features)
                y_train.append(target_displacement)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        print(f"Generated {len(X_train)} training samples for landmark '{landmark_key}'")
        
        return X_train, y_train
    
    def train_landmark_detector(self, X_train, y_train, landmark_key, 
                               n_trees=10, max_depth=15):
        """
        Train a Random Forest regressor for a specific landmark.
        
        Following the paper: "We train a set of randomised decision trees 
        on the pairs {features, displacement}"
        """
        print(f"Training Random Forest for landmark '{landmark_key}'...")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Features per sample: {X_train.shape[1]}")
        print(f"  Number of trees: {n_trees}")
        
        # Create and train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',  # Use sqrt of features at each split
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        # Train the model
        rf_model.fit(X_train, y_train)
        
        # Store the trained model
        self.rf_models[landmark_key] = rf_model
        
        # Calculate training error (for monitoring)
        predictions = rf_model.predict(X_train)
        mae = np.mean(np.abs(predictions - y_train))
        rmse = np.sqrt(np.mean((predictions - y_train)**2))
        
        print(f"  Training MAE: {mae:.2f} pixels")
        print(f"  Training RMSE: {rmse:.2f} pixels")
        
        return rf_model
    
    def train_all_landmarks(self, preprocessed_data, landmark_keys=None):
        """
        Train Random Forest models for all landmarks.
        """
        if landmark_keys is None:
            # Use all common landmarks
            landmark_keys = set()
            for sample in preprocessed_data:
                if not landmark_keys:
                    landmark_keys = set(sample['normalized_landmarks'].keys())
                else:
                    landmark_keys = landmark_keys.intersection(sample['normalized_landmarks'].keys())
            landmark_keys = sorted(landmark_keys)
        
        print(f"\nTraining models for {len(landmark_keys)} landmarks...")
        
        for landmark_key in landmark_keys:
            print(f"\n{'='*50}")
            print(f"Training landmark: {landmark_key}")
            print(f"{'='*50}")
            
            # Generate training data
            X_train, y_train = self.generate_training_samples(
                preprocessed_data, landmark_key
            )
            
            if len(X_train) > 0:
                # Train model
                self.train_landmark_detector(X_train, y_train, landmark_key)
            else:
                print(f"Warning: No training samples for landmark '{landmark_key}'")
        
        print(f"\n{'='*50}")
        print(f"Training complete! Trained {len(self.rf_models)} models")
        print(f"{'='*50}")
        
        return self.rf_models
    
    def visualize_displacement_distribution(self, preprocessed_data, landmark_key):
        """
        Visualize the distribution of training displacements for a landmark.
        Helps understand the training data distribution.
        """
        if landmark_key not in preprocessed_data[0]['normalized_landmarks']:
            print(f"Landmark '{landmark_key}' not found")
            return
        
        # Generate some training samples
        X_train, y_train = self.generate_training_samples(
            preprocessed_data[:1],  # Just use first image for visualization
            landmark_key
        )
        
        if len(y_train) == 0:
            print("No training samples generated")
            return
        
        # Plot displacement distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot of displacements
        ax1.scatter(y_train[:, 0], y_train[:, 1], alpha=0.5, s=10)
        ax1.set_xlabel('X displacement (pixels)')
        ax1.set_ylabel('Y displacement (pixels)')
        ax1.set_title(f'Training Displacement Distribution for {landmark_key}')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Add circle to show max displacement
        circle = plt.Circle((0, 0), self.max_displacement, 
                           fill=False, color='red', linestyle='--')
        ax1.add_patch(circle)
        
        # Histogram of displacement magnitudes
        magnitudes = np.sqrt(y_train[:, 0]**2 + y_train[:, 1]**2)
        ax2.hist(magnitudes, bins=30, edgecolor='black')
        ax2.set_xlabel('Displacement magnitude (pixels)')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Displacement Magnitudes')
        ax2.axvline(self.max_displacement, color='red', linestyle='--', 
                   label=f'Max displacement: {self.max_displacement}px')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
