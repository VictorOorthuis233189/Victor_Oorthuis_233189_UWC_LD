import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

class RFRVDetector:
    """
    Implements the detection phase using Random Forest Regression Voting.
    Following the paper: "At each point, features are sampled. The regressor 
    predicts the most likely position of the target point. The predictions 
    are used to vote for the best position in an accumulator array."
    """
    
    def __init__(self, trainer, feature_extractor, vote_sigma=3.0):
        """
        Initialize the detector.
        
        Args:
            trainer: Trained RFRVTrainer with RF models
            feature_extractor: HaarFeatureExtractor instance
            vote_sigma: Standard deviation for Gaussian voting (pixels)
        """
        self.trainer = trainer
        self.feature_extractor = feature_extractor
        self.vote_sigma = vote_sigma
        
    def detect_landmark(self, image, landmark_key, 
                       search_region=None, step_size=3, 
                       voting_method='gaussian'):
        """
        Detect a single landmark using regression voting.
        
        Args:
            image: Input image (preprocessed)
            landmark_key: Which landmark to detect
            search_region: (x_min, y_min, x_max, y_max) or None for full image
            step_size: Spacing between sample points (paper suggests 3 for speed)
            voting_method: 'single', 'gaussian', or 'weighted'
        
        Returns:
            predicted_position: (x, y) coordinates
            vote_map: The voting accumulator array (heatmap)
        """
        if landmark_key not in self.trainer.rf_models:
            raise ValueError(f"No trained model for landmark '{landmark_key}'")
        
        rf_model = self.trainer.rf_models[landmark_key]
        h, w = image.shape
        
        # Define search region
        if search_region is None:
            x_min, y_min = 0, 0
            x_max, y_max = w, h
        else:
            x_min, y_min, x_max, y_max = search_region
        
        # Initialize vote accumulator
        vote_map = np.zeros((h, w), dtype=np.float32)
        
        # Sample points on a grid with given step size
        # Paper: "Good results can be obtained by evaluating on a sparse grid"
        sample_points_x = range(x_min, x_max, step_size)
        sample_points_y = range(y_min, y_max, step_size)
        
        print(f"Detecting landmark '{landmark_key}'...")
        print(f"  Search region: ({x_min}, {y_min}) to ({x_max}, {y_max})")
        print(f"  Step size: {step_size}")
        print(f"  Sample points: {len(sample_points_x)} x {len(sample_points_y)}")
        
        # For each sample point
        vote_count = 0
        for y in tqdm(sample_points_y, desc="Scanning"):
            for x in sample_points_x:
                # Extract features at this point
                features = self.feature_extractor.extract_features(image, x, y)
                
                # Predict displacement to landmark (returns [dx, dy])
                # This is an ensemble prediction from all trees in the forest
                displacement = rf_model.predict(features.reshape(1, -1))[0]
                
                # Calculate predicted landmark position
                predicted_x = x + displacement[0]
                predicted_y = y + displacement[1]
                
                # Cast vote(s) based on method
                if voting_method == 'single':
                    # Single unit vote at predicted position
                    pred_x_int = int(round(predicted_x))
                    pred_y_int = int(round(predicted_y))
                    
                    if 0 <= pred_x_int < w and 0 <= pred_y_int < h:
                        vote_map[pred_y_int, pred_x_int] += 1
                        vote_count += 1
                        
                elif voting_method == 'gaussian':
                    # Gaussian spread of votes around predicted position
                    # This accounts for uncertainty in the prediction
                    self._add_gaussian_vote(vote_map, predicted_x, predicted_y, 
                                           self.vote_sigma)
                    vote_count += 1
                    
                elif voting_method == 'weighted':
                    # Weight by prediction confidence (using variance from trees)
                    # Get predictions from individual trees
                    tree_predictions = np.array([
                        tree.predict(features.reshape(1, -1))[0] 
                        for tree in rf_model.estimators_
                    ])
                    
                    # Calculate variance as measure of uncertainty
                    variance = np.var(tree_predictions, axis=0)
                    weight = 1.0 / (1.0 + np.sum(variance))  # Lower variance = higher weight
                    
                    pred_x_int = int(round(predicted_x))
                    pred_y_int = int(round(predicted_y))
                    
                    if 0 <= pred_x_int < w and 0 <= pred_y_int < h:
                        vote_map[pred_y_int, pred_x_int] += weight
                        vote_count += 1
        
        print(f"  Cast {vote_count} votes")
        
        # Smooth the vote map slightly to handle discretization
        if voting_method == 'single':
            vote_map = cv2.GaussianBlur(vote_map, (5, 5), 1.0)
        
        # Find peak in vote map
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(vote_map)
        predicted_position = max_loc  # (x, y)
        
        print(f"  Detected position: ({predicted_position[0]:.1f}, {predicted_position[1]:.1f})")
        print(f"  Max votes: {max_val:.2f}")
        
        return predicted_position, vote_map
    
    def _add_gaussian_vote(self, vote_map, center_x, center_y, sigma):
        """
        Add a Gaussian-weighted vote centered at (center_x, center_y).
        """
        h, w = vote_map.shape
        
        # Define region of influence (3 sigma radius)
        radius = int(3 * sigma)
        x_min = max(0, int(center_x - radius))
        x_max = min(w, int(center_x + radius + 1))
        y_min = max(0, int(center_y - radius))
        y_max = min(h, int(center_y + radius + 1))
        
        # Add Gaussian weights
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                dist_sq = (x - center_x)**2 + (y - center_y)**2
                weight = np.exp(-dist_sq / (2 * sigma**2))
                vote_map[y, x] += weight
    
    def detect_all_landmarks(self, image, step_size=3):
        """
        Detect all trained landmarks in an image.
        
        Returns:
            Dictionary of landmark positions and vote maps
        """
        results = {}
        
        for landmark_key in self.trainer.rf_models.keys():
            position, vote_map = self.detect_landmark(
                image, landmark_key, step_size=step_size
            )
            results[landmark_key] = {
                'position': position,
                'vote_map': vote_map
            }
        
        return results
    
    def visualize_detection(self, image, landmark_key, true_position=None):
        """
        Visualize the detection process and results.
        """
        # Detect landmark
        predicted_pos, vote_map = self.detect_landmark(
            image, landmark_key, step_size=3, voting_method='gaussian'
        )
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image with detection
        axes[0].imshow(image, cmap='gray')
        axes[0].plot(predicted_pos[0], predicted_pos[1], 'r*', 
                    markersize=15, label='Predicted')
        if true_position is not None:
            axes[0].plot(true_position[0], true_position[1], 'g+', 
                        markersize=15, label='True')
            # Calculate error
            error = np.sqrt((predicted_pos[0] - true_position[0])**2 + 
                          (predicted_pos[1] - true_position[1])**2)
            axes[0].set_title(f'Detection (Error: {error:.1f} pixels)')
        else:
            axes[0].set_title('Detection Result')
        axes[0].legend()
        axes[0].axis('off')
        
        # Vote map (heatmap)
        im = axes[1].imshow(vote_map, cmap='hot')
        axes[1].plot(predicted_pos[0], predicted_pos[1], 'b*', markersize=10)
        axes[1].set_title(f'Vote Map for {landmark_key}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Overlay: Image with vote map
        axes[2].imshow(image, cmap='gray', alpha=0.7)
        axes[2].imshow(vote_map, cmap='hot', alpha=0.3)
        axes[2].plot(predicted_pos[0], predicted_pos[1], 'r*', markersize=15)
        if true_position is not None:
            axes[2].plot(true_position[0], true_position[1], 'g+', markersize=15)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(f"RFRV Detection for Landmark '{landmark_key}'")
        plt.tight_layout()
        plt.show()
    
    def evaluate_accuracy(self, test_data):
        """
        Evaluate detection accuracy on test data.
        """
        errors = {key: [] for key in self.trainer.rf_models.keys()}
        
        print("\nEvaluating detection accuracy...")
        
        for sample in test_data:
            image = sample['enhanced_image']
            
            for landmark_key in self.trainer.rf_models.keys():
                if landmark_key not in sample['normalized_landmarks']:
                    continue
                
                # Detect landmark
                predicted_pos, _ = self.detect_landmark(
                    image, landmark_key, step_size=5  # Larger step for speed
                )
                
                # Calculate error
                true_pos = sample['normalized_landmarks'][landmark_key]
                error = np.sqrt((predicted_pos[0] - true_pos[0])**2 + 
                              (predicted_pos[1] - true_pos[1])**2)
                errors[landmark_key].append(error)
        
        # Calculate statistics
        print("\nDetection Accuracy Results:")
        print("-" * 50)
        
        for landmark_key in sorted(errors.keys()):
            if errors[landmark_key]:
                err_array = np.array(errors[landmark_key])
                print(f"{landmark_key:15s}: Mean={err_array.mean():.2f} px, "
                      f"Median={np.median(err_array):.2f} px, "
                      f"Std={err_array.std():.2f} px")
        
        return errors