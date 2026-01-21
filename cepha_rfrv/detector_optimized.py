import numpy as np
import cv2
from joblib import Parallel, delayed
import multiprocessing
import time
from tqdm import tqdm

class OptimizedRFRVDetector:
    """
    Optimized RFRV Detector with parallel processing.
    Much faster than the original sequential version.
    """
    
    def __init__(self, trainer, feature_extractor, vote_sigma=3.0):
        """
        Initialize the optimized detector.
        
        Args:
            trainer: Trained RFRVTrainer with RF models
            feature_extractor: HaarFeatureExtractor instance
            vote_sigma: Standard deviation for Gaussian voting (pixels)
        """
        self.trainer = trainer
        self.feature_extractor = feature_extractor
        self.vote_sigma = vote_sigma
        self.n_jobs = max(1, multiprocessing.cpu_count() - 1)  # Use all cores except 1
        
    def process_image_chunk(self, image, x_coords, y_coord, rf_model):
        """
        Process a chunk of the image (one row of sample points).
        This function will be called in parallel.
        
        Args:
            image: Input image
            x_coords: List of x coordinates to process
            y_coord: The y coordinate for this row
            rf_model: The trained Random Forest model
            
        Returns:
            List of predicted landmark positions
        """
        predictions = []
        
        for x in x_coords:
            # Extract features at this point
            features = self.feature_extractor.extract_features(image, x, y_coord)
            
            # Predict displacement to landmark
            displacement = rf_model.predict(features.reshape(1, -1))[0]
            
            # Calculate predicted landmark position
            predicted_x = x + displacement[0]
            predicted_y = y_coord + displacement[1]
            
            predictions.append((predicted_x, predicted_y))
        
        return predictions
    
    def detect_landmark_parallel(self, image, landmark_key, 
                                search_region=None, step_size=5):
        """
        Detect landmark using parallel processing.
        Much faster than sequential scanning.
        
        Args:
            image: Input image (preprocessed)
            landmark_key: Which landmark to detect
            search_region: (x_min, y_min, x_max, y_max) or None for full image
            step_size: Spacing between sample points
            
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
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
        
        # Create sampling grid
        x_coords = list(range(x_min, x_max, step_size))
        y_coords = list(range(y_min, y_max, step_size))
        
        print(f"Detecting '{landmark_key}' with parallel processing...")
        print(f"  Using {self.n_jobs} CPU cores")
        print(f"  Grid size: {len(x_coords)} x {len(y_coords)} = {len(x_coords)*len(y_coords)} points")
        
        start_time = time.time()
        
        # Process rows in parallel
        all_predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(self.process_image_chunk)(image, x_coords, y, rf_model)
            for y in tqdm(y_coords, desc="Processing rows")
        )
        
        # Flatten predictions
        predictions = [pred for row_preds in all_predictions for pred in row_preds]
        
        # Create vote map
        vote_map = np.zeros((h, w), dtype=np.float32)
        
        # Add votes
        for pred_x, pred_y in predictions:
            if self.vote_sigma > 0:
                self._add_gaussian_vote(vote_map, pred_x, pred_y, self.vote_sigma)
            else:
                # Simple vote (faster but less smooth)
                pred_x_int = int(round(pred_x))
                pred_y_int = int(round(pred_y))
                if 0 <= pred_x_int < w and 0 <= pred_y_int < h:
                    vote_map[pred_y_int, pred_x_int] += 1
        
        # Smooth if using simple voting
        if self.vote_sigma == 0:
            vote_map = cv2.GaussianBlur(vote_map, (5, 5), 1.0)
        
        # Find peak in vote map
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(vote_map)
        predicted_position = max_loc  # (x, y)
        
        elapsed = time.time() - start_time
        print(f"  Detection completed in {elapsed:.1f} seconds")
        print(f"  Detected position: {predicted_position}")
        print(f"  Peak confidence: {max_val:.2f}")
        
        return predicted_position, vote_map
    
    def detect_landmark_vectorized(self, image, landmark_key, 
                                  search_region=None, step_size=5,
                                  batch_size=1000):
        """
        Alternative: Vectorized detection using batch prediction.
        Processes multiple points at once for better efficiency.
        
        Args:
            image: Input image
            landmark_key: Landmark to detect
            search_region: Region to search in
            step_size: Grid spacing
            batch_size: Number of points to process at once
            
        Returns:
            predicted_position: (x, y) coordinates
            vote_map: Voting heatmap
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
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)
        
        # Create sampling grid
        x_coords = np.arange(x_min, x_max, step_size)
        y_coords = np.arange(y_min, y_max, step_size)
        
        print(f"Detecting '{landmark_key}' with vectorized processing...")
        print(f"  Grid size: {len(x_coords)} x {len(y_coords)} = {len(x_coords)*len(y_coords)} points")
        print(f"  Batch size: {batch_size}")
        
        start_time = time.time()
        
        # Create mesh grid
        xx, yy = np.meshgrid(x_coords, y_coords)
        sample_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        # Process in batches to avoid memory issues
        all_predictions = []
        
        for i in tqdm(range(0, len(sample_points), batch_size), desc="Processing batches"):
            batch_points = sample_points[i:i+batch_size]
            batch_features = []
            
            # Extract features for batch
            for point in batch_points:
                features = self.feature_extractor.extract_features(image, point[0], point[1])
                batch_features.append(features)
            
            # Predict displacements for batch
            batch_features = np.array(batch_features)
            batch_predictions = rf_model.predict(batch_features)
            
            # Calculate predicted positions
            predicted_positions = batch_points + batch_predictions
            all_predictions.extend(predicted_positions)
        
        # Create vote map
        vote_map = np.zeros((h, w), dtype=np.float32)
        
        # Add all votes
        for pred_x, pred_y in all_predictions:
            if self.vote_sigma > 0:
                self._add_gaussian_vote(vote_map, pred_x, pred_y, self.vote_sigma)
            else:
                pred_x_int = int(round(pred_x))
                pred_y_int = int(round(pred_y))
                if 0 <= pred_x_int < w and 0 <= pred_y_int < h:
                    vote_map[pred_y_int, pred_x_int] += 1
        
        # Smooth if needed
        if self.vote_sigma == 0:
            vote_map = cv2.GaussianBlur(vote_map, (5, 5), 1.0)
        
        # Find peak
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(vote_map)
        predicted_position = max_loc
        
        elapsed = time.time() - start_time
        print(f"  Detection completed in {elapsed:.1f} seconds")
        
        return predicted_position, vote_map
    
    def coarse_to_fine_detection(self, image, landmark_key, 
                                coarse_step=15, fine_step=3, 
                                refinement_radius=30):
        """
        Two-stage detection: coarse then fine.
        Much faster than single-stage with small step size.
        
        Args:
            image: Input image
            landmark_key: Landmark to detect
            coarse_step: Step size for initial coarse search
            fine_step: Step size for fine search
            refinement_radius: Radius around coarse detection for fine search
            
        Returns:
            predicted_position: Final detected position
            vote_map: Final vote map from fine detection
        """
        print(f"\n=== Coarse-to-Fine Detection for '{landmark_key}' ===")
        
        # Stage 1: Coarse search with large steps
        print("Stage 1: Coarse detection...")
        coarse_pos, coarse_map = self.detect_landmark_parallel(
            image, landmark_key, step_size=coarse_step
        )
        
        # Stage 2: Fine search around coarse position
        print(f"Stage 2: Fine detection around {coarse_pos}...")
        search_region = (
            max(0, coarse_pos[0] - refinement_radius),
            max(0, coarse_pos[1] - refinement_radius),
            min(image.shape[1], coarse_pos[0] + refinement_radius),
            min(image.shape[0], coarse_pos[1] + refinement_radius)
        )
        
        fine_pos, fine_map = self.detect_landmark_parallel(
            image, landmark_key, 
            search_region=search_region,
            step_size=fine_step
        )
        
        print(f"Final position: {fine_pos}")
        
        return fine_pos, fine_map
    
    def detect_all_landmarks_parallel(self, image, use_coarse_to_fine=True, 
                                     step_size=5):
        """
        Detect all trained landmarks in parallel.
        
        Args:
            image: Input image
            use_coarse_to_fine: Whether to use two-stage detection
            step_size: Step size if not using coarse-to-fine
            
        Returns:
            Dictionary of landmark positions and vote maps
        """
        results = {}
        
        print(f"\nDetecting {len(self.trainer.rf_models)} landmarks...")
        
        for landmark_key in self.trainer.rf_models.keys():
            if use_coarse_to_fine:
                position, vote_map = self.coarse_to_fine_detection(
                    image, landmark_key
                )
            else:
                position, vote_map = self.detect_landmark_parallel(
                    image, landmark_key, step_size=step_size
                )
            
            results[landmark_key] = {
                'position': position,
                'vote_map': vote_map
            }
        
        return results
    
    def _add_gaussian_vote(self, vote_map, center_x, center_y, sigma):
        """
        Add a Gaussian-weighted vote centered at (center_x, center_y).
        
        Args:
            vote_map: Accumulator array to add votes to
            center_x: X coordinate of vote center
            center_y: Y coordinate of vote center
            sigma: Standard deviation of Gaussian
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
    
    def evaluate_accuracy(self, test_data, use_coarse_to_fine=True):
        """
        Evaluate detection accuracy on test data.
        
        Args:
            test_data: List of preprocessed test samples
            use_coarse_to_fine: Whether to use two-stage detection
            
        Returns:
            Dictionary of errors per landmark
        """
        errors = {key: [] for key in self.trainer.rf_models.keys()}
        
        print("\nEvaluating detection accuracy...")
        print(f"Test samples: {len(test_data)}")
        print(f"Method: {'Coarse-to-fine' if use_coarse_to_fine else 'Single-stage'}")
        
        for i, sample in enumerate(test_data):
            print(f"\nProcessing test sample {i+1}/{len(test_data)}...")
            image = sample['enhanced_image']
            
            for landmark_key in self.trainer.rf_models.keys():
                if landmark_key not in sample['normalized_landmarks']:
                    continue
                
                # Detect landmark
                if use_coarse_to_fine:
                    predicted_pos, _ = self.coarse_to_fine_detection(
                        image, landmark_key
                    )
                else:
                    predicted_pos, _ = self.detect_landmark_parallel(
                        image, landmark_key, step_size=5
                    )
                
                # Calculate error
                true_pos = sample['normalized_landmarks'][landmark_key]
                error = np.sqrt((predicted_pos[0] - true_pos[0])**2 + 
                              (predicted_pos[1] - true_pos[1])**2)
                errors[landmark_key].append(error)
        
        # Calculate statistics
        print("\n" + "="*60)
        print("Detection Accuracy Results:")
        print("="*60)
        print(f"{'Landmark':<15} {'Mean±Std':<20} {'Median':<10} {'Max':<10}")
        print("-"*60)
        
        for landmark_key in sorted(errors.keys()):
            if errors[landmark_key]:
                err_array = np.array(errors[landmark_key])
                print(f"{landmark_key:<15} "
                      f"{err_array.mean():.2f}±{err_array.std():.2f} px"
                      f"{' '*5}{np.median(err_array):>7.2f} px"
                      f"{np.max(err_array):>10.2f} px")
        
        # Overall statistics
        all_errors = np.concatenate([e for e in errors.values() if e])
        print("-"*60)
        print(f"{'OVERALL':<15} "
              f"{all_errors.mean():.2f}±{all_errors.std():.2f} px"
              f"{' '*5}{np.median(all_errors):>7.2f} px"
              f"{np.max(all_errors):>10.2f} px")
        print("="*60)
        
        return errors
    
    def benchmark_methods(self, image, landmark_key):
        """
        Compare different detection methods for speed.
        
        Args:
            image: Test image
            landmark_key: Landmark to detect
            
        Returns:
            Dictionary of timing results
        """
        print(f"\nBenchmarking detection methods for '{landmark_key}'...")
        print("-"*50)
        
        results = {}
        
        # Method 1: Parallel with step=10
        start = time.time()
        pos1, _ = self.detect_landmark_parallel(image, landmark_key, step_size=10)
        results['parallel_step10'] = {
            'time': time.time() - start,
            'position': pos1
        }
        
        # Method 2: Parallel with step=5
        start = time.time()
        pos2, _ = self.detect_landmark_parallel(image, landmark_key, step_size=5)
        results['parallel_step5'] = {
            'time': time.time() - start,
            'position': pos2
        }
        
        # Method 3: Vectorized
        start = time.time()
        pos3, _ = self.detect_landmark_vectorized(image, landmark_key, step_size=10)
        results['vectorized'] = {
            'time': time.time() - start,
            'position': pos3
        }
        
        # Method 4: Coarse-to-fine
        start = time.time()
        pos4, _ = self.coarse_to_fine_detection(image, landmark_key)
        results['coarse_to_fine'] = {
            'time': time.time() - start,
            'position': pos4
        }
        
        # Display results
        print("\nTiming Results:")
        print("-"*50)
        for method, data in results.items():
            print(f"{method:<20}: {data['time']:>6.2f} seconds  "
                  f"Position: {data['position']}")
        
        # Find fastest
        fastest = min(results.keys(), key=lambda k: results[k]['time'])
        print(f"\nFastest method: {fastest} ({results[fastest]['time']:.2f}s)")
        
        return results