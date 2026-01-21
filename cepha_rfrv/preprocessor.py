import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class ImagePreprocessor:
    """
    Preprocesses images following the RFRV paper approach:
    1. Normalize images to a reference frame (standardized size)
    2. Apply histogram equalization for better feature extraction
    3. Create a shape model for initialization
    """
    
    def __init__(self, reference_width=800):
        """
        Initialize preprocessor.
        
        Args:
            reference_width: Target width for normalized images (paper uses ~70-210 pixels,
                           we'll use 800 for cephalograms which are larger)
        """
        self.reference_width = reference_width
        self.mean_shape = None
        self.shape_model = None
        
    def normalize_image(self, image, landmarks):
        """
        Normalize image to reference frame.
        Following the paper: "Each image is resampled into a standardized reference frame"
        
        Args:
            image: Input grayscale image
            landmarks: Dictionary of landmark positions
            
        Returns:
            normalized_image: Resized image
            normalized_landmarks: Scaled landmark positions
            scale_factor: Scale factor used
        """
        h, w = image.shape
        
        # Calculate scale factor to make width = reference_width
        scale_factor = self.reference_width / w
        
        # Calculate new dimensions
        new_width = self.reference_width
        new_height = int(h * scale_factor)
        
        # Resize image
        normalized_image = cv2.resize(image, (new_width, new_height), 
                                     interpolation=cv2.INTER_LINEAR)
        
        # Scale landmarks
        normalized_landmarks = {}
        for key, coords in landmarks.items():
            normalized_landmarks[key] = coords * scale_factor
        
        return normalized_image, normalized_landmarks, scale_factor
    
    def enhance_image(self, image):
        """
        Apply image enhancement for better feature extraction.
        CLAHE (Contrast Limited Adaptive Histogram Equalization) helps with
        varying contrast in X-ray images.
        """
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Apply CLAHE
        enhanced = clahe.apply(image)
        
        return enhanced
    
    def compute_mean_shape(self, data_list):
        """
        Compute the mean shape from all training samples.
        This is used as initialization for the model fitting process.
        Following paper: "The model is scaled so that the width of the 
        bounding box of the mean shape is a given value"
        """
        all_landmarks = []
        
        # Get all landmarks that are present in ALL samples
        common_landmarks = None
        for sample in data_list:
            # Use normalized_landmarks for preprocessed data
            landmark_dict = sample.get('normalized_landmarks', sample.get('landmarks', {}))
            if common_landmarks is None:
                common_landmarks = set(landmark_dict.keys())
            else:
                common_landmarks = common_landmarks.intersection(landmark_dict.keys())
        
        print(f"Common landmarks across all samples: {sorted(common_landmarks)}")
        
        # Collect all landmark positions
        for sample in data_list:
            # Use normalized_landmarks for preprocessed data
            landmark_dict = sample.get('normalized_landmarks', sample.get('landmarks', {}))
            landmarks_array = []
            for landmark_key in sorted(common_landmarks):
                if landmark_key in landmark_dict:
                    landmarks_array.append(landmark_dict[landmark_key])
            if landmarks_array:
                all_landmarks.append(np.array(landmarks_array))
        
        if not all_landmarks:
            print("Warning: No common landmarks found")
            return None
        
        # Stack and compute mean
        all_landmarks = np.array(all_landmarks)
        self.mean_shape = np.mean(all_landmarks, axis=0)
        
        # Also store which landmarks we're using
        self.landmark_keys = sorted(common_landmarks)
        
        return self.mean_shape, self.landmark_keys
    
    def preprocess_dataset(self, data_list):
        """
        Preprocess entire dataset.
        
        Returns:
            List of preprocessed samples with normalized images and landmarks
        """
        preprocessed_data = []
        
        print("Preprocessing dataset...")
        for sample in tqdm(data_list):
            # Normalize to reference frame
            norm_img, norm_landmarks, scale = self.normalize_image(
                sample['image'], 
                sample['landmarks']
            )
            
            # Enhance image
            enhanced_img = self.enhance_image(norm_img)
            
            preprocessed_data.append({
                'ceph_id': sample['ceph_id'],
                'original_image': sample['image'],
                'normalized_image': norm_img,
                'enhanced_image': enhanced_img,
                'original_landmarks': sample['landmarks'],
                'normalized_landmarks': norm_landmarks,
                'scale_factor': scale,
                'original_shape': sample['original_shape']
            })
        
        # Compute mean shape from normalized landmarks
        self.compute_mean_shape(preprocessed_data)
        
        return preprocessed_data
    
    def visualize_preprocessing(self, original_sample, preprocessed_sample):
        """
        Visualize the preprocessing effects.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_sample['image'], cmap='gray')
        axes[0].set_title('Original Image')
        for key, coords in original_sample['landmarks'].items():
            axes[0].plot(coords[0], coords[1], 'ro', markersize=2)
        axes[0].axis('off')
        
        # Normalized image
        axes[1].imshow(preprocessed_sample['normalized_image'], cmap='gray')
        axes[1].set_title('Normalized Image')
        for key, coords in preprocessed_sample['normalized_landmarks'].items():
            axes[1].plot(coords[0], coords[1], 'go', markersize=2)
        axes[1].axis('off')
        
        # Enhanced image
        axes[2].imshow(preprocessed_sample['enhanced_image'], cmap='gray')
        axes[2].set_title('Enhanced (CLAHE) Image')
        for key, coords in preprocessed_sample['normalized_landmarks'].items():
            axes[2].plot(coords[0], coords[1], 'bo', markersize=2)
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
