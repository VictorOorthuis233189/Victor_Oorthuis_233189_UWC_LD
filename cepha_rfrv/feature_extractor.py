import numpy as np
import cv2
import matplotlib.pyplot as plt

class HaarFeatureExtractor:
    """
    Extracts Haar-like features as described in the paper.
    "We use Haar-like features sampled in a box around the current point,
    as they have been found to be effective for a range of applications 
    and can be calculated efficiently from integral images."
    """
    
    def __init__(self, patch_size=24, num_features=100):
        """
        Initialize the Haar feature extractor.
        
        Args:
            patch_size: Size of the patch around each point (paper uses 13x13 to 24x24)
            num_features: Number of random Haar features to generate
        """
        self.patch_size = patch_size
        self.num_features = num_features
        self.features = []
        self.generate_random_haar_features()
        
    def generate_random_haar_features(self):
        """
        Generate random Haar-like feature configurations.
        We'll use three types:
        1. Two-rectangle features (horizontal and vertical)
        2. Three-rectangle features
        3. Four-rectangle features
        """
        self.features = []
        
        for i in range(self.num_features):
            # Randomly choose feature type
            feature_type = np.random.choice(['two_horizontal', 'two_vertical', 
                                            'three_horizontal', 'three_vertical', 
                                            'four'])
            
            # Generate random positions within patch
            x1 = np.random.randint(0, self.patch_size - 2)
            y1 = np.random.randint(0, self.patch_size - 2)
            
            # Ensure we have at least 2x2 rectangles
            max_width = self.patch_size - x1
            max_height = self.patch_size - y1
            
            width = np.random.randint(2, min(max_width, self.patch_size//2) + 1)
            height = np.random.randint(2, min(max_height, self.patch_size//2) + 1)
            
            self.features.append({
                'type': feature_type,
                'x': x1,
                'y': y1,
                'width': width,
                'height': height
            })
    
    def compute_integral_image(self, image):
        """
        Compute integral image for fast Haar feature calculation.
        Integral image at (x,y) contains sum of all pixels above and to the left.
        """
        return cv2.integral(image)
    
    def extract_patch(self, image, x, y):
        """
        Extract a patch from the image centered at (x, y).
        Handles boundary cases by padding with zeros.
        """
        half_size = self.patch_size // 2
        h, w = image.shape
        
        # Calculate patch boundaries
        x_start = int(x - half_size)
        x_end = int(x + half_size)
        y_start = int(y - half_size)
        y_end = int(y + half_size)
        
        # Create patch with zero padding for out-of-bounds areas
        patch = np.zeros((self.patch_size, self.patch_size), dtype=image.dtype)
        
        # Calculate valid region to copy
        valid_x_start = max(0, x_start)
        valid_x_end = min(w, x_end)
        valid_y_start = max(0, y_start)
        valid_y_end = min(h, y_end)
        
        # Calculate where to place in patch
        patch_x_start = valid_x_start - x_start
        patch_x_end = patch_x_start + (valid_x_end - valid_x_start)
        patch_y_start = valid_y_start - y_start
        patch_y_end = patch_y_start + (valid_y_end - valid_y_start)
        
        # Copy valid region
        if valid_x_end > valid_x_start and valid_y_end > valid_y_start:
            patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = \
                image[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
        
        return patch
    
    def compute_haar_feature(self, integral_patch, feature):
        """
        Compute a single Haar feature value from an integral image patch.
        """
        x, y = feature['x'], feature['y']
        w, h = feature['width'], feature['height']
        
        def rect_sum(x1, y1, x2, y2):
            """Sum of pixels in rectangle using integral image."""
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(integral_patch.shape[1]-1, x2), min(integral_patch.shape[0]-1, y2)
            
            if x1 >= x2 or y1 >= y2:
                return 0
                
            return (integral_patch[y2, x2] - 
                   integral_patch[y1, x2] - 
                   integral_patch[y2, x1] + 
                   integral_patch[y1, x1])
        
        if feature['type'] == 'two_horizontal':
            # White rectangle - black rectangle (horizontal)
            white = rect_sum(x, y, x + w//2, y + h)
            black = rect_sum(x + w//2, y, x + w, y + h)
            return white - black
            
        elif feature['type'] == 'two_vertical':
            # White rectangle - black rectangle (vertical)
            white = rect_sum(x, y, x + w, y + h//2)
            black = rect_sum(x, y + h//2, x + w, y + h)
            return white - black
            
        elif feature['type'] == 'three_horizontal':
            # White - black - white (horizontal)
            white1 = rect_sum(x, y, x + w//3, y + h)
            black = rect_sum(x + w//3, y, x + 2*w//3, y + h)
            white2 = rect_sum(x + 2*w//3, y, x + w, y + h)
            return white1 - black + white2
            
        elif feature['type'] == 'three_vertical':
            # White - black - white (vertical)
            white1 = rect_sum(x, y, x + w, y + h//3)
            black = rect_sum(x, y + h//3, x + w, y + 2*h//3)
            white2 = rect_sum(x, y + 2*h//3, x + w, y + h)
            return white1 - black + white2
            
        elif feature['type'] == 'four':
            # Checkerboard pattern
            white1 = rect_sum(x, y, x + w//2, y + h//2)
            black1 = rect_sum(x + w//2, y, x + w, y + h//2)
            black2 = rect_sum(x, y + h//2, x + w//2, y + h)
            white2 = rect_sum(x + w//2, y + h//2, x + w, y + h)
            return white1 - black1 - black2 + white2
        
        return 0
    
    def extract_features(self, image, x, y):
        """
        Extract all Haar features for a point (x, y) in the image.
        
        Returns:
            feature_vector: Array of feature values
        """
        # Extract patch
        patch = self.extract_patch(image, x, y)
        
        # Compute integral image of patch
        integral_patch = self.compute_integral_image(patch)
        
        # Compute all Haar features
        feature_values = []
        for feature in self.features:
            value = self.compute_haar_feature(integral_patch, feature)
            feature_values.append(value)
        
        return np.array(feature_values)
    
    def visualize_features(self, image, x, y, num_to_show=9):
        """
        Visualize the Haar features being extracted at a point.
        """
        patch = self.extract_patch(image, x, y)
        
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        axes = axes.flatten()
        
        for i in range(min(num_to_show, len(self.features))):
            ax = axes[i]
            ax.imshow(patch, cmap='gray')
            
            feature = self.features[i]
            
            # Draw rectangles to show Haar feature
            from matplotlib.patches import Rectangle
            
            if feature['type'] in ['two_horizontal', 'two_vertical']:
                # Draw the feature regions
                rect1 = Rectangle((feature['x'], feature['y']), 
                                 feature['width'], feature['height'],
                                 linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect1)
            
            ax.set_title(f"{feature['type']}", fontsize=8)
            ax.axis('off')
        
        plt.suptitle(f"Haar Features at point ({x:.0f}, {y:.0f})")
        plt.tight_layout()
        plt.show()