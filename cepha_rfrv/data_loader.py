import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

class CephalometricDataLoader:
    """
    Loads and preprocesses cephalometric images and their landmark annotations.
    Following the paper's approach, we need to:
    1. Load images and annotations
    2. Normalize images to a reference frame
    3. Extract landmarks in a consistent format
    """
    
    def __init__(self, annotations_path, images_path):
        self.annotations_path = Path(annotations_path)
        self.images_path = Path(images_path)
        self.data = []
        
    def load_data(self, max_samples=None):
        """
        Load all images and their corresponding annotations.
        Returns a list of dictionaries with image data and landmarks.
        """
        annotation_files = list(self.annotations_path.glob("*.json"))
        
        if max_samples:
            annotation_files = annotation_files[:max_samples]
            
        print(f"Loading {len(annotation_files)} samples...")
        
        for ann_file in tqdm(annotation_files):
            # Load annotation
            with open(ann_file, 'r') as f:
                ann_data = json.load(f)
            
            # Get corresponding image
            ceph_id = ann_data['ceph_id']
            
            # Try different image extensions
            image_file = None
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                potential_file = self.images_path / f"{ceph_id}{ext}"
                if potential_file.exists():
                    image_file = potential_file
                    break
            
            if image_file is None:
                print(f"Warning: Image not found for ceph_id {ceph_id}")
                continue
            
            # Load image
            image = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Could not load image {image_file}")
                continue
            
            # Extract landmarks into a more convenient format
            landmarks = {}
            for landmark in ann_data['landmarks']:
                landmarks[landmark['symbol']] = np.array([
                    landmark['value']['x'],
                    landmark['value']['y']
                ])
            
            self.data.append({
                'ceph_id': ceph_id,
                'image': image,
                'landmarks': landmarks,
                'image_path': str(image_file),
                'original_shape': image.shape
            })
        
        print(f"Successfully loaded {len(self.data)} samples")
        return self.data
    
    def visualize_sample(self, idx=0):
        """
        Visualize a sample image with its landmarks.
        This helps us understand our data before processing.
        """
        if not self.data:
            print("No data loaded yet. Call load_data() first.")
            return
        
        sample = self.data[idx]
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Display image
        ax.imshow(sample['image'], cmap='gray')
        
        # Plot landmarks
        for symbol, coords in sample['landmarks'].items():
            ax.plot(coords[0], coords[1], 'ro', markersize=3)
            ax.text(coords[0] + 10, coords[1], symbol, 
                   color='yellow', fontsize=8, fontweight='bold')
        
        ax.set_title(f"Cephalogram {sample['ceph_id']} with {len(sample['landmarks'])} landmarks")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    def get_statistics(self):
        """
        Get statistics about the dataset.
        Important for understanding image sizes and landmark distributions.
        """
        if not self.data:
            print("No data loaded yet. Call load_data() first.")
            return
        
        # Image size statistics
        heights = [d['original_shape'][0] for d in self.data]
        widths = [d['original_shape'][1] for d in self.data]
        
        # Landmark statistics
        all_landmarks = set()
        for d in self.data:
            all_landmarks.update(d['landmarks'].keys())
        
        stats = {
            'num_samples': len(self.data),
            'image_heights': {
                'min': min(heights),
                'max': max(heights),
                'mean': np.mean(heights),
                'std': np.std(heights)
            },
            'image_widths': {
                'min': min(widths),
                'max': max(widths),
                'mean': np.mean(widths),
                'std': np.std(widths)
            },
            'landmarks': sorted(list(all_landmarks)),
            'num_landmarks': len(all_landmarks)
        }
        
        return stats