import os
import re
import json
import math
import torch
import numpy as np
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class KeymakrSegmentation(Dataset):
    """
    Custom dataset for Keymakr annotation format with colored segmentation masks.
    
    Dataset structure:
    root_dir/
        ├── sequence1.json
        ├── sequence1.images/
        │   ├── 00001/
        │   │   ├── all.png      # RGB image
        │   │   ├── 0.1.png      # Individual object masks (optional)
        │   │   ├── 0.2.png
        │   │   └── ...
        │   └── 00002/
        │       └── ...
        ├── sequence2.json
        ├── sequence2.images/
        └── ...
    
    The JSON files contain annotation metadata with color-to-type mappings.
    The mask images are colored where each color represents a specific object type.
    """
    
    def __init__(self, root_dir, image_set='train', transforms=None, val_split=0.2, random_seed=42, class_mapping={}, return_paths=False):
        """
        Args:
            root_dir (string): Root directory containing JSON annotation files and .images folders
            image_set (string): 'train' or 'val' for dataset split
            transforms (callable, optional): Optional transform to be applied on samples
            val_split (float): Fraction of data to use for validation (default: 0.2)
            random_seed (int): Random seed for reproducible train/val splits
            class_mapping (dict): Optional conversion of classes in the dataset to target classes
            return_paths (bool): If True, __getitem__ returns (image, target, image_path)
        """
        self.root_dir = root_dir
        self.image_set = image_set
        self.transforms = transforms
        self.val_split = val_split
        self.return_paths = return_paths
        
        # Initialize class mapping
        self.color_to_class = {}
        self.class_to_index = {}
        self.index_to_class = {}
        self.index_to_color = {}
        self.class_mapping = class_mapping  # User-defined class remapping
        self.num_classes = 0
        
        # Storage for image paths and annotation data
        self.images = []
        self.targets = []
        self.annotations = []
        
        # Build global color mapping and collect all data
        self._build_global_mapping()
        self._collect_data()
        self._split_data(random_seed)
        
        print(f"Keymakr Dataset initialized:")
        print(f"  - {len(self.images)} images in {image_set} set")
        print(f"  - {self.num_classes} classes found")
        print(f"  - Classes: {list(self.class_to_index.keys())}")
    
    def _build_global_mapping(self):
        """
        Scan all JSON files to build a global mapping from colors to class types.
        This ensures consistent class indices across all sequences.
        If class_mapping is provided, remaps everything to user classes.
        """
        print("Building global color-to-class mapping...")
        
        type_to_colors = defaultdict(set)
        
        # Scan all JSON files
        for filename in os.listdir(self.root_dir):
            if not filename.endswith('.json'):
                continue
                
            json_path = os.path.join(self.root_dir, filename)
            
            try:
                with open(json_path, 'r') as f:
                    annotated_frames = json.load(f)
                
                # Extract type-color mappings
                for annotation in annotated_frames:
                    for obj in annotation.get('objects', []):
                        obj_type = obj.get('type', 'background')
                        color = obj.get('color', '#000000')
                        type_to_colors[obj_type].add(color)
                    
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not read {self._get_relative_path(json_path)}: {e}")
                continue
        
        # Verify consistency: each type should map to exactly one color
        original_color_to_class = {}
        for obj_type, colors in type_to_colors.items():
            if len(colors) > 1:
                print(f"Warning: Type '{obj_type}' has multiple colors: {colors}")
                print("Using the first color found.")
            
            # Use the first color for this type
            color = list(colors)[0]
            original_color_to_class[color] = obj_type
        
        # If class mapping is provided, remap everything to user classes
        if self.class_mapping:
            print("Applying class mapping to create user class mappings...")
            
            # Remap colors to user classes
            self.color_to_class = {}
            user_class_to_colors = defaultdict(list)
            
            for color, original_class in original_color_to_class.items():
                # Only remap if explicitly mapped, otherwise keep original class
                user_class = self.class_mapping.get(original_class, original_class)
                self.color_to_class[color] = user_class
                user_class_to_colors[user_class].append(color)
            
            # Get unique user classes (includes both mapped and unmapped original classes)
            unique_classes = sorted(set(self.color_to_class.values()))
            
            print("User class color assignments:")
            for user_class, colors in user_class_to_colors.items():
                mapped_status = "mapped" if user_class in self.class_mapping.values() else "unmapped"
                print(f"  - {user_class} ({mapped_status}): {colors}")
        else:
            # No mapping, use original classes
            self.color_to_class = original_color_to_class
            unique_classes = sorted(set(self.color_to_class.values()))
        
        # Remove 'background' if it exists, we'll handle it separately
        if 'background' in unique_classes:
            unique_classes.remove('background')
        
        # Build class-to-index mapping (background = 0, classes start from 1)
        self.class_to_index = {cls: idx + 1 for idx, cls in enumerate(unique_classes)}
        self.class_to_index['background'] = 0  # Background class
        
        # Build reverse mapping
        self.index_to_class = {idx: cls for cls, idx in self.class_to_index.items()}
        
        # Build index-to-color mapping (maps class indices to their hex colors)
        # For user classes, use the first available color
        self.index_to_color = {}
        class_to_first_color = {}
        
        for color, class_name in self.color_to_class.items():
            if class_name not in class_to_first_color:
                class_to_first_color[class_name] = color
        
        for class_name, color in class_to_first_color.items():
            class_idx = self.class_to_index.get(class_name, 0)
            self.index_to_color[class_idx] = color
        
        # Ensure background (index 0) has a color
        if 0 not in self.index_to_color:
            self.index_to_color[0] = '#000000'
        
        self.num_classes = len(self.class_to_index)
        
        print(f"Found {len(unique_classes)} classes (+ background)")

        print("Final colour to class mapping:")
        for color, cls in self.color_to_class.items():
            print(f"  - {color} -> {cls}")

        print("Final class to index mapping:")
        for cls, idx in sorted(self.class_to_index.items(), key=lambda x: x[1]):
            print(f"  - {cls} -> {idx}")

        print("Index to color mapping:")
        for idx, color in sorted(self.index_to_color.items()):
            print(f"  - {idx} -> {color}")

    def _collect_data(self):
        """
        Collect all image paths and corresponding annotation data.
        Maps RGB images from *_data_* directories to mask files in *.images directories.
        Multiple RGB images per sequence are paired sequentially with mask frames.
        """
        print("Collecting image and annotation data...")
        
        # First, group all RGB data directories by sequence prefix
        rgb_dirs_by_sequence = defaultdict(list)
        for item in os.listdir(self.root_dir):
            item_path = os.path.join(self.root_dir, item)
            if os.path.isdir(item_path) and '_data_' in item and not item.endswith('.images'):
                # Extract sequence name (part before _data_)
                sequence_base = item.split('_data_')[0]
                rgb_dirs_by_sequence[sequence_base].append(item)
        
        # Sort RGB directories for consistent ordering
        for sequence_base in rgb_dirs_by_sequence:
            rgb_dirs_by_sequence[sequence_base].sort()
        
        # Debug: Print RGB directory groupings
        print("RGB directory groupings:")
        for sequence_base, dirs in rgb_dirs_by_sequence.items():
            print(f"  {sequence_base}: {len(dirs)} directories")
            for dir_name in dirs:
                print(f"    - {dir_name}")
        
        # For each sequence, collect all RGB images and pair with masks sequentially
        for sequence_base, rgb_dirs in rgb_dirs_by_sequence.items():
            # Find corresponding .images directory and .json file
            images_dir = os.path.join(self.root_dir, f"{sequence_base}.images")
            json_path = os.path.join(self.root_dir, f"{sequence_base}.json")
            
            if not os.path.exists(images_dir):
                print(f"Warning: Images directory not found: {self._get_relative_path(images_dir)}")
                continue
            
            if not os.path.exists(json_path):
                print(f"Warning: JSON file not found: {self._get_relative_path(json_path)}")
                continue
            
            try:
                with open(json_path, 'r') as f:
                    annotation = json.load(f)
                
                # Collect all RGB images from all data directories for this sequence
                all_rgb_images = []
                for rgb_dir in rgb_dirs:
                    rgb_dir_path = os.path.join(self.root_dir, rgb_dir)
                    if not os.path.exists(rgb_dir_path):
                        continue
                    
                    rgb_files = [f for f in os.listdir(rgb_dir_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    
                    # Add full paths and sort within this directory
                    rgb_file_paths = [os.path.join(rgb_dir_path, f) for f in rgb_files]
                    rgb_file_paths.sort()  # Sort by filename
                    all_rgb_images.extend(rgb_file_paths)
                
                # Find all frame directories in the .images folder
                frame_dirs = []
                for item in os.listdir(images_dir):
                    frame_path = os.path.join(images_dir, item)
                    if os.path.isdir(frame_path):
                        frame_dirs.append(item)
                
                # Sort frame directories numerically
                frame_dirs = self._sorted_alphanumeric(frame_dirs)
                
                print(f"Sequence {sequence_base}: {len(all_rgb_images)} RGB images, {len(frame_dirs)} mask frames")
                
                # Pair RGB images with mask frames sequentially
                for i, frame_dir in enumerate(frame_dirs):
                    if i >= len(all_rgb_images):
                        print(f"Warning: More mask frames than RGB images for sequence {sequence_base}")
                        break
                    
                    frame_path = os.path.join(images_dir, frame_dir)
                    mask_path = os.path.join(frame_path, 'all.png')
                    
                    if not os.path.exists(mask_path):
                        print(f"Warning: Mask file not found: {self._get_relative_path(mask_path)}")
                        continue
                    
                    rgb_image_path = all_rgb_images[i]
                    
                    # Store RGB image path and mask generation data
                    mask_data = {
                        'annotation': annotation,
                        'frame_path': frame_path,
                        'mask_path': mask_path,
                        'sequence_name': sequence_base,
                        'frame_id': frame_dir
                    }
                    
                    self.images.append(rgb_image_path)  # RGB image path
                    self.targets.append(None)  # Will be generated on-demand
                    self.annotations.append(mask_data)
                    
                    print(f"Paired: {self._get_relative_path(rgb_image_path)} -> {self._get_relative_path(mask_path)}")
                
                # Warn if there are leftover RGB images
                if len(all_rgb_images) > len(frame_dirs):
                    leftover_count = len(all_rgb_images) - len(frame_dirs)
                    print(f"Warning: {leftover_count} RGB images without corresponding mask frames for sequence {sequence_base}")
                        
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not process {self._get_relative_path(json_path)}: {e}")
                continue
        
        print(f"Collected {len(self.images)} RGB-mask pairs")
    
    def _split_data(self, random_seed):
        """
        Split data into train and validation sets.
        """
        if self.val_split <= 0 or self.val_split >= 1:
            # No split needed
            return
            
        # Set random seed for reproducible splits
        np.random.seed(random_seed)
        
        # Create indices and shuffle
        total_samples = len(self.images)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        
        # Calculate split point
        val_size = int(total_samples * self.val_split)
        
        if self.image_set == 'train':
            selected_indices = indices[val_size:]
        elif self.image_set == 'val':
            selected_indices = indices[:val_size]
        else:
            raise ValueError(f"image_set must be 'train' or 'val', got '{self.image_set}'")
        
        # Filter data based on selected indices
        self.images = [self.images[i] for i in selected_indices]
        self.targets = [self.targets[i] for i in selected_indices]
        self.annotations = [self.annotations[i] for i in selected_indices]
    
    def _sorted_alphanumeric(self, data):
        """Sort alphanumeric strings naturally."""
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)
    
    def _get_relative_path(self, path):
        """Get path relative to root_dir for logging purposes."""
        return os.path.relpath(path, self.root_dir)
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _apply_class_mapping(self, mask):
        """
        Apply user-defined class mapping to the generated mask.
        Since remapping is now handled in _build_global_mapping, this just returns the mask as-is.
        """
        return mask  # Class mapping is already applied during color-to-class mapping
    
    def _generate_mask(self, annotation_data):
        """
        Generate a grayscale segmentation mask by mapping pixel colors in all.png to class indices.
        Uses the global color-to-class mapping built during initialization.
        """
        # Use the mask path stored in annotation data
        mask_image_path = annotation_data['mask_path']
        print("Generating mask for:", self._get_relative_path(mask_image_path))
        
        try:
            with Image.open(mask_image_path) as img:
                # Convert to RGB to ensure consistent color format
                img_rgb = img.convert('RGB')
                img_array = np.array(img_rgb)
                height, width = img_array.shape[:2]
            
            # Initialize mask with background (class 0)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Use global color-to-class mapping
            for hex_color, class_name in self.color_to_class.items():
                # Get class index for this class name
                class_idx = self.class_to_index.get(class_name, 0)
                
                if class_idx == 0:  # Skip background or unknown classes
                    continue
                
                # Convert hex color to RGB tuple
                target_rgb = self._hex_to_rgb(hex_color)
                
                # First try exact matching
                exact_match = np.all(img_array == target_rgb, axis=2)
                exact_count = np.sum(exact_match)
                
                if exact_count > 0:
                    # Perfect! Use exact matches
                    mask[exact_match] = class_idx
                    print(f"Exact color match for {class_name} ({hex_color}): {exact_count} pixels")
                else:
                    # There are no pixels with this class/colour in the image
                    print(f"Warning: No pixels found for {class_name} ({hex_color})")
            
            # Report background pixels and total verification
            background_count = np.sum(mask == 0)
            total_pixels = height * width
            assigned_pixels = np.sum(mask > 0)
            
            # Get background color for reporting (default to #000000 if not found)
            background_color = self.index_to_color.get(0, '#000000')
            print(f"Background pixels ({background_color}): {background_count}")
            print(f"Total pixels: {total_pixels}, Assigned: {assigned_pixels + background_count}, Expected: {total_pixels}")
            
            if assigned_pixels + background_count != total_pixels:
                raise ValueError(f"WARNING: Pixel count mismatch! Missing {total_pixels - (assigned_pixels + background_count)} pixels")
            
            # Map classes to User-defined classes if mapping is provided
            if self.class_mapping:
                return self._apply_class_mapping(Image.fromarray(mask, mode='L'))
            else: 
                return Image.fromarray(mask, mode='L')
            
        except Exception as e:
            print(f"Warning: Could not process mask image at {self._get_relative_path(mask_image_path)}: {e}")
            # Return a blank mask as fallback
            return Image.fromarray(np.zeros((224, 224), dtype=np.uint8), mode='L')

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # Load RGB image
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')
        
        # Generate segmentation mask on-demand
        target = self._generate_mask(self.annotations[index])
        
        # Apply transforms if provided
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        # Return with or without path based on configuration
        if self.return_paths:
            return image, target, self._get_relative_path(image_path)
        else:
            return image, target
    
    def get_class_info(self):
        """
        Returns information about the classes in the dataset.
        """
        return {
            'num_classes': self.num_classes,
            'class_to_index': self.class_to_index,
            'index_to_class': self.index_to_class,
            'color_to_class': self.color_to_class,
            'index_to_color': self.index_to_color
        }


def create_keymakr_dataloader(root_dir, image_set='train', batch_size=4, num_workers=4, 
                             transforms=None, val_split=0.2, random_seed=42, return_paths=False):
    """
    Convenience function to create a Keymakr DataLoader.
    
    Args:
        root_dir (str): Root directory containing Keymakr annotations and images
        image_set (str): 'train' or 'val'
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of worker processes for data loading
        transforms: Transform pipeline to apply to images and masks
        val_split (float): Fraction of data for validation
        random_seed (int): Random seed for reproducible splits
        return_paths (bool): If True, return image paths along with images and masks
    
    Returns:
        DataLoader: Configured PyTorch DataLoader
        dict: Class information dictionary
    """
    from torch.utils.data import DataLoader
    from utils import collate_fn  # Import from the utils module in the repository
    
    dataset = KeyMakrSegmentation(
        root_dir=root_dir,
        image_set=image_set,
        transforms=transforms,
        val_split=val_split,
        random_seed=random_seed,
        return_paths=return_paths
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(image_set == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=(image_set == 'train')
    )
    
    return dataloader, dataset.get_class_info()


# Example usage and testing
if __name__ == "__main__":
    # This example demonstrates basic usage of the KeymakrSegmentation dataset
    
    # Example: Create dataset (adjust path as needed)
    dataset = KeyMakrSegmentation(
        root_dir="/home/nvidia/Downloads/keymakr/batch_09",
        image_set='train',
        transforms=None
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Test loading a sample
    if len(dataset) > 0:
        sample = dataset[0]
        if len(sample) == 3:  # return_paths=True
            image, mask, path = sample
            print(f"Image size: {image.size}")
            print(f"Image path: {path}")
        else:  # return_paths=False
            image, mask = sample
            print(f"Image size: {image.size}")
        print(f"Mask size: {mask.size}")
        print(f"Unique mask values: {np.unique(np.array(mask))}")
    