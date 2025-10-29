# KeyMakr Segmentation Dataset Integration

This directory contains a complete integration for training segmentation models on KeyMakr annotation format datasets. The KeyMakr format uses JSON metadata files with colored segmentation masks for computer vision tasks.

## Dataset Format

The KeyMakr dataset should be organized as follows:

```
your_dataset/
├── sequence1.json                 # Annotation metadata
├── sequence1.images/             # Image and mask directory
│   ├── 00001/                   # Frame directory
│   │   ├── all.png             # RGB input image
│   │   ├── 0.1.png             # Individual object masks
│   │   ├── 0.2.png
│   │   └── ...
│   ├── 00002/
│   └── ...
├── sequence2.json
├── sequence2.images/
└── ...
```

### JSON Annotation Format

Each JSON file contains metadata like this:

```json
{
    "version": "1.5",
    "file": "sequence_name", 
    "width": 1920,
    "height": 1200,
    "objects": [
        {
            "nm": "0.2",
            "type": "car",
            "shape": "bitmap", 
            "color": "#ff0000"
        },
        {
            "nm": "0.3", 
            "type": "tree",
            "shape": "bitmap",
            "color": "#00ff00"
        }
    ]
}
```

**Important**: The `type` and `color` pairing must be globally consistent across all annotation files. For example, if "car" maps to "#ff0000" in one file, it must use the same color in all files.

## Files Included

### Core Dataset Implementation
- **`datasets/keymakr.py`**: Main dataset class implementing PyTorch Dataset interface
- **`example_keymakr_training.py`**: Complete training example with DataLoader setup
- **`validate_keymakr_dataset.py`**: Dataset validation and inspection tool

## Quick Start

### 1. Validate Your Dataset

First, validate your dataset structure and check for consistency issues:

```bash
python validate_keymakr_dataset.py /path/to/your/keymakr/data
```

This will check for:
- Proper directory structure
- Missing files
- Color-to-class mapping consistency
- Data loading capabilities

### 2. Run Training Example

Train a segmentation model on your dataset:

```bash
python example_keymakr_training.py \
    --data-dir /path/to/your/keymakr/data \
    --batch-size 4 \
    --epochs 10 \
    --resolution 512 512 \
    --visualize \
    --model-arch fcn_resnet18
```

### 3. Custom Integration

Use the dataset in your own training scripts:

```python
from datasets.keymakr import KeyMakrSegmentation, create_keymakr_dataloader
import transforms as T

# Create transforms
transforms = T.Compose([
    T.Resize((512, 512)),
    T.RandomHorizontalFlip(0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset
dataset = KeyMakrSegmentation(
    root_dir="/path/to/your/data",
    image_set='train',
    transforms=transforms,
    val_split=0.2
)

# Or use the convenience function
dataloader, class_info = create_keymakr_dataloader(
    root_dir="/path/to/your/data",
    image_set='train', 
    batch_size=4,
    transforms=transforms
)

print(f"Number of classes: {class_info['num_classes']}")
print(f"Classes: {list(class_info['class_to_index'].keys())}")
```

## Features

### Automatic Class Mapping
- Scans all annotation files to build a global color-to-class mapping
- Ensures consistent class indices across the entire dataset
- Background pixels are automatically assigned class index 0
- Object classes start from index 1

### Flexible Data Splitting
- Configurable train/validation splits
- Reproducible splits using random seeds
- Supports both 'train' and 'val' image sets

### Robust Error Handling
- Graceful handling of missing files
- Validation of color-class mapping consistency
- Detailed error reporting and warnings

### Transform Compatibility
- Full compatibility with existing torchvision transforms
- Consistent application to both images and masks
- Support for common augmentations (flip, resize, crop, etc.)

## Dataset Class Details

### KeyMakrSegmentation Class

**Key Methods:**
- `__init__(root_dir, image_set, transforms, val_split, random_seed)`: Initialize dataset
- `__getitem__(index)`: Load image and mask pair
- `__len__()`: Get dataset size
- `get_class_info()`: Get class mapping information

**Key Attributes:**
- `num_classes`: Total number of classes (including background)
- `class_to_index`: Mapping from class names to indices
- `color_to_class`: Mapping from hex colors to class names

### Color-to-Mask Conversion

The dataset automatically converts colored segmentation masks to grayscale class indices:

1. **Color Extraction**: Parses hex colors from JSON annotations
2. **Global Mapping**: Builds consistent color-to-class mapping across all sequences  
3. **Mask Generation**: Combines individual object masks into unified segmentation masks
4. **Class Assignment**: Assigns appropriate class indices based on object types

## Troubleshooting

### Common Issues

**"Color mapping inconsistency" warnings:**
- Check that each object type uses the same color across all annotation files
- Use the validation script to identify specific inconsistencies

**Missing mask files:**
- Ensure all referenced object masks (e.g., "0.1.png") exist in frame directories
- Check that file names match the "nm" field in annotations exactly

**Empty dataset:**
- Verify that "all.png" files exist in frame directories
- Check that JSON files are valid and contain object annotations

**Memory issues with large datasets:**
- Reduce batch size or image resolution
- Consider using fewer workers in DataLoader

### Performance Tips

1. **Preprocessing**: Consider preprocessing masks once and saving them to disk for faster loading
2. **Caching**: The dataset generates masks on-demand; implement caching for repeated access
3. **Multiprocessing**: Use multiple DataLoader workers for faster data loading
4. **Image Formats**: Ensure images are in efficient formats (PNG for masks, JPEG for photos)

## Integration with Existing Training Scripts

The KeyMakr dataset integrates seamlessly with the existing training infrastructure:

```python
# In train.py, add KeyMakr support:
from datasets.keymakr import KeyMakrSegmentation

def get_dataset(name, path, image_set, transform, num_classes):
    paths = {
        "voc": (path, torchvision.datasets.VOCSegmentation, num_classes),
        "keymakr": (path, KeyMakrSegmentation, num_classes),  # Add this line
        # ... other datasets
    }
    # ... rest of function
```

Then train with:
```bash
python train.py /path/to/keymakr/data --dataset keymakr --classes 10
```

## Advanced Usage

### Custom Class Filtering

Filter or remap specific classes:

```python
class FilteredKeymakrDataset(KeyMakrSegmentation):
    def __init__(self, *args, keep_classes=None, **kwargs):
        super().__init__(*args, **kwargs)
        if keep_classes:
            self._filter_classes(keep_classes)
    
    def _filter_classes(self, keep_classes):
        # Implementation to filter specific classes
        pass
```

### Multi-Scale Training

Use different resolutions during training:

```python
class MultiScaleTransforms:
    def __init__(self, scales=[480, 512, 544]):
        self.scales = scales
    
    def __call__(self, image, target):
        scale = random.choice(self.scales)
        # Apply scale-specific transforms
        return image, target
```

## Support

For issues specific to the KeyMakr dataset integration:

1. **Validation**: Always run `validate_keymakr_dataset.py` first
2. **Examples**: Check `example_keymakr_training.py` for reference implementation
3. **Debugging**: Enable detailed logging in the dataset class
4. **Performance**: Monitor memory usage and adjust batch sizes accordingly

The implementation follows PyTorch best practices and integrates with the existing segmentation training pipeline in this repository.