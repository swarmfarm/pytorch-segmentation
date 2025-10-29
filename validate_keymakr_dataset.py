#!/usr/bin/env python3
"""
KeyMakr Dataset Validation and Inspection Tool

This script helps validate your KeyMakr dataset structure and provides insights into:
- Dataset structure validation
- Color-to-class mapping consistency
- Missing files detection
- Class distribution analysis
- Data quality checks

Usage:
    python validate_keymakr_dataset.py /path/to/your/keymakr/data
"""

import os
import sys
import json
import argparse
from collections import defaultdict, Counter
import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_dataset_structure(root_dir):
    """
    Validate the basic structure of the KeyMakr dataset.
    """
    print("=== Dataset Structure Validation ===")
    
    if not os.path.exists(root_dir):
        print(f"ERROR: Root directory does not exist: {root_dir}")
        return False
    
    # Find all JSON files
    json_files = [f for f in os.listdir(root_dir) if f.endswith('.json')]
    
    if not json_files:
        print("ERROR: No JSON annotation files found")
        return False
    
    print(f"Found {len(json_files)} JSON annotation files")
    
    # Find RGB data directories (those without .images suffix)
    all_dirs = [d for d in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, d)) and not d.endswith('.images')]
    
    # Group RGB dirs by sequence name
    rgb_dirs_by_sequence = defaultdict(list)
    for dir_name in all_dirs:
        # Extract sequence name (part before _data_)
        if '_data_' in dir_name:
            sequence_base = dir_name.split('_data_')[0]
            rgb_dirs_by_sequence[sequence_base].append(dir_name)
    
    # Sort RGB directories for consistent ordering
    for sequence_base in rgb_dirs_by_sequence:
        rgb_dirs_by_sequence[sequence_base].sort()
    
    print(f"Found {len(all_dirs)} RGB data directories")
    
    # Validate each sequence
    valid_sequences = 0
    total_images = 0
    total_matched_rgb = 0
    
    for json_file in json_files:
        sequence_name = json_file[:-5]  # Remove .json
        images_dir = os.path.join(root_dir, f"{sequence_name}.images")
        
        print(f"\nValidating sequence: {sequence_name}")
        
        # Check if images directory exists
        if not os.path.exists(images_dir):
            print(f"  ERROR: Images directory missing: {images_dir}")
            continue
        
        # Check for corresponding RGB directories
        rgb_dirs = rgb_dirs_by_sequence.get(sequence_name, [])
        if not rgb_dirs:
            print(f"  WARNING: No RGB data directories found for {sequence_name}")
        else:
            print(f"  Found {len(rgb_dirs)} RGB data directories: {rgb_dirs}")
        
        # Count frame directories
        frame_dirs = [d for d in os.listdir(images_dir) 
                     if os.path.isdir(os.path.join(images_dir, d))]
        
        if not frame_dirs:
            print(f"  ERROR: No frame directories found in {images_dir}")
            continue
        
        # Check each frame directory for all.png and corresponding RGB
        valid_frames = 0
        matched_rgb_images = 0
        
        for frame_dir in frame_dirs:
            frame_path = os.path.join(images_dir, frame_dir)
            all_png = os.path.join(frame_path, 'all.png')
            
            if os.path.exists(all_png):
                valid_frames += 1
                
                # Check for corresponding RGB image across ALL RGB directories for this sequence
                rgb_found = False
                for rgb_dir in rgb_dirs:
                    rgb_dir_path = os.path.join(root_dir, rgb_dir)
                    # Look for image files with similar frame name
                    if os.path.exists(rgb_dir_path):
                        rgb_files = [f for f in os.listdir(rgb_dir_path) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                        
                        # Try to match by frame name or number
                        for rgb_file in rgb_files:
                            # Strategy 1: Direct frame name match
                            if frame_dir in rgb_file:
                                rgb_found = True
                                matched_rgb_images += 1
                                break
                            
                            # Strategy 2: Extract numbers and match
                            frame_numbers = ''.join([c for c in frame_dir if c.isdigit()])
                            rgb_numbers = ''.join([c for c in rgb_file if c.isdigit()])
                            
                            if frame_numbers and rgb_numbers and frame_numbers == rgb_numbers:
                                rgb_found = True
                                matched_rgb_images += 1
                                break
                        
                        if rgb_found:
                            break
                
                if not rgb_found and rgb_dirs:
                    print(f"  WARNING: No matching RGB image found for frame {frame_dir} in any of {len(rgb_dirs)} RGB directories")
            else:
                print(f"  WARNING: Missing all.png in {frame_path}")
        
        print(f"  Found {len(frame_dirs)} frame directories, {valid_frames} with all.png")
        if rgb_dirs:
            print(f"  Matched {matched_rgb_images}/{valid_frames} mask-RGB pairs")
        
        total_images += valid_frames
        total_matched_rgb += matched_rgb_images
        
        if valid_frames > 0:
            valid_sequences += 1
    
    print(f"\n=== Summary ===")
    print(f"Valid sequences: {valid_sequences}/{len(json_files)}")
    print(f"Total valid images: {total_images}")
    print(f"Total RGB-mask pairs: {total_matched_rgb}")
    
    return valid_sequences > 0


def analyze_annotations(root_dir):
    """
    Analyze annotation files for consistency and completeness.
    """
    print("\n=== Annotation Analysis ===")
    
    json_files = [f for f in os.listdir(root_dir) if f.endswith('.json')]
    
    if not json_files:
        print("ERROR: No JSON annotation files found")
        return set(), defaultdict(set)
    
    all_types = set()
    type_colors = defaultdict(set)
    color_types = defaultdict(set)
    sequences_info = []
    failed_files = []
    
    for json_file in json_files:
        json_path = os.path.join(root_dir, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                annotation = json.load(f)
            
            # Debug: Print raw structure for first file
            if len(sequences_info) == 0:
                print(f"DEBUG: First JSON file structure for {json_file}:")
                print(f"  Top-level keys: {list(annotation.keys())}")
                if 'objects' in annotation and annotation['objects']:
                    first_obj = annotation['objects'][0]
                    print(f"  First object keys: {list(first_obj.keys())}")
                    print(f"  First object sample: {first_obj}")
                print()
            
            # Extract sequence info with more robust handling
            seq_info = {
                'file': json_file,
                'width': annotation.get('width', annotation.get('w', 'unknown')),
                'height': annotation.get('height', annotation.get('h', 'unknown')),
                'objects': len(annotation.get('objects', annotation.get('annotations', [])))
            }
            sequences_info.append(seq_info)
            
            # Analyze objects with multiple field name attempts
            objects_list = annotation.get('objects', annotation.get('annotations', []))
            
            if not objects_list:
                print(f"WARNING: No objects found in {json_file}")
                print(f"  Available keys: {list(annotation.keys())}")
                continue
            
            for i, obj in enumerate(objects_list):
                # Try multiple possible field names for type
                obj_type = (obj.get('type') or 
                           obj.get('class') or 
                           obj.get('category') or 
                           obj.get('label') or 
                           'unknown')
                
                # Try multiple possible field names for color
                obj_color = (obj.get('color') or 
                            obj.get('colour') or 
                            obj.get('rgb') or 
                            obj.get('hex') or 
                            '#000000')
                
                # Try multiple possible field names for name
                obj_name = (obj.get('nm') or 
                           obj.get('name') or 
                           obj.get('id') or 
                           f'object_{i}')
                
                # Debug: Print object structure for first few objects if fields are missing
                if obj_type == 'unknown' or obj_color == '#000000':
                    print(f"WARNING: Missing fields in {json_file}, object {i}:")
                    print(f"  Object keys: {list(obj.keys())}")
                    print(f"  Object data: {obj}")
                    print(f"  Extracted - type: '{obj_type}', color: '{obj_color}', name: '{obj_name}'")
                
                all_types.add(obj_type)
                type_colors[obj_type].add(obj_color)
                color_types[obj_color].add(obj_type)
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error in {json_file}: {e}"
            print(f"ERROR: {error_msg}")
            failed_files.append((json_file, error_msg))
        except FileNotFoundError as e:
            error_msg = f"File not found: {json_file}"
            print(f"ERROR: {error_msg}")
            failed_files.append((json_file, error_msg))
        except Exception as e:
            error_msg = f"Unexpected error in {json_file}: {e}"
            print(f"ERROR: {error_msg}")
            failed_files.append((json_file, error_msg))
    
    # Print sequence information
    print("Sequence Information:")
    for seq in sequences_info:
        print(f"  {seq['file']}: {seq['width']}x{seq['height']}, {seq['objects']} objects")
    
    if failed_files:
        print(f"\nFailed to process {len(failed_files)} files:")
        for filename, error in failed_files:
            print(f"  {filename}: {error}")
    
    # Check for type-color consistency
    print(f"\nFound {len(all_types)} unique object types:")
    inconsistent_types = []
    
    for obj_type in sorted(all_types):
        colors = type_colors[obj_type]
        print(f"  {obj_type}: {len(colors)} color(s) - {list(colors)}")
        
        if len(colors) > 1:
            inconsistent_types.append(obj_type)
    
    if inconsistent_types:
        print(f"\nWARNING: Inconsistent color mappings for types: {inconsistent_types}")
        print("Each object type should map to exactly one color across all sequences.")
    
    # Check for color-type consistency
    print(f"\nColor-to-type mappings:")
    inconsistent_colors = []
    
    for color in sorted(color_types.keys()):
        types = color_types[color]
        print(f"  {color}: {list(types)}")
        
        if len(types) > 1:
            inconsistent_colors.append(color)
    
    if inconsistent_colors:
        print(f"\nWARNING: Colors used for multiple types: {inconsistent_colors}")
        print("Each color should map to exactly one object type.")
    
    # Additional validation checks
    print(f"\nAdditional Validation:")
    print(f"  Total sequences processed: {len(sequences_info)}")
    print(f"  Failed sequences: {len(failed_files)}")
    print(f"  Success rate: {(len(sequences_info) / max(len(json_files), 1) * 100):.1f}%")
    
    return all_types, type_colors


def check_rgb_mask_correspondence(root_dir, sample_sequences=3):
    """
    Check detailed correspondence between RGB images and mask files.
    """
    print(f"\n=== RGB-Mask Correspondence Check (sampling {sample_sequences} sequences) ===")
    
    json_files = [f for f in os.listdir(root_dir) if f.endswith('.json')][:sample_sequences]
    
    # Find RGB data directories
    all_dirs = [d for d in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, d)) and not d.endswith('.images')]
    
    # Group RGB dirs by sequence name
    rgb_dirs_by_sequence = defaultdict(list)
    for dir_name in all_dirs:
        if '_data_' in dir_name:
            sequence_base = dir_name.split('_data_')[0]
            rgb_dirs_by_sequence[sequence_base].append(dir_name)
    
    # Sort RGB directories for consistent ordering
    for sequence_base in rgb_dirs_by_sequence:
        rgb_dirs_by_sequence[sequence_base].sort()
    
    total_pairs_checked = 0
    successful_pairs = 0
    
    for json_file in json_files:
        sequence_name = json_file[:-5]
        images_dir = os.path.join(root_dir, f"{sequence_name}.images")
        rgb_dirs = rgb_dirs_by_sequence.get(sequence_name, [])
        
        print(f"\nChecking RGB-mask pairs for sequence: {sequence_name}")
        
        if not rgb_dirs:
            print(f"  No RGB directories found for {sequence_name}")
            continue
        else:
            print(f"  Found {len(rgb_dirs)} RGB directories: {[d.split('_data_')[1] for d in rgb_dirs]}")
        
        if not os.path.exists(images_dir):
            print(f"  Images directory missing: {images_dir}")
            continue
        
        # Check frame directories
        frame_dirs = [d for d in os.listdir(images_dir) 
                     if os.path.isdir(os.path.join(images_dir, d))][:3]  # Sample first 3 frames
        
        for frame_dir in frame_dirs:
            frame_path = os.path.join(images_dir, frame_dir)
            all_png = os.path.join(frame_path, 'all.png')
            
            if not os.path.exists(all_png):
                continue
            
            print(f"  Frame {frame_dir}:")
            
            # Look for matching RGB image across ALL RGB directories for this sequence
            matched_rgb = None
            matched_rgb_path = None
            matched_rgb_dir = None
            
            for rgb_dir in rgb_dirs:
                rgb_dir_path = os.path.join(root_dir, rgb_dir)
                if not os.path.exists(rgb_dir_path):
                    continue
                
                rgb_files = [f for f in os.listdir(rgb_dir_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                
                # Try different matching strategies
                for rgb_file in rgb_files:
                    rgb_file_base = os.path.splitext(rgb_file)[0]
                    
                    # Strategy 1: Direct frame name match
                    if frame_dir in rgb_file:
                        matched_rgb = rgb_file
                        matched_rgb_path = os.path.join(rgb_dir_path, rgb_file)
                        matched_rgb_dir = rgb_dir
                        break
                    
                    # Strategy 2: Extract numbers and match
                    frame_numbers = ''.join([c for c in frame_dir if c.isdigit()])
                    rgb_numbers = ''.join([c for c in rgb_file_base if c.isdigit()])
                    
                    if frame_numbers and rgb_numbers and frame_numbers == rgb_numbers:
                        matched_rgb = rgb_file
                        matched_rgb_path = os.path.join(rgb_dir_path, rgb_file)
                        matched_rgb_dir = rgb_dir
                        break
                
                if matched_rgb:
                    break
            
            total_pairs_checked += 1
            
            if matched_rgb:
                # Verify the files can be loaded and have compatible dimensions
                try:
                    with Image.open(all_png) as mask_img:
                        mask_size = mask_img.size
                    
                    with Image.open(matched_rgb_path) as rgb_img:
                        rgb_size = rgb_img.size
                    
                    rgb_dir_suffix = matched_rgb_dir.split('_data_')[1] if matched_rgb_dir and '_data_' in matched_rgb_dir else matched_rgb_dir or 'unknown'
                    
                    if mask_size == rgb_size:
                        print(f"    ✓ {matched_rgb} (from {rgb_dir_suffix}) ({rgb_size[0]}x{rgb_size[1]}) - MATCH")
                        successful_pairs += 1
                    else:
                        print(f"    ⚠ {matched_rgb} (from {rgb_dir_suffix}) - SIZE MISMATCH: mask{mask_size} vs rgb{rgb_size}")
                
                except Exception as e:
                    print(f"    ✗ {matched_rgb} - ERROR loading: {e}")
            else:
                print(f"    ✗ No matching RGB image found in any of {len(rgb_dirs)} RGB directories")
    
    print(f"\nRGB-Mask Correspondence Summary:")
    print(f"  Pairs checked: {total_pairs_checked}")
    print(f"  Successful matches: {successful_pairs}")
    if total_pairs_checked > 0:
        success_rate = (successful_pairs / total_pairs_checked) * 100
        print(f"  Success rate: {success_rate:.1f}%")
    
    return successful_pairs, total_pairs_checked


def check_mask_files(root_dir, sample_sequences=3):
    """
    Check individual mask files for a sample of sequences.
    """
    print(f"\n=== Mask File Analysis (sampling {sample_sequences} sequences) ===")
    
    json_files = [f for f in os.listdir(root_dir) if f.endswith('.json')][:sample_sequences]
    
    total_mask_files = 0
    missing_mask_files = 0
    
    for json_file in json_files:
        sequence_name = json_file[:-5]
        json_path = os.path.join(root_dir, json_file)
        images_dir = os.path.join(root_dir, f"{sequence_name}.images")
        
        print(f"\nChecking masks for sequence: {sequence_name}")
        
        try:
            with open(json_path, 'r') as f:
                annotation = json.load(f)
        except:
            continue
        
        # Check a few frame directories
        frame_dirs = [d for d in os.listdir(images_dir) 
                     if os.path.isdir(os.path.join(images_dir, d))][:2]  # Sample first 2 frames
        
        for frame_dir in frame_dirs:
            frame_path = os.path.join(images_dir, frame_dir)
            print(f"  Frame {frame_dir}:")
            
            # Check for individual object masks
            for obj in annotation.get('objects', []):
                obj_name = obj.get('nm', '')
                mask_file = os.path.join(frame_path, f"{obj_name}.png")
                
                total_mask_files += 1
                
                if os.path.exists(mask_file):
                    # Try to load and check the mask
                    try:
                        with Image.open(mask_file) as img:
                            img_array = np.array(img)
                            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
                            print(f"    {obj_name}.png: OK ({unique_colors} unique colors)")
                    except Exception as e:
                        print(f"    {obj_name}.png: ERROR - {e}")
                        missing_mask_files += 1
                else:
                    print(f"    {obj_name}.png: MISSING")
                    missing_mask_files += 1
    
    print(f"\nMask File Summary:")
    print(f"  Total checked: {total_mask_files}")
    print(f"  Missing/invalid: {missing_mask_files}")
    print(f"  Success rate: {((total_mask_files - missing_mask_files) / max(total_mask_files, 1) * 100):.1f}%")


def test_dataset_loading(root_dir):
    """
    Test loading the dataset using the KeyMakrSegmentation class.
    """
    print("\n=== Dataset Loading Test ===")
    
    try:
        from datasets.keymakr import KeymakrSegmentation
        
        # Try to create dataset
        dataset = KeymakrSegmentation(
            root_dir=root_dir,
            image_set='train',
            transforms=None,
            val_split=0.8  # Use most data for this test
        )
        
        print(f"Dataset created successfully!")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Number of classes: {dataset.num_classes}")
        print(f"  Classes: {list(dataset.class_to_index.keys())}")
        
        # Try to load a few samples
        if len(dataset) > 0:
            print(f"\nTesting sample loading...")
            
            for i in range(min(3, len(dataset))):
                try:
                    image, mask = dataset[i]
                    print(f"  Sample {i}: Image {image.size}, Mask {mask.size}")
                    
                    # Check mask values
                    mask_array = np.array(mask)
                    unique_values = np.unique(mask_array)
                    print(f"    Mask classes: {unique_values}")
                    
                except Exception as e:
                    print(f"  Sample {i}: ERROR - {e}")
        
        return True
        
    except ImportError:
        print("ERROR: Could not import KeyMakrSegmentation class")
        print("Make sure the keymakr.py file is in the datasets/ directory")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def generate_report(root_dir):
    """
    Generate a comprehensive validation report.
    """
    print(f"\n{'='*60}")
    print(f"KEYMAKR DATASET VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Dataset path: {root_dir}")
    
    # Run all validation checks
    structure_ok = validate_dataset_structure(root_dir)
    all_types, type_colors = analyze_annotations(root_dir)
    successful_pairs, total_pairs = check_rgb_mask_correspondence(root_dir)
    check_mask_files(root_dir)
    dataset_loading_ok = test_dataset_loading(root_dir)
    
    # Calculate RGB-mask correspondence rate
    rgb_correspondence_ok = False
    if total_pairs > 0:
        correspondence_rate = (successful_pairs / total_pairs) * 100
        rgb_correspondence_ok = correspondence_rate >= 80  # 80% threshold
    
    # Summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Dataset structure: {'PASS' if structure_ok else 'FAIL'}")
    print(f"✓ RGB-mask correspondence: {'PASS' if rgb_correspondence_ok else 'FAIL'} ({successful_pairs}/{total_pairs} matched)")
    print(f"✓ Dataset loading: {'PASS' if dataset_loading_ok else 'FAIL'}")
    print(f"✓ Found {len(all_types)} object types")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    
    inconsistent_types = [t for t, colors in type_colors.items() if len(colors) > 1]
    if inconsistent_types:
        print(f"• Fix inconsistent color mappings for: {inconsistent_types}")
    
    if not rgb_correspondence_ok and total_pairs > 0:
        print(f"• Improve RGB-mask file naming/organization for better correspondence")
        print(f"• Ensure RGB images and mask frames use consistent naming schemes")
    
    if structure_ok and dataset_loading_ok and rgb_correspondence_ok:
        print(f"• Dataset is ready for training!")
        print(f"• Consider running the example training script:")
        print(f"  python example_keymakr_training.py --data-dir {root_dir} --visualize")
    else:
        print(f"• Fix the identified issues before training")


def main():
    parser = argparse.ArgumentParser(description='Validate KeyMakr Dataset')
    parser.add_argument('data_dir', type=str, help='Path to KeyMakr dataset directory')
    parser.add_argument('--detailed', action='store_true', 
                       help='Run detailed checks (slower but more thorough)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    generate_report(args.data_dir)


if __name__ == '__main__':
    main()