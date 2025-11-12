#
# Note -- this training script is tweaked from the original version at:
#
#           https://github.com/pytorch/vision/tree/v0.3.0/references/segmentation
#
#
import argparse
import datetime
import time
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.utils.data
from torch import nn
import torchvision
from models import segmentation
import numpy as np

from datasets.coco_utils import get_coco, get_coco_sf, ConvertCocoPolysToMask
from datasets.cityscapes_utils import get_cityscapes
from datasets.deepscene import DeepSceneSegmentation
from datasets.custom_dataset import CustomSegmentation
from datasets.mhp import MHPSegmentation
from datasets.nyu import NYUDepth
from datasets.sun import SunRGBDSegmentation
from datasets.keymakr import KeymakrSegmentation

import transforms as T
import utils

model_names = sorted(name for name in segmentation.__dict__
    if name.islower() and not name.startswith("__")
    and callable(segmentation.__dict__[name]))

#
# parse command-line arguments
#
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training')

    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--dataset', default='voc', help='dataset type: voc, voc_aug, coco, cocosf, cityscapes, deepscene, keymakr, mhp, nyu, sun, custom (default: voc)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fcn_resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: fcn_resnet18)')
    parser.add_argument('--classes', default=21, type=int, metavar='C', help='number of classes in your dataset (outputs)')
    parser.add_argument('--aux-loss', action='store_true', help='train with auxilliary loss')
    parser.add_argument('--resolution', default=320, type=int, metavar='N',
                        help='NxN resolution used for scaling the training dataset (default: 320x320) '
                         'to specify a non-square resolution, use the --width and --height options')
    parser.add_argument('--width', default=argparse.SUPPRESS, type=int, metavar='X',
                        help='desired width of the training dataset. if this option is not set, --resolution will be used')
    parser.add_argument('--height', default=argparse.SUPPRESS, type=int, metavar='Y',
                        help='desired height of the training dataset. if this option is not set, --resolution will be used')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--model-dir', default='./results/', help='Path where to save output models')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models (only supported for fcn_resnet101)", action="store_true")
    parser.add_argument('--debug-gt', action='store_true', help='Output some debug images with ground-truth overlays')
    parser.add_argument('--map-classes', action='store_true', help='Map classes in the original dataset to user-specified classes (only for Keymakr dataset)')
    parser.add_argument('--validate', action='store_true', help='(NEEDS TO BE RELOCATED TO A SEPARATE SCRIPT) Validate the model on a separate dataset')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args


#
# load desired dataset
#
def get_dataset(name, path, image_set, transform, num_classes, user_class_mapping: Dict[str, str] = {}):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "voc": (path, torchvision.datasets.VOCSegmentation, num_classes),
        "voc_aug": (path, sbd, num_classes),
        "coco": (path, get_coco, num_classes),
        "cocosf": (path, get_coco_sf, num_classes),
        "cityscapes": (path, get_cityscapes, num_classes),
        "deepscene": (path, DeepSceneSegmentation, 5),
        "mhp": (path, MHPSegmentation, num_classes),
        "nyu": (path, NYUDepth, num_classes),
        "sun": (path, SunRGBDSegmentation, num_classes),
        "custom": (path, CustomSegmentation, num_classes)
    }

    if name == "keymakr":
        # Special case for Keymakr dataset to allow User class mappings
        if image_set == "test":
            # This a hacky way to load the entire dataset for testing
            #   It still needs to be called either "train" or "val" due to how the dataset class is implemented, but we want to load all data
            ds = KeymakrSegmentation(
                root_dir=path, 
                image_set="train", 
                transforms=transform,
                val_split=0.,
                class_mapping=user_class_mapping,
                return_paths=True  # get the image paths as well if running on a test set
            )
        else: 
            ds = KeymakrSegmentation(
                root_dir=path, 
                image_set=image_set, 
                transforms=transform,
                class_mapping=user_class_mapping
            )

        # Override num_classes for Keymakr based on dataset
        num_classes = ds.num_classes
    else: 
        p, ds_fn, num_classes = paths[name]
        ds = ds_fn(p, image_set=image_set, transforms=transform)

    return ds, num_classes


#
# create data transform
#
def get_transform(train, resolution):
    transforms = []

    # if square resolution, perform some aspect cropping
    # otherwise, resize to the resolution as specified
    if resolution[0] == resolution[1]:
        base_size = resolution[0] + 32 #520
        crop_size = resolution[0]      #480

        min_size = int((0.5 if train else 1.0) * base_size)
        max_size = int((2.0 if train else 1.0) * base_size)

        transforms.append(T.RandomResize(min_size, max_size))

        # during training mode, perform some data randomization
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
            transforms.append(T.RandomCrop(crop_size))
    else:
        transforms.append(T.Resize(resolution))

        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))

    transforms.append(T.ToTensor())

    if train: 
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


#
# define the loss functions
#
def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


#
# evaluate model IoU (intersection over union)
#
def evaluate(model, data_loader, device, num_classes, visualise_dir=None):

    overlay = None
    dataset = data_loader.dataset
    if visualise_dir:
        if not isinstance(visualise_dir, Path) and not isinstance(visualise_dir, str):
            print("ERROR: visualise_dir must be a string or Path object, not {}".format(type(visualise_dir)))
            return
        
        if isinstance(visualise_dir, str):
            visualise_dir = Path(visualise_dir)
        
        visualise_dir.mkdir(exist_ok=True)
        overlay, _ = create_dataset_mask_visualiser(dataset)
        overlay.create_legend(dataset.index_to_class, save_path=visualise_dir / "_legend.png")

    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        iter = 1        
        iterator = metric_logger.log_every(data_loader, 100, header)

        for next in iterator:
            if hasattr(dataset, "return_paths") and dataset.return_paths:
                image, target, path = next
                if isinstance(path, tuple):  # this is returned as a tuple sometimes??
                    path = path[0]
            else: 
                image, target = next

            image, target = image.to(device), target.to(device)

            output = model(image)
            output = output['out']
            output = output.argmax(1)

            confmat.update(target.flatten(), output.flatten())

            if overlay:
                # The image name to be save should be the iteration index padded to 6 digits leading zeros
                assert visualise_dir is not None
                assert path is not None
                assert isinstance(path, str)
                path = path.replace("/", "_")  # avoid subdirectories
                file_path = os.path.join(visualise_dir, f"{Path(path).with_suffix('.jpg')}")

                # Assume that the batch size of 1 is used during evaluation
                assert image.shape[0] == 1
                assert target.shape[0] == 1
                assert output.shape[0] == 1
                image = image[0]
                target = target[0]
                output = output[0]

                # Overlay the predicted mask onto the input image and save to file
                overlay.overlay_on_image(
                    mask=output, 
                    image=image, 
                    alpha=0.5,
                    background_alpha=0.2,  # Keep background transparent
                    save_path=file_path
                )

            iter += 1

        confmat.reduce_from_all_processes()

    return confmat


#
# train for one epoch over the dataset
#
def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        output = model(image)

        print()
        print("target:", target.shape, torch.unique(target))
        print("type(output):", type(output))
        
        # iterate over the ordered dict output to find the main output
        for key, value in output.items():
            print(key)
            print(type(key), type(value))
            print(f"  output['{key}']:", value.shape, torch.unique(value))


        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


def verify_dataset_labels(datasets: List[KeymakrSegmentation], verbose: bool = False):
    if not isinstance(datasets, list):
        datasets = [datasets]

    if len(datasets) <= 1:
        # Only one dataset, nothing to verify
        return

    # Check that all datasets have identical class mappings
    reference_dataset = datasets[0]
    
    print(f"Verifying consistency across {len(datasets)} KeymakrSegmentation datasets...")
    
    for i, dataset in enumerate(datasets[1:], 1):
        print(f"Comparing dataset {i+1} with reference dataset...")
        
        # Verify color_to_class mapping
        if verbose:
            print()
            print(f"Reference color_to_class: {reference_dataset.color_to_class}")
            print(f"Dataset {i+1} color_to_class: {dataset.color_to_class}")
        if dataset.color_to_class != reference_dataset.color_to_class:
            print(f"ERROR: color_to_class mismatch in dataset {i+1}")
            print(f"Reference: {reference_dataset.color_to_class}")
            print(f"Dataset {i+1}: {dataset.color_to_class}")
            raise ValueError(f"Inconsistent color_to_class mapping in dataset {i+1}")
        
        # Verify class_to_index mapping
        if verbose:
            print()
            print(f"Reference class_to_index: {reference_dataset.class_to_index}")
            print(f"Dataset {i+1} class_to_index: {dataset.class_to_index}")
        if dataset.class_to_index != reference_dataset.class_to_index:
            print(f"ERROR: class_to_index mismatch in dataset {i+1}")
            print(f"Reference: {reference_dataset.class_to_index}")
            print(f"Dataset {i+1}: {dataset.class_to_index}")
            raise ValueError(f"Inconsistent class_to_index mapping in dataset {i+1}")
        
        # Verify index_to_class mapping
        if verbose:
            print()
            print(f"Reference index_to_class: {reference_dataset.index_to_class}")
            print(f"Dataset {i+1} index_to_class: {dataset.index_to_class}")
        if dataset.index_to_class != reference_dataset.index_to_class:
            print(f"ERROR: index_to_class mismatch in dataset {i+1}")
            print(f"Reference: {reference_dataset.index_to_class}")
            print(f"Dataset {i+1}: {dataset.index_to_class}")
            raise ValueError(f"Inconsistent index_to_class mapping in dataset {i+1}")
        
        # Verify index_to_color mapping
        if verbose:
            print()
            print(f"Reference index_to_color: {reference_dataset.index_to_color}")
            print(f"Dataset {i+1} index_to_color: {dataset.index_to_color}")
        if dataset.index_to_color != reference_dataset.index_to_color:
            print(f"ERROR: index_to_color mismatch in dataset {i+1}")
            print(f"Reference: {reference_dataset.index_to_color}")
            print(f"Dataset {i+1}: {dataset.index_to_color}")
            raise ValueError(f"Inconsistent index_to_color mapping in dataset {i+1}")
        
        # Verify class_mapping
        if verbose:
            print()
            print(f"Reference class_mapping: {reference_dataset.class_mapping}")
            print(f"Dataset {i+1} class_mapping: {dataset.class_mapping}")
        if dataset.class_mapping != reference_dataset.class_mapping:
            print(f"ERROR: class_mapping mismatch in dataset {i+1}")
            print(f"Reference: {reference_dataset.class_mapping}")
            print(f"Dataset {i+1}: {dataset.class_mapping}")
            raise ValueError(f"Inconsistent class_mapping in dataset {i+1}")
        
        # Verify num_classes
        if verbose:
            print()
            print(f"Reference num_classes: {reference_dataset.num_classes}")
            print(f"Dataset {i+1} num_classes: {dataset.num_classes}")
        if dataset.num_classes != reference_dataset.num_classes:
            print(f"ERROR: num_classes mismatch in dataset {i+1}")
            print(f"Reference: {reference_dataset.num_classes}")
            print(f"Dataset {i+1}: {dataset.num_classes}")
            raise ValueError(f"Inconsistent num_classes in dataset {i+1}")
        
        print(f"✓ Dataset {i+1} mappings are consistent with reference")
    
    print()
    print("✅ All dataset mappings are consistent!")
    print(f"Reference mappings:")
    print(f"  - num_classes: {reference_dataset.num_classes}")
    print(f"  - class_to_index: {reference_dataset.class_to_index}")
    print(f"  - class_mapping: {reference_dataset.class_mapping}")


def create_dataset_mask_visualiser(keymakr_dataset: KeymakrSegmentation):
    """
    Create a mask visualiser for a Keymakr dataset.
    Args:
        keymakr_dataset (KeymakrSegmentation): An instance of the KeymakrSegmentation dataset.
    Returns:
        overlay (utils.MaskOverlay): An instance of the MaskOverlay utility for visualising masks.
        class_index_to_rgb_colour_map (Dict[int, Tuple[int, int, int]]): The mapping from class indices to RGB colours.
    """
    # Map the class indices to RGB colours
    class_index_to_rgb_colour_map = {k: keymakr_dataset._hex_to_rgb(v) for k, v in keymakr_dataset.index_to_color.items()}
    class_index_to_rgb_colour_map = dict(sorted(class_index_to_rgb_colour_map.items()))
    for class_index, rgb_colour in class_index_to_rgb_colour_map.items():
        print("Class {:d} : Colour {}".format(class_index, rgb_colour))

    # Create the overlay utility instance
    overlay = utils.MaskOverlay(class_index_to_rgb_colour_map)

    return overlay, class_index_to_rgb_colour_map
        

#
# main training function
#
def main(args):
    if args.model_dir:
        utils.mkdir(args.model_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.map_classes and args.dataset == "keymakr":
        # Define a lookup for aggregating certain classes together
        user_class_mapping = {
            # Sky classes
            "sky": "sky",
            
            # Terrain classes (vehicles, equipment, infrastructure)
            "long_grass": "terrain",
            
            # Vegetation classes
            "tree": "vegetation",
            "tree_cluster": "vegetation",
            "bush": "vegetation",
            "bush_cluster": "vegetation",
            "weed_cluster": "vegetation",
            "weed": "vegetation",

            # Obstacle classes
            "ute": "obstacle",
            "truck": "obstacle",
            "tractor": "obstacle",
            "front_end_loader": "obstacle",
            "hopper_trailer": "obstacle",
            "car": "obstacle",
            "person": "obstacle",
            "sprayer": "obstacle",
            "forklift": "obstacle",
            "trailer": "obstacle",
            "swarmbot": "obstacle",
            "sprayer_sp": "obstacle",
            
            # Everything else maps to background
            "background": "background",
            "swarmbot_body": "background",
            "mower_attachment": "background", 
        }
    else:
        user_class_mapping = {}

    if args.debug_gt:
        # Create a torch dataset specifically for visualising ground-truth overlays
        dataset_visualise = KeymakrSegmentation(
            root_dir=args.data, 
            image_set="train", 
            transforms=None,
            val_split=0.,  # for visualisation, use the entire dataset
            return_paths=True,
            class_mapping=user_class_mapping,
            debug_or_vis=True
        )

        # Create the directories for storing debug images
        debug_dir = Path(f"{str(Path(args.data))}_debug")
        if args.map_classes:
            debug_dir = Path(f"{str(debug_dir)}_mapped")
        else:
            debug_dir = Path(f"{str(debug_dir)}_original")
        debug_dir.mkdir(exist_ok=True)

        # # Map the class indices to RGB colours
        # class_index_to_rgb_colour_map = {k: dataset_visualise._hex_to_rgb(v) for k, v in dataset_visualise.index_to_color.items()}
        # class_index_to_rgb_colour_map = dict(sorted(class_index_to_rgb_colour_map.items()))
        # for class_index, rgb_colour in class_index_to_rgb_colour_map.items():
        #     print("Class {:d} : Colour {}".format(class_index, rgb_colour))

        # # Create the overlay utility instance
        # overlay = utils.MaskOverlay(class_index_to_rgb_colour_map)

        overlay, _ = create_dataset_mask_visualiser(dataset_visualise)
        overlay.create_legend(dataset_visualise.index_to_class, save_path=debug_dir / "_legend.png")
        
        img_idx = 0
        for image, target, path in dataset_visualise:
            file_path = os.path.join(debug_dir, path.replace("/", "_"))            
            overlay_image = overlay.overlay_on_image(
                mask=target, 
                image=image, 
                alpha=0.5,
                background_alpha=0.2,  # Keep background transparent
                save_path=file_path
            )
            img_idx += 1

        return

    # determine the desired resolution
    resolution = (args.resolution, args.resolution)

    if "width" in args and "height" in args:
        resolution = (args.height, args.width)     
    
    if args.test_only:
        dataset_test_full, num_classes = get_dataset(args.dataset, args.data, "test", get_transform(train=False, resolution=resolution), args.classes, user_class_mapping=user_class_mapping)
        data_loader_test_full = torch.utils.data.DataLoader(dataset_test_full)#,collate_fn=utils.collate_fn)
    else: 
        # load the train and val datasets
        dataset, num_classes = get_dataset(args.dataset, args.data, "train", get_transform(train=True, resolution=resolution), args.classes, user_class_mapping=user_class_mapping)
        dataset_test, _ = get_dataset(args.dataset, args.data, "val", get_transform(train=False, resolution=resolution), args.classes, user_class_mapping=user_class_mapping)
        verify_dataset_labels([dataset,dataset_test], verbose=True)

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=args.workers,
            collate_fn=utils.collate_fn, drop_last=True)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1,
            sampler=test_sampler, num_workers=args.workers,
            collate_fn=utils.collate_fn)

        print("=> training with dataset: '{:s}' (train={:d}, val={:d})".format(args.dataset, len(dataset), len(dataset_test)))
        print("=> training with resolution: {:d}x{:d}, {:d} classes".format(resolution[1], resolution[0], num_classes))
        print("=> training with model: {:s}".format(args.arch))

    # create the segmentation model
    model = segmentation.__dict__[args.arch](
        num_classes=num_classes,
        aux_loss=args.aux_loss,
        pretrained=args.pretrained
    )
    model.to(device)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # eval-only mode
    if args.test_only:
        # Run this below to verify that all datasets ("train", "val", "test") have consistent class mappings
        # verify_dataset_labels([dataset, dataset_test, dataset_test_full], verbose=True)

        try:
            # Check that the dataset has been loaded correctly
            if not data_loader_test_full:
                print("ERROR: The test dataset hasn't been created properly - exiting...")
                return
        except NameError:
            print("ERROR: data_loader_test_full is not defined - exiting...")
            return
        
        # Create the directories for storing debug images
        visualise_dir = Path(f"{str(Path(args.data))}_test")
        if args.map_classes:
            visualise_dir = Path(f"{str(visualise_dir)}_mapped")
        else:
            visualise_dir = Path(f"{str(visualise_dir)}_original")
        visualise_dir.mkdir(exist_ok=True)
        
        # Run evaluation
        print('yep')
        confmat = evaluate(model, data_loader_test_full, device=device, num_classes=num_classes, visualise_dir=visualise_dir)
        print(confmat)
        return

    # create the optimizer
    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]

    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # training loop
    start_time = time.time()
    best_IoU = 0.0

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train the model over the next epoc
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq)

        # test the model on the val dataset
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)

        # save model checkpoint
        checkpoint_path = os.path.join(args.model_dir, 'model_{:04d}.pth'.format(epoch))

        utils.save_on_master(
            {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
                'arch': args.arch,
                'dataset': args.dataset,                
                'num_classes': num_classes,
                'resolution': resolution,
                'accuracy': confmat.acc_global,
                'mean_IoU': confmat.mean_IoU
            },
            checkpoint_path)

        print('saved checkpoint to:  {:s}  ({:.3f}% mean IoU, {:.3f}% accuracy)'.format(checkpoint_path, confmat.mean_IoU, confmat.acc_global))

        if confmat.mean_IoU > best_IoU:
            best_IoU = confmat.mean_IoU
            best_path = os.path.join(args.model_dir, 'model_best.pth')
            shutil.copyfile(checkpoint_path, best_path)
            print('saved best model to:  {:s}  ({:.3f}% mean IoU, {:.3f}% accuracy)'.format(best_path, best_IoU, confmat.acc_global))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    args = parse_args()
    main(args)

