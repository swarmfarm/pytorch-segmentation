from pathlib import Path
import numpy as np
import argparse

from sf_seg_model import SwarmfarmSegModel

from swarmfarm_computer_vision.mlops.inference.segmentation.inference import seg_inference_s3


model_names = [
    "fcn_resnet18",
    "fcn_resnet34",
    "fcn_resnet50",
    "fcn_resnet101",
]


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training')

    parser.add_argument('-a', '--arch', metavar='ARCH', default='fcn_resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: fcn_resnet18)')
    parser.add_argument('-f', '--model-file', help='path to .pth model file')
    parser.add_argument('--s3-image-dir', metavar='DIR', help='path to where images are stored in S3')
    parser.add_argument('--image-dir', metavar='DIR', help='path to where to store local input image files')
    parser.add_argument('--out-dir', metavar='DIR', help='path to where to store output files')
    parser.add_argument('--prefixes-file', help='path to a file containing prefixes defining what files to run inference on')
    parser.add_argument('--downsample', default=2, type=int, metavar='C', help='number of times to downsample')
    parser.add_argument('--classes', default=6, type=int, metavar='C', help='number of classes in your dataset (outputs)')
    parser.add_argument('--device', default='cuda', help='device')

    args = parser.parse_args()
    return args


def run_inference(args):

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    inference_data_dir = Path("/home/paperspace/data/segnet_training")
    # Definitions of the model and target classes.
    # This model has been trained to predict swarmfarm classes directly.
    model_classes_file = inference_data_dir /"swarmfarm_classes.csv"

    model = SwarmfarmSegModel(
        args.arch,
        args.classes,
        args.model_file,
        mean=mean,
        std=std,
        device=args.device,
    )

    seg_inference_s3(
        model,
        None,
        None,
        args.downsample,
        args.s3_image_dir,
        image_dir=Path(args.image_dir),
        out_dir=Path(args.out_dir),
        model_classes_file=model_classes_file,
        target_classes_file=None,
        model_to_target_mapping_file=None,
        interval=1,
        prefixes_file=args.prefixes_file,
        render_model_overlay=True,
        render_target_overlay=False,
        render_vis=False,
    )


def main():
    args = parse_args()

    if 0:
        # Overwrite args for testing.

        args.device = "cuda"
        args.arch = "fcn_resnet50"  # 18, 34, 50, 101

        #args.model_file = "/home/paperspace/data/segnet_training/training_runs/251118_011803/model_best.pth"
        # args.model_file = "/home/paperspace/data/segnet_training/training_runs/251118_011803_resnet18/model_best.pth"
        #args.model_file = "/home/paperspace/data/segnet_training/training_runs/251125_030424_resnet101/model_best.pth"
        args.model_file = "/home/paperspace/data/segnet_training/training_runs/251126_001729_resnet50/model_best.pth"
        # args.model_file = "/home/paperspace/data/segnet_training/training_runs/251126_081149_resnet34/model_best.pth"
        args.downsample = 2

        if 0:
            # Batch 9
            args.s3_image_dir = "s3://swarmfarm-vision/data_collection/sb-0014/images"
            args.image_dir = "/home/paperspace/data/segnet_training/results/tmp/input_images"
            args.out_dir = "/home/paperspace/data/segnet_training/results/tmp"
            args.prefixes_file = "/home/paperspace/data/segnet_training/batch-9.txt"
        elif 0:
            # Ray obstacles dataset.
            args.s3_image_dir = "s3://swarmfarm-vision/data_collection/sb-0026/images"
            args.image_dir = "/home/paperspace/data/segnet_training/datasets/batch-rayobstacles/input_images"
            args.out_dir = "/home/paperspace/data/segnet_training/results/tmp"
            args.prefixes_file = "/home/paperspace/data/segnet_training/batch-rayobstacles.txt"
        elif 0:
            # Ray crop dataset.
            args.s3_image_dir = "s3://swarmfarm-vision/data_collection/sb-0026/images"
            args.image_dir = "/home/paperspace/data/segnet_training/datasets/batch-raycrop/input_images"
            args.out_dir = "/home/paperspace/data/segnet_training/results/tmp"
            args.prefixes_file = "/home/paperspace/data/segnet_training/batch-raycrop.txt"
        elif 1:
            # BB 202509 dataset.
            args.s3_image_dir = "s3://swarmfarm-vision/data_collection/sb-0172/images"
            args.image_dir = "/home/paperspace/data/segnet_training/datasets/batch-bb202509/input_images"
            args.out_dir = "/home/paperspace/data/segnet_training/results/tmp"
            args.prefixes_file = "/home/paperspace/data/segnet_training/batch-bb202509.txt"

    run_inference(args)


if __name__ == '__main__':
    main()
