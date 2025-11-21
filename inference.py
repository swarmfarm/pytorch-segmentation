from pathlib import Path
import numpy as np
import argparse

from sf_seg_model import SwarmfarmSegModel

from swarmfarm_computer_vision.mlops.inference.segmentation.inference import seg_inference_s3

def main():

    args = argparse.Namespace()
    args.device = "cuda"
    args.batch_size = 4
    args.resolution = 512
    args.workers = 1
    args.arch = "fcn_resnet18"
    args.dataset = "segformer"
    args.aux_loss = False  # TODO: See what this does exactly.
    args.pretrained = False  # TODO: We should use pretrained model for most of the network.
    args.distributed = False
    args.resume = False
    args.test_only = False
    args.model_dir = "/home/paperspace/data/segnet_training"
    num_classes = 6  # TODO: Get this from the below model classes file.
    #checkpoint_file = "/home/paperspace/data/segnet_training/training_runs/251118_011803/model_best.pth"
    checkpoint_file = "/home/paperspace/data/segnet_training/training_runs/251121_025426/model_best.pth"

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    downsample = 2

    inference_data_dir = Path("/home/paperspace/data/segnet_training")
    # Definitions of the model and target classes.
    # This model has been trained to predict swarmfarm classes directly.
    model_classes_file = inference_data_dir /"swarmfarm_classes.csv"
    target_classes_file = None
    # File mapping from the model output classes to target classes.
    model_to_target_mapping_file = None

    model = SwarmfarmSegModel(
        args.arch,
        num_classes,
        args.aux_loss,
        False, 
        checkpoint_file,
        mean=mean,
        std=std,
        device=args.device,
    )

    if 0:
        seg_inference_s3(
            model,
            None,
            None,
            downsample,
            "s3://swarmfarm-vision/data_collection/sb-0014/images",
            # Path("/home/paperspace/data/svo-inference/251114_225236_batch-9_segnet-pth"),
            Path("/home/paperspace/data/segnet_training/results/251121_025426_batch-9_segnet-pth"),
            model_classes_file=model_classes_file,
            target_classes_file=target_classes_file,
            model_to_target_mapping_file=model_to_target_mapping_file,
            interval=1,
            prefixes_file="/home/paperspace/data/segnet_training/batch-9.txt",
            render_model_overlay=True,
            render_target_overlay=False,
            render_vis=False,
        )
    elif 1:
        # Ray obstacles dataset.
        seg_inference_s3(
            model,
            None,
            None,
            downsample,
            "s3://swarmfarm-vision/data_collection/sb-0026/images",
            Path("/home/paperspace/data/segnet_training/results/251121_025426_batch-rayobstacles_segnet-pth"),
            model_classes_file=model_classes_file,
            target_classes_file=target_classes_file,
            model_to_target_mapping_file=model_to_target_mapping_file,
            interval=1,
            prefixes_file="/home/paperspace/data/segnet_training/batch-rayobstacles.txt",
            render_model_overlay=True,
            render_target_overlay=False,
            render_vis=False,
        )
    elif 0:
        # Ray crops dataset.
        seg_inference_s3(
            model,
            None,
            None,
            downsample,
            "s3://swarmfarm-vision/data_collection/sb-0026/images",
            Path("/home/paperspace/data/segnet_training/results/251121_025426_batch-raycrop_segnet-pth"),
            model_classes_file=model_classes_file,
            target_classes_file=target_classes_file,
            model_to_target_mapping_file=model_to_target_mapping_file,
            interval=1,
            prefixes_file="/home/paperspace/data/segnet_training/batch-raycrop.txt",
            render_model_overlay=True,
            render_vis=False,
        )


if __name__ == '__main__':
    main()
