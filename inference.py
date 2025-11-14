from pathlib import Path
import numpy as np
import argparse

from sf_seg_model import SwarmfarmSegModel

from swarmfarm_computer_vision.mlops.inference.segmentation.segformer import seg_inference_s3

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
    checkpoint_file = "/home/paperspace/data/segnet_training/251113_020505/model_12.pth"

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    inference_data_dir = Path("/home/paperspace/data/svo-inference")
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

    seg_inference_s3(
        model,
        None,
        None,
        "s3://swarmfarm-vision/data_collection/sb-0014/images",
        Path("/home/paperspace/data/svo-inference/251112_batch-9_pth"),
        model_classes_file=model_classes_file,
        target_classes_file=target_classes_file,
        model_to_target_mapping_file=model_to_target_mapping_file,
        output_dir=Path("/home/paperspace/data/svo-inference/out"),
        interval=1,
        prefixes_file="/home/paperspace/data/svo-inference/batch-9.txt",
        render_vis=False,
    )


if __name__ == '__main__':
    main()
