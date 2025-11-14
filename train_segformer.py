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
import numpy as np
import cv2

import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torchvision
from torchvision.transforms import v2
from models import segmentation

from datasets.coco_utils import get_coco
from datasets.cityscapes_utils import get_cityscapes
from datasets.deepscene import DeepSceneSegmentation
from datasets.custom_dataset import CustomSegmentation
from datasets.mhp import MHPSegmentation
from datasets.nyu import NYUDepth
from datasets.sun import SunRGBDSegmentation

from datasets.segformer import SegformerDataset

import transforms as T
import utils

torch.manual_seed(123)

model_names = sorted(name for name in segmentation.__dict__
    if name.islower() and not name.startswith("__")
    and callable(segmentation.__dict__[name]))

#
# parse command-line arguments
#
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training')

    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--dataset', default='voc', help='dataset type: voc, voc_aug, coco, cityscapes, deepscene, mhp, nyu, sun, custom (default: voc)')
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
    parser.add_argument('--model-dir', default='.', help='path where to save output models')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--pretrained", dest="pretrained", help="Use pre-trained models (only supported for fcn_resnet101)", action="store_true")

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args


#
# load desired dataset
#
def get_dataset(name, path, image_set, transform, num_classes):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "voc": (path, torchvision.datasets.VOCSegmentation, num_classes),
        "voc_aug": (path, sbd, num_classes),
        "coco": (path, get_coco, num_classes),
        "cityscapes": (path, get_cityscapes, num_classes),
        "deepscene": (path, DeepSceneSegmentation, 5),
        "mhp": (path, MHPSegmentation, num_classes),
        "nyu": (path, NYUDepth, num_classes),
        "sun": (path, SunRGBDSegmentation, num_classes),
        "custom": (path, CustomSegmentation, num_classes)
    }
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
def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def output_sample(model, model_pretrained, device, train_dir: Path, epoch: int, data_loader, num_batches, mean, std, class_colours, pretrained_sf_id_mapping):

    # epoch_dir = train_dir / f"epoch_{epoch:04d}"
    # epoch_dir.mkdir(exist_ok=True)
    with torch.no_grad():

        batch_num = 0
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            output = model(image)
            logits = output['out']

            if model_pretrained is not None:
                output_pretrained = model_pretrained(image)
                logits_pretrained = output_pretrained['out']

            batch_size = image.shape[0]
            h = image.shape[2]
            w = image.shape[3]

            # Create an image for the batch.
            num_imgs = 3 if model_pretrained is None else 4
            batch_img = np.zeros((batch_size * h, w * num_imgs, 3), dtype=np.uint8)
            for b in range(batch_size):
                image_sample = image[b].cpu().numpy().transpose(1, 2, 0)
                target_sample = target[b].cpu().numpy()
                #logits_sample = logits[b].cpu().numpy()
                pred_sample = logits[b].argmax(dim=0).cpu().numpy()

                if model_pretrained is not None:
                    pred_sample_pretrained = logits_pretrained[b].argmax(dim=0).cpu().numpy()
                    # Convert to SF classes.
                    pred_img_pretrained = np.zeros((pred_sample_pretrained.shape[0], pred_sample_pretrained.shape[1], 3), dtype=np.uint8)
                    for id_pretrained, id_sf in pretrained_sf_id_mapping.items():
                        colour = class_colours[id_sf]
                        pred_img_pretrained[pred_sample_pretrained == id_pretrained] = colour

                image_sample = (image_sample * np.array(std)) + np.array(mean)
                image_sample = (image_sample * 255).astype(np.uint8)

                # Convert to colours.
                pred_img = np.zeros((pred_sample.shape[0], pred_sample.shape[1], 3), dtype=np.uint8)
                for id, colour in class_colours.items():
                    pred_img[pred_sample == id] = colour

                target_img = np.zeros((target_sample.shape[0], target_sample.shape[1], 3), dtype=np.uint8)
                for id, colour in class_colours.items():
                    target_img[target_sample == id] = colour

                batch_img[b * h: b * h + h, :w, :] = image_sample[:, :, ::-1]
                batch_img[b * h: b * h + h, w: 2 * w, :] = pred_img[:, :, ::-1]
                batch_img[b * h: b * h + h, 2 * w: 3 * w, :] = target_img[:, :, ::-1]
                if model_pretrained is not None:
                    batch_img[b * h: b * h + h, 3 * w: 4 * w, :] = pred_img_pretrained[:, :, ::-1]

            cv2.imwrite(train_dir / f"test_epoch-{epoch:04d}_batch-{batch_num:04d}.png", batch_img)

            batch_num += 1
            if batch_num >= num_batches:
                break


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
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


#
# main training function
#
def main(args):
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

    args.epochs = 20
    args.print_freq = 1
    args.lr = 0.01  # TODO: Experiment with this.
    args.momentum = 0.9  # TODO: Experiment with this.
    args.weight_decay = 1e-4  # TODO: Experiment with this.
    
    # Map from cityscapes names to swarmfarm names.
    cs_sf_name_mapping = {}
    with open("/home/paperspace/data/svo-inference/cityscapes_swarmfarm_mapping.csv", 'r') as f:
        lines = f.readlines()
        lines = [l.strip().split(',') for l in lines if l.strip()]
        cs_sf_name_mapping = {l[0]: l[1] for l in lines}

    # Read mappings from class indices to names and colours.
    def read_names_colours(csv_file):
        names = {}
        colours = {}
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            lines = [l.strip().split(',') for l in lines if l.strip()]
            names = {int(l[0]): l[1] for l in lines}
            colours = {int(l[0]): (int(l[2]), int(l[3]), int(l[4])) for l in lines}
        return names, colours
    cs_names, cs_colours = read_names_colours("/home/paperspace/data/svo-inference/cityscapes_classes.csv")
    sf_names, sf_colours = read_names_colours("/home/paperspace/data/svo-inference/swarmfarm_classes.csv")

    # Map from cityscapes ids to swarmfarm ids.
    sf_ids = {n: i for i, n in sf_names.items()}
    cs_sf_id_mapping = {}
    for cs_id, cs_name in cs_names.items():
        sf_name = cs_sf_name_mapping[cs_name]
        sf_id = sf_ids[sf_name]
        cs_sf_id_mapping[cs_id] = sf_id

    # Directory to store all training results.
    train_dir = Path(args.model_dir) / datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    train_dir.mkdir(exist_ok=False)

    

    # if args.model_dir:
    #     utils.mkdir(args.model_dir)

    # utils.init_distributed_mode(args)
    # print(args)

    device = torch.device(args.device)

    if 0:
        # determine the desired resolution
        resolution = (args.resolution, args.resolution)

        if "width" in args and "height" in args:
            resolution = (args.height, args.width)     
        
        # load the train and val datasets
        dataset, num_classes = get_dataset(args.dataset, args.data, "train", get_transform(train=True, resolution=resolution), args.classes)
        dataset_test, _ = get_dataset(args.dataset, args.data, "val", get_transform(train=False, resolution=resolution), args.classes)
    else:
        resolution = (args.resolution, args.resolution)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # TODO: Add resize to half res at start.
        transforms = v2.Compose([
            v2.RandomResizedCrop(size=resolution, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])
        dataset = SegformerDataset(Path("/home/paperspace/data/svo-inference/251112_batch-11"), transforms=transforms)
        
        # TODO: Add resize to half res at start.
        transforms_test = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])
        dataset_test = SegformerDataset(Path("/home/paperspace/data/svo-inference/251112_batch-9"), transforms=transforms_test)

        num_classes = 6  # 21 in the COCO pretrained model.

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    # else:
    #     train_sampler = torch.utils.data.RandomSampler(dataset)
    #     test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=args.batch_size,
    #     sampler=train_sampler, num_workers=args.workers,
    #     collate_fn=utils.collate_fn, drop_last=True)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # data_loader_test = torch.utils.data.DataLoader(
    #     dataset_test, batch_size=1,
    #     sampler=test_sampler, num_workers=args.workers,
    #     collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=4, shuffle=True, num_workers=args.workers)

    print("=> training with dataset: '{:s}' (train={:d}, val={:d})".format(args.dataset, len(dataset), len(dataset_test)))
    # print("=> training with resolution: {:d}x{:d}, {:d} classes".format(resolution[1], resolution[0], num_classes))
    print("=> training with model: {:s}".format(args.arch))

    # Pre-trained model for comparison.
    model_cs = None
    if 0:
        model_cs = segmentation.__dict__["fcn_resnet101"](pretrained=True)
        model_cs.to(device)
        model_cs.eval()

    # create the segmentation model
    model = segmentation.__dict__[args.arch](num_classes=num_classes, aux_loss=args.aux_loss, pretrained=args.pretrained)
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
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
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

    output_sample(model, model_cs, device, train_dir, 999, data_loader_test, 1, mean, std, sf_colours, cs_sf_id_mapping)

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
        checkpoint_path = os.path.join(train_dir, 'model_{}.pth'.format(epoch))

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

        # Save some predictions.
        output_sample(model, model_cs, device, train_dir, epoch, data_loader_test, 1, mean, std, sf_colours, cs_sf_id_mapping)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    args = None#parse_args()
    main(args)

