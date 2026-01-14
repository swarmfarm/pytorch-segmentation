import argparse
import datetime
import time
import os
import shutil
from pathlib import Path
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn
from torchvision.transforms import v2
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from models import segmentation

from datasets.swarmfarm import SwarmfarmDataset

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

    # parser.add_argument('data', metavar='DIR', help='path to dataset')
    # parser.add_argument('--dataset', default='voc', help='dataset type: voc, voc_aug, coco, cityscapes, deepscene, mhp, nyu, sun, custom (default: voc)')
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

    args = parser.parse_args()
    return args


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
def evaluate(model, criterion, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    val_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)

            loss = criterion(output, target)
            batch_size = image.shape[0]
            val_loss += loss.item() * batch_size
            num_samples += batch_size

            output = output['out']
            # Take argmax to determine predicted class at each pixel.
            output_classes = output.argmax(1)
            confmat.update(target.flatten(), output_classes.flatten())

        confmat.reduce_from_all_processes()

    avg_val_loss = val_loss / num_samples

    return avg_val_loss, confmat


def output_sample(model, model_pretrained, device, train_dir: Path, epoch: int, data_loader, num_batches, mean, std, class_colours, pretrained_sf_id_mapping, prefix):

    with torch.no_grad():

        batch_num = 0
        batch_imgs = []
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

            cv2.imwrite(train_dir / f"{prefix}_epoch-{epoch:04d}_batch-{batch_num:04d}.png", batch_img)

            batch_imgs.append(batch_img)

            batch_num += 1
            if batch_num >= num_batches:
                break

    return batch_imgs


#
# train for one epoch over the dataset
#
def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0.0
    num_samples = 0
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        batch_size = image.shape[0]
        train_loss += loss.item() * batch_size
        num_samples += batch_size

    avg_train_loss = train_loss / num_samples
    return avg_train_loss


#
# main training function
#
def train(args):
    
    # Map from cityscapes names to swarmfarm names.
    # TODO: Move to utils function in mlops.
    cs_sf_name_mapping = {}
    with open("/home/paperspace/data/segnet_training/cityscapes_swarmfarm_mapping.csv", 'r') as f:
        lines = f.readlines()
        lines = [l.strip().split(',') for l in lines if l.strip()]
        cs_sf_name_mapping = {l[0]: l[1] for l in lines}

    # Read mappings from class indices to names and colours.
    # TODO: Move to utils function in mlops.
    def read_names_colours(csv_file):
        names = {}
        colours = {}
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            lines = [l for l in lines if not l.strip().startswith("#")]
            lines = [l.strip().split(',') for l in lines if l.strip()]
            names = {int(l[0]): l[1] for l in lines}
            colours = {int(l[0]): (int(l[2]), int(l[3]), int(l[4])) for l in lines}
        return names, colours
    cs_names, cs_colours = read_names_colours("/home/paperspace/data/segnet_training/cityscapes_classes.csv")
    sf_names, sf_colours = read_names_colours("/home/paperspace/data/segnet_training/swarmfarm_classes.csv")

    # Map from cityscapes ids to swarmfarm ids.
    # TODO: Move to utils function in mlops.
    sf_ids = {n: i for i, n in sf_names.items()}
    cs_sf_id_mapping = {}
    for cs_id, cs_name in cs_names.items():
        sf_name = cs_sf_name_mapping[cs_name]
        sf_id = sf_ids[sf_name]
        cs_sf_id_mapping[cs_id] = sf_id

    # Directory to store all training results.
    run_name = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    train_dir = Path(args.model_dir) / run_name
    train_dir.mkdir(exist_ok=False)

    # Tensorboard logs
    tensorboard_dir = Path(args.model_dir) / "tensorboard_runs" / run_name
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    device = torch.device(args.device)

    resolution = (args.resolution, args.resolution)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transforms = v2.Compose([
        v2.Resize((540, 960)),
        v2.RandomResizedCrop(size=256, antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    # Replace "unknown" mask values, that may have been inserted during dataset creation.
    class_mapping = {
        10: 0,
    }

    # For batch 9, use annotated masks merged with Segformer masks.
    dataset_batch9 = SwarmfarmDataset(
        Path("/home/paperspace/data/segnet_training/datasets/SWA-001-009-Video-Annotation-ds-98611a459fdd4846bb84ffe6febf98f8_2025-10-07_12-35-02_6763aa85eea8ffdac30ba12a_68e508f57175641d11102028"), 
        mask_subdir="sf_mask_indices_merged",
        class_mapping=class_mapping,
        transforms=transforms
        )
    
    dataset_batch10 = SwarmfarmDataset(
        Path("/home/paperspace/data/segnet_training/datasets/SWA-001-010-Video-Annotation-ds-e9b42db94c3845b69f52dab8f488de82"), 
        mask_subdir="sf_mask_indices",
        class_mapping=class_mapping,
        transforms=transforms
        )
    
    dataset_batch11 = SwarmfarmDataset(
        Path("/home/paperspace/data/segnet_training/datasets/SWA-001-011-Video-Annotation-ds-08e197904470459e8cb2ca76209d9ef0_2025-11-13_14-02-42_685e558c28a9857d720a6d94_6915e501bf9b9b256e013a83"), 
        class_mapping=class_mapping,
        transforms=transforms
        )
    
    dataset = ConcatDataset([dataset_batch9, dataset_batch10, dataset_batch11])

    num_classes = 6  # 21 in the COCO pretrained model.

    train_sampler = None
    test_sampler = None
    if 1:
        # Use the training dataset for training and testing, with a random split.
        validation_split = 0.2
        shuffle_dataset = True
        random_seed = 1

        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers)
        data_loader_test = DataLoader(dataset, batch_size=8, sampler=test_sampler, num_workers=args.workers)
    else:
        # Create a separate test dataset.
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        transforms_test = v2.Compose([
            v2.Resize((540, 960)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ])

        dataset_test = SwarmfarmDataset(
            Path("/home/paperspace/data/svo-inference/251112_batch-9_tmp"), 
            class_mapping=class_mapping,
            transforms=transforms_test
            )
        data_loader_test = DataLoader(dataset_test, batch_size=4, shuffle=True, num_workers=args.workers)

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
        avg_val_loss, confmat = evaluate(model, criterion, data_loader_test, device=device, num_classes=num_classes)
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

    output_sample(model, model_cs, device, train_dir, 999, data_loader_test, 1, mean, std, sf_colours, cs_sf_id_mapping, prefix="test_epoch")
    output_sample(model, model_cs, device, train_dir, 999, data_loader, 1, mean, std, sf_colours, cs_sf_id_mapping, prefix="train_epoch")

    # training loop
    start_time = time.time()
    best_IoU = 0.0

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train the model over the next epoc
        avg_train_loss = train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq)
        print(f"Average train loss for epoch {epoch}: {avg_train_loss:.4f}")

        # test the model on the val dataset
        avg_val_loss, confmat = evaluate(model, criterion, data_loader_test, device=device, num_classes=num_classes)
        print(f"Average val loss for epoch {epoch}: {avg_val_loss:.4f}")
        print(confmat)

        writer.add_scalar("loss/train", avg_train_loss, epoch)
        writer.add_scalar("loss/val", avg_val_loss, epoch)
        acc_global, acc, iou = confmat.compute()
        acc = acc.cpu().numpy()
        iou = iou.cpu().numpy()
        miou = np.mean(iou)
        writer.add_scalar("accuracy/val", acc_global, epoch)
        writer.add_scalar("iou/val", miou, epoch)
        for class_id, class_name in sf_names.items():
            writer.add_scalar(f"accuracy/val_{class_name}", acc[class_id], epoch)
            writer.add_scalar(f"iou/val_{class_name}", iou[class_id], epoch)
        writer.flush()

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
            best_path = os.path.join(train_dir, 'model_best.pth')
            shutil.copyfile(checkpoint_path, best_path)
            print('saved best model to:  {:s}  ({:.3f}% mean IoU, {:.3f}% accuracy)'.format(best_path, best_IoU, confmat.acc_global))

        # Save some predictions.
        test_samples = output_sample(model, model_cs, device, train_dir, epoch, data_loader_test, 1, mean, std, sf_colours, cs_sf_id_mapping, prefix="test")
        output_sample(model, model_cs, device, train_dir, epoch, data_loader, 1, mean, std, sf_colours, cs_sf_id_mapping, prefix="train")

        if len(test_samples) > 1:
            # Join the samples.
            test_samples = np.concat(test_samples, axis=1)
        else:
            test_samples = test_samples[0]
        print(test_samples.shape)
        writer.add_image(f"images/val", test_samples[:, :, ::-1], epoch, dataformats='HWC')

    writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def main():
    args = parse_args()
    
    if 1:
        args.device = "cuda"
        args.batch_size = 64  # 64 for resnet18, 16 for resnet50, 8 for resnet101
        args.resolution = 512
        args.workers = 8
        args.arch = "fcn_resnet34"  # 18, 50, 101
        args.dataset = "swarmfarm"
        args.aux_loss = False
        args.pretrained = False  # Setting to False will still use a pretrained backbone.
        args.distributed = False
        args.resume = False
        args.test_only = False
        args.model_dir = "/home/paperspace/data/segnet_training/training_runs"
        args.epochs = 200

    train(args)


if __name__ == "__main__":
    main()
