
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision import tv_tensors


class SegformerDataset(Dataset):

    def __init__(self, dataset_dir: Path, transforms=None):
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.input_images_dir = dataset_dir / "input_images"
        self.mask_images_dir = dataset_dir / "sf_mask_indices"

        # Get a list of files.
        image_files = self.input_images_dir.rglob(f"*.png")
        image_files = list(image_files)

        # Remove the input dir, so we can also get masks.
        image_files = [f.relative_to(self.input_images_dir) for f in image_files]
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        f = self.image_files[idx]
        img_path = self.input_images_dir / f
        mask_path = self.mask_images_dir / f

        # Read images, removing alpha channel.
        input_image = decode_image(str(img_path))[:3, : , :]
        mask_image = decode_image(str(mask_path)).squeeze().to(torch.int64)

        input_image = tv_tensors.Image(input_image)
        mask_image = tv_tensors.Mask(mask_image)

        if self.transforms is not None:
            input_image, mask_image = self.transforms(input_image, mask_image)

        return input_image, mask_image


def main():
    print(torch.cuda.is_available())

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # TODO: Maybe some random colour scaling.
    transforms = v2.Compose([
        v2.RandomResizedCrop(size=(512, 512), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

    dataset = SegformerDataset(Path("/home/nvidia/data/segformer_datasets/251112"), transforms=transforms)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=1)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(inputs.shape)
        print(targets.shape)
        print(inputs.dtype)
        print(targets.dtype)

        num_samples = inputs.shape[0]
        fig, axes = plt.subplots(num_samples, 2, squeeze=False, figsize=(20, 20), tight_layout=True)

        for i in range(num_samples):
            img = inputs[i].cpu().numpy().transpose(1, 2, 0)[:, :, :3]
            mask = targets[i].cpu().numpy().squeeze()

            # Un-normalise input image.
            img = (img * np.array(std)) + np.array(mean)

            ax = axes[i, 0]
            ax.imshow(img)

            ax = axes[i, 1]
            ax.imshow(mask)

        plt.savefig("/home/nvidia/data/segformer_datasets/batch.png")

        return


if __name__ == '__main__':
    main()
