from __future__ import print_function
from collections import defaultdict, deque
import datetime
import math
import time
import torch
import torch.distributed as dist

import errno
import os


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
        self.acc_global = 0.0
        self.mean_IoU = 0.0

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
      
        # not all object classes may have data from the val category
        h_sum1 = h.sum(1)
        h_sum1[h_sum1 == 0] = 1

        if not torch.equal(h.sum(1), h_sum1):
           print('Test:  Warning -- some classes may be missing validation examples')

        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h_sum1 + h.sum(0) - torch.diag(h))
        self.acc_global = acc_global.item() * 100
        self.mean_IoU = iu.mean().item() * 100
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / MB))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{}  Total time: {}'.format(header, total_time_str))


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


class MaskOverlay(object):
    """
    Creates RGB mask visualizations from COCO ID mask images using a color lookup dictionary.
    
    Example usage:
        color_map = {
            0: (0, 0, 0),      # background - black
            1: (255, 0, 0),    # class 1 - red
            2: (0, 255, 0),    # class 2 - green
            3: (0, 0, 255),    # class 3 - blue
        }
        overlay = MaskOverlay(color_map)
        rgb_mask = overlay(mask_tensor_or_array)
    """
    
    def __init__(self, color_map=None, default_color=(128, 128, 128)):
        """
        Args:
            color_map (dict): Dictionary mapping class IDs to RGB tuples
            default_color (tuple): RGB color for unmapped class IDs
        """
        self.color_map = color_map or {}
        self.default_color = default_color
        
        # Pre-compute color array for efficient lookup
        if self.color_map:
            max_id = max(self.color_map.keys())
            self.color_array = torch.zeros((max_id + 1, 3), dtype=torch.uint8)
            for class_id, color in self.color_map.items():
                self.color_array[class_id] = torch.tensor(color, dtype=torch.uint8)
        else:
            self.color_array = None
    
    def __call__(self, mask):
        """
        Convert mask to RGB visualization.
        
        Args:
            mask: Input mask as torch.Tensor or numpy array with shape (H, W)
                  Values should be class IDs (0, 1, 2, ...)
        
        Returns:
            RGB image as torch.Tensor with shape (3, H, W) or (H, W, 3) depending on input type
        """
        import numpy as np
        
        # Handle different input types
        if isinstance(mask, torch.Tensor):
            return self._tensor_to_rgb(mask)
        elif isinstance(mask, np.ndarray):
            return self._numpy_to_rgb(mask)
        else:
            raise TypeError(f"Unsupported mask type: {type(mask)}")
    
    def _tensor_to_rgb(self, mask):
        """Convert torch.Tensor mask to RGB tensor."""
        if mask.dim() != 2:
            raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
        
        h, w = mask.shape
        rgb_mask = torch.zeros((3, h, w), dtype=torch.uint8, device=mask.device)
        
        # Convert mask to numpy for easier processing, then back to tensor
        mask_np = mask.cpu().numpy().astype(np.uint8)
        rgb_np = self._numpy_to_rgb(mask_np)
        
        # Convert back to tensor format (H, W, 3) -> (3, H, W)
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).to(mask.device)
        
        return rgb_tensor
    
    def _numpy_to_rgb(self, mask):
        """Convert numpy array mask to RGB numpy array."""
        import numpy as np
        
        if mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
        
        h, w = mask.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        if self.color_map:
            # Map each class ID to its color
            for class_id, color in self.color_map.items():
                class_mask = (mask == class_id)
                rgb_mask[class_mask] = color
            
            # Handle unmapped class IDs with default color
            mapped_ids = set(self.color_map.keys())
            unique_ids = set(np.unique(mask))
            unmapped_ids = unique_ids - mapped_ids
            
            for unmapped_id in unmapped_ids:
                if unmapped_id >= 0:  # Ignore negative values
                    unmapped_mask = (mask == unmapped_id)
                    rgb_mask[unmapped_mask] = self.default_color
        else:
            # No color map provided, use default color for all non-zero pixels
            non_zero_mask = mask > 0
            rgb_mask[non_zero_mask] = self.default_color
        
        return rgb_mask
    
    def create_legend(self, class_names=None, save_path=None):
        """
        Create a color legend for the mask overlay.
        
        Args:
            class_names (dict): Optional mapping from class IDs to names
            save_path (str): Optional path to save the legend image
        
        Returns:
            PIL Image of the color legend
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            raise ImportError("PIL is required for legend creation")
        
        if not self.color_map:
            print("No color map available for legend creation")
            return None
        
        # Legend parameters
        box_size = 30
        text_margin = 10
        line_height = box_size + 5
        
        # Calculate image dimensions
        max_class_id = max(self.color_map.keys())
        legend_height = len(self.color_map) * line_height + 20
        
        # Estimate text width (approximate)
        if class_names:
            max_text_length = max(len(str(class_names.get(cid, f"Class {cid}"))) 
                                for cid in self.color_map.keys())
        else:
            max_text_length = max(len(f"Class {cid}") for cid in self.color_map.keys())
        
        legend_width = box_size + text_margin + max_text_length * 8 + 20
        
        # Create legend image
        legend_img = Image.new('RGB', (legend_width, legend_height))
        legend_img.paste((255, 255, 255), (0, 0, legend_width, legend_height))
        draw = ImageDraw.Draw(legend_img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # Draw legend entries
        y_offset = 10
        for class_id in sorted(self.color_map.keys()):
            color = self.color_map[class_id]
            
            # Draw color box
            draw.rectangle([10, y_offset, 10 + box_size, y_offset + box_size], 
                         fill=color, outline=(0, 0, 0))
            
            # Draw text
            if class_names and class_id in class_names:
                text = f"{class_id}: {class_names[class_id]}"
            else:
                text = f"Class {class_id}"
            
            draw.text((10 + box_size + text_margin, y_offset + box_size//4), 
                     text, fill=(0, 0, 0), font=font)
            
            y_offset += line_height
        
        if save_path:
            legend_img.save(save_path)
            print(f"Legend saved to: {save_path}")
        
        return legend_img
    
    def overlay_on_image(self, mask, image, alpha=0.5, save_path=None, background_alpha=0.0):
        """
        Overlay the colored mask onto an RGB image with configurable transparency.
        
        Args:
            mask: Input mask as torch.Tensor or numpy array with shape (H, W)
            image: RGB image as torch.Tensor (3, H, W), numpy array (H, W, 3), or PIL Image
            alpha (float): Transparency of the mask overlay (0.0 = transparent, 1.0 = opaque)
            save_path (str, optional): Path to save the overlaid image
            background_alpha (float): Alpha for background pixels (class 0), default 0.0 (transparent)
        
        Returns:
            PIL Image of the overlaid result
        """
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            raise ImportError("PIL and numpy are required for image overlay")
        
        # Convert mask to numpy if needed and ensure it's the right format
        if isinstance(mask, torch.Tensor):
            original_mask = mask.cpu().numpy().astype(np.uint8)
        else:
            original_mask = np.array(mask).astype(np.uint8)
        
        # Handle different input image types
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:
                # Convert from (3, H, W) to (H, W, 3)
                image_np = image.permute(1, 2, 0).cpu().numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
            else:
                raise ValueError(f"Expected image tensor with shape (3, H, W), got {image.shape}")
        elif isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image_np = image.astype(np.uint8)
            else:
                raise ValueError(f"Expected image array with shape (H, W, 3), got {image.shape}")
        elif hasattr(image, 'convert'):  # PIL Image
            image_np = np.array(image.convert('RGB'))
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Create RGB mask using numpy method (more reliable)
        mask_np = self._numpy_to_rgb(original_mask)
        
        # Ensure dimensions match
        if image_np.shape[:2] != mask_np.shape[:2]:
            # Resize mask to match image dimensions
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray(mask_np)
            mask_pil = mask_pil.resize((image_np.shape[1], image_np.shape[0]), PILImage.NEAREST)
            mask_np = np.array(mask_pil)
            
            # Also resize original mask for alpha calculations
            original_mask_pil = PILImage.fromarray(original_mask)
            original_mask_pil = original_mask_pil.resize((image_np.shape[1], image_np.shape[0]), PILImage.NEAREST)
            original_mask = np.array(original_mask_pil)
        
        # Create per-pixel alpha values
        alpha_mask = np.full(original_mask.shape, alpha, dtype=np.float32)
        
        # Set background pixels (class 0) to background_alpha
        background_pixels = (original_mask == 0)
        alpha_mask[background_pixels] = background_alpha
        
        # Expand alpha mask to 3 channels
        alpha_mask = np.stack([alpha_mask, alpha_mask, alpha_mask], axis=2)
        
        # Convert to float for blending
        image_float = image_np.astype(np.float32)
        mask_float = mask_np.astype(np.float32)
        
        # Blend the images: result = (1-alpha) * image + alpha * mask
        blended = (1.0 - alpha_mask) * image_float + alpha_mask * mask_float
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Convert to PIL Image
        result_img = Image.fromarray(blended)
        
        # Save if path provided
        if save_path:
            result_img.save(save_path)
            print(f"Overlaid image saved to: {save_path}")
        
        return result_img
    
    def create_side_by_side(self, mask, image, save_path=None, titles=None):
        """
        Create a side-by-side comparison of original image, colored mask, and overlay.
        
        Args:
            mask: Input mask as torch.Tensor or numpy array
            image: RGB image in any supported format
            save_path (str, optional): Path to save the comparison image
            titles (list, optional): Titles for [original, mask, overlay] panels
        
        Returns:
            PIL Image of the side-by-side comparison
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
        except ImportError:
            raise ImportError("PIL is required for side-by-side comparison")
        
        # Default titles
        if titles is None:
            titles = ["Original Image", "Colored Mask", "Overlay"]
        
        # Convert image to PIL if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:
                image_np = image.permute(1, 2, 0).cpu().numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
            img_pil = Image.fromarray(image_np)
        elif isinstance(image, np.ndarray):
            img_pil = Image.fromarray(image.astype(np.uint8))
        else:
            img_pil = image.convert('RGB') if hasattr(image, 'convert') else image
        
        # Create colored mask
        rgb_mask = self.__call__(mask)
        if isinstance(rgb_mask, torch.Tensor):
            if rgb_mask.dim() == 3 and rgb_mask.shape[0] == 3:
                mask_np = rgb_mask.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            else:
                mask_np = rgb_mask.astype(np.uint8)
        else:
            mask_np = rgb_mask.astype(np.uint8)
        mask_pil = Image.fromarray(mask_np)
        
        # Create overlay
        overlay_pil = self.overlay_on_image(mask, image, alpha=0.5)
        
        # Get dimensions
        width, height = img_pil.size
        title_height = 30
        
        # Create combined image
        combined_width = width * 3
        combined_height = height + title_height
        combined_img = Image.new('RGB', (combined_width, combined_height), 'white')
        
        # Paste images
        combined_img.paste(img_pil, (0, title_height))
        combined_img.paste(mask_pil, (width, title_height))
        combined_img.paste(overlay_pil, (width * 2, title_height))
        
        # Add titles
        draw = ImageDraw.Draw(combined_img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        for i, title in enumerate(titles):
            text_bbox = draw.textbbox((0, 0), title, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            x_pos = i * width + (width - text_width) // 2
            draw.text((x_pos, 5), title, fill='black', font=font)
        
        # Save if path provided
        if save_path:
            combined_img.save(save_path)
            print(f"Side-by-side comparison saved to: {save_path}")
        
        return combined_img
    
    def debug_mask_colors(self, mask, save_prefix="debug"):
        """
        Debug method to save individual components and analyze color mapping issues.
        
        Args:
            mask: Input mask to debug
            save_prefix (str): Prefix for debug output files
        """
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            raise ImportError("PIL and numpy are required for debugging")
        
        # Convert mask to numpy
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().astype(np.uint8)
        else:
            mask_np = np.array(mask).astype(np.uint8)
        
        print(f"Debug info for mask:")
        print(f"  Shape: {mask_np.shape}")
        print(f"  Dtype: {mask_np.dtype}")
        print(f"  Min value: {mask_np.min()}")
        print(f"  Max value: {mask_np.max()}")
        print(f"  Unique values: {np.unique(mask_np)}")
        
        # Save original mask
        mask_img = Image.fromarray(mask_np)
        mask_img.save(f"{save_prefix}_original_mask.png")
        print(f"Saved original mask to: {save_prefix}_original_mask.png")
        
        # Create and save RGB mask
        rgb_mask = self._numpy_to_rgb(mask_np)
        rgb_img = Image.fromarray(rgb_mask)
        rgb_img.save(f"{save_prefix}_rgb_mask.png")
        print(f"Saved RGB mask to: {save_prefix}_rgb_mask.png")
        
        # Print color mapping info
        print(f"Color mapping used:")
        for class_id, color in self.color_map.items():
            pixel_count = np.sum(mask_np == class_id)
            print(f"  Class {class_id}: {color} ({pixel_count} pixels)")
        
        # Check for unmapped classes
        unique_classes = np.unique(mask_np)
        mapped_classes = set(self.color_map.keys())
        unmapped = set(unique_classes) - mapped_classes
        if unmapped:
            print(f"WARNING: Unmapped classes found: {unmapped}")
            print(f"These will use default color: {self.default_color}")
        
        return rgb_mask
    
    @staticmethod
    def create_default_colormap(num_classes, colormap='tab20'):
        """
        Create a default color mapping for a given number of classes.
        
        Args:
            num_classes (int): Number of classes (including background)
            colormap (str): Matplotlib colormap name
        
        Returns:
            Dictionary mapping class IDs to RGB tuples
            Note: Class 0 always maps to black (0,0,0) and class 255 always maps to white (255,255,255)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            # Fallback to simple rainbow colors
            colors = {}
            colors[0] = (0, 0, 0)      # Background always black
            colors[255] = (255, 255, 255)  # Class 255 always white
            
            # Handle regular classes (1 to num_classes-1, excluding 255 if it's in range)
            regular_classes = [i for i in range(1, num_classes) if i != 255]
            
            for idx, i in enumerate(regular_classes):
                # Simple rainbow distribution for regular classes
                if len(regular_classes) > 1:
                    hue = idx / (len(regular_classes) - 1)
                else:
                    hue = 0.5
                r = int(255 * (1 - hue))
                g = int(255 * hue)
                b = int(255 * (0.5 + 0.5 * abs(0.5 - hue)))
                colors[i] = (r, g, b)
            return colors
        
        # Use matplotlib colormap
        if num_classes <= 1:
            return {0: (0, 0, 0), 255: (255, 255, 255)}
        
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(colormap)
        except (ImportError, AttributeError):
            # Fallback if matplotlib colormap fails
            cmap = None
        
        colors = {}
        colors[0] = (0, 0, 0)          # Background always black
        colors[255] = (255, 255, 255)  # Class 255 always white
        
        # Handle regular classes (1 to num_classes-1, excluding 255 if it's in range)
        regular_classes = [i for i in range(1, num_classes) if i != 255]
        
        if cmap is not None:
            for idx, i in enumerate(regular_classes):
                if len(regular_classes) > 1:
                    color_val = cmap(idx / (len(regular_classes) - 1))
                else:
                    color_val = cmap(0.5)
                colors[i] = tuple(int(c * 255) for c in color_val[:3])
        else:
            # Fallback to simple colors if matplotlib fails
            for idx, i in enumerate(regular_classes):
                if len(regular_classes) > 1:
                    hue = idx / (len(regular_classes) - 1)
                else:
                    hue = 0.5
                r = int(255 * (1 - hue))
                g = int(255 * hue)
                b = int(255 * (0.5 + 0.5 * abs(0.5 - hue)))
                colors[i] = (r, g, b)
        
        return colors


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)
