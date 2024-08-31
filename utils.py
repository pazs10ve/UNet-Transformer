import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import zoom
from typing import Tuple


def resize_vol(file_path : str, target_volume : Tuple[float, float, float] = (96, 96, 96), is_label : bool = False) -> torch.Tensor:
    vol = nib.load(file_path)
    vol_data = vol.get_fdata()
    factors = np.array(target_volume) / np.array(vol_data.shape)
    order = 1 if is_label == False else 0
    resized_vol = zoom(vol_data, factors, order = order, mode = 'constant', cval = 0)
    return torch.Tensor(resized_vol)


def plot_pairs(image, label, n_samples = 5):
    indices = torch.randint(low = 0, high = image.shape[0], size = (n_samples, ))

    for idx in indices:
        plt.figure(figsize=(10, 20))

        plt.subplot(1, 2, 1)
        plt.imshow(image[:, :, idx])

        plt.subplot(1, 2, 2)
        plt.imshow(label[:, :, idx])

        plt.axis("off")
        plt.tight_layout()
        plt.show()



def get_transform():
    pass

def get_target_transform():
    pass

def get_transforms():
    pass