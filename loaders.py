import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from utils import resize_vol
from typing import Any

class BTCV(Dataset):
    def __init__(self, image_dir : str, 
                label_dir : str = None, 
                transform : v2.Transform = None,
                target_transform : v2.Transform = None) -> None:
        
        self.image_dir = image_dir
        self.label_dir = label_dir

        self.img_files = sorted([file for file in os.listdir(self.image_dir) if file.endswith(".nii.gz")])

        if self.label_dir:
            self.label_files = sorted([file for file in os.listdir(self.label_dir) if file.endswith(".nii.gz")])
            assert len(self.img_files) == len(self.label_files)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_file_path = os.path.join(self.image_dir, img_file)
        img = resize_vol(img_file_path)
        if self.transform:
            img = self.transform(img)
        img = img.unsqueeze(0)

        if self.label_dir:
            label_file = self.label_files[idx]  
            label_file_path = os.path.join(self.label_dir, label_file)
            label = resize_vol(label_file_path)


            if self.target_transform:
                label = self.target_transform(label)
            label = label.unsqueeze(0)

            return img, label
        return img
    


def get_loaders(train_img_dir : str, 
                train_labels_dir : str, 
                val_img_dir : str, 
                val_labels_dir : str, 
                test_img_dir : str,
                batch_size : int = 4,
                shuffle : bool = True,
                transform : v2.Transform = None, 
                target_transform : v2.Transform = None) -> Any:
    
    train_dataset = BTCV(train_img_dir, train_labels_dir, transform = transform, target_transform = target_transform)
    val_dataset = BTCV(val_img_dir, val_labels_dir, transform = transform, target_transform = target_transform)
    test_dataset = BTCV(test_img_dir, transform = transform, target_transform = target_transform)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = shuffle)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = shuffle)

    return train_loader, val_loader, test_loader


"""
train_img_dir = r"Abdomen\RawData\Training\img"
train_label_dir = r"Abdomen\RawData\Training\label"
train_set = BTCV(train_img_dir, train_label_dir)
print(len(train_set))
batch_size = 4
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
images, labels = next(iter(train_loader))
print(images.shape)
print(labels.shape)



val_img_dir = r"Abdomen\RawData\Validation\img"
val_label_dir = r"Abdomen\RawData\Validation\label"
val_set = BTCV(val_img_dir, val_label_dir)
print(len(val_set))
batch_size = 4
val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True)
images, labels = next(iter(val_loader))
print(images.shape)
print(labels.shape)



test_dir = r"Abdomen\RawData\Testing\img"
test_set = BTCV(test_dir)
print(len(test_set))
batch_size = 4
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)
images = next(iter(test_loader))
print(images.shape)
"""

"""
train_img_dir = r"Abdomen\RawData\Training\img"
train_label_dir = r"Abdomen\RawData\Training\label"
val_img_dir = r"Abdomen\RawData\Validation\img"
val_label_dir = r"Abdomen\RawData\Validation\label"
test_dir = r"Abdomen\RawData\Testing\img"


train_loader, val_loader, test_loader = get_loaders(train_img_dir, 
                train_label_dir, 
                val_img_dir, 
                val_label_dir, 
                test_dir,
                batch_size = 4,
                shuffle = True)


train_images, train_labels = next(iter(train_loader))
val_images, val_labels = next(iter(val_loader))
test_images = next(iter(test_loader))
print(len(train_loader), train_images.shape, train_labels.shape)
print(len(val_loader), val_images.shape, val_labels.shape)
print(len(test_loader), test_images.shape)

"""