from loaders import get_loaders
from model import get_model
from train import train, plot_history
from losses import SoftDiceLoss
import torch
import torch.optim as optim

train_img_dir = r"Abdomen\RawData\Training\img"
train_label_dir = r"Abdomen\RawData\Training\label"
val_img_dir = r"Abdomen\RawData\Validation\img"
val_label_dir = r"Abdomen\RawData\Validation\label"
test_dir = r"Abdomen\RawData\Testing\img"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 6
in_channels = 1
out_channels = 1
img_size = (96, 96, 96)
patch_size = 16
embed_dim = 768
num_heads = 12
num_layers = 12
criterion = SoftDiceLoss()
lr = 0.0001
num_epochs = 20000

train_loader, val_loader, test_loader = get_loaders(train_img_dir, 
                train_label_dir, 
                val_img_dir, 
                val_label_dir, 
                test_dir,
                batch_size = batch_size,
                shuffle = False)

model = get_model(in_channels, out_channels, img_size, patch_size, embed_dim, num_heads, num_layers).to(device)

optimizer = optim.AdamW(params = model.parameters(), lr = lr)


if __name__ == "__main__":
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

